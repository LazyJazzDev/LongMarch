#include "grassland/rhi/backend/metal/metal_command_context.h"

#include <array>
#include <cstring>
#include <list>
#include <mutex>
#include <system_error>
#include <unordered_map>

#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/metal/metal_command_context_impl.h"
#include "grassland/rhi/backend/metal/metal_compute_program.h"
#include "grassland/rhi/backend/metal/metal_diagnostics.h"
#include "grassland/rhi/backend/metal/metal_download_token.h"
#include "grassland/rhi/backend/metal/metal_program.h"
#include "grassland/rhi/backend/metal/metal_queue.h"
#include "grassland/rhi/backend/metal/metal_swap_chain_image.h"
#include "grassland/rhi/backend/metal/metal_util.h"
#include "grassland/rhi/backend/metal/metal_window.h"
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_command_context.h"
#include "grassland/rhi/rhi_image.h"
#include "grassland/rhi/rhi_program.h"
#include "grassland/rhi/rhi_queue.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

namespace grassland::rhi::backend::metal {

namespace {
MTL::ClearColor NativeClearColor(ImageFormat format, const ClearColorValue &value) {
  if (IsSignedIntegerFormat(format)) {
    const auto &c = value.int32;
    return MTL::ClearColor::Make(c[0], c[1], c[2], c[3]);
  }
  if (IsUnsignedIntegerFormat(format)) {
    const auto &c = value.uint32;
    return MTL::ClearColor::Make(c[0], c[1], c[2], c[3]);
  }
  const auto &c = value.float32;
  return MTL::ClearColor::Make(c[0], c[1], c[2], c[3]);
}
}  // namespace

MetalCommandContextImpl::MetalCommandContextImpl(IMetalQueue *queue) : device_(queue->Owner()) {
  InitializeCommandContext(queue);
}

ICommandContext *MetalCommandContextImpl::Create(IMetalQueue *queue) {
  auto object = std::unique_ptr<MetalCommandContextImpl>(new MetalCommandContextImpl(queue));
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  // Metal command buffers are single-use. The persistent native queue manages
  // their backing storage; completed buffers cannot be reset like Vulkan/D3D12.
  object->commands_ = queue->CreateCommandBuffer();
  if (!object->commands_) {
    ReportError(-1, "Metal command buffer creation failed.");
    return nullptr;
  }
  return object.release();
}

bool MetalCommandContextImpl::CmdClearImage(rhi::IImage *image, const ClearValue &value) {
  if (render_target_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before clear, present or submission.");
    return false;
  }
  const auto native = rhi::QueryInterfacePtr<IMetalImage>(image).Get();
  if (!native || !image->IsValid() || (native->DeviceIdentity() != device_ || !native->SupportsQueue(OwnerQueue()))) {
    recording_failed_ = true;
    ReportError(-1, "CmdClearImage requires a valid image from this device.");
    return false;
  }
  if (!RetainResource(native))
    return false;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto pass = MTL::RenderPassDescriptor::renderPassDescriptor();
  if (!pass) {
    recording_failed_ = true;
    return false;
  }
  const auto format = image->GetDesc().format;
  if (IsDepthFormat(format)) {
    auto depth = pass->depthAttachment();
    depth->setTexture(native->NativeImage());
    depth->setLoadAction(MTL::LoadActionClear);
    depth->setStoreAction(MTL::StoreActionStore);
    depth->setClearDepth(value.depth_stencil.depth);
    if (HasStencil(format)) {
      auto stencil = pass->stencilAttachment();
      stencil->setTexture(native->NativeImage());
      stencil->setLoadAction(MTL::LoadActionClear);
      stencil->setStoreAction(MTL::StoreActionStore);
      stencil->setClearStencil(value.depth_stencil.stencil);
    }
  } else {
    auto attachment = pass->colorAttachments()->object(0);
    attachment->setTexture(native->NativeImage());
    attachment->setLoadAction(MTL::LoadActionClear);
    attachment->setStoreAction(MTL::StoreActionStore);
    attachment->setClearColor(NativeClearColor(format, value.color));
  }
  auto encoder = commands_->renderCommandEncoder(pass);
  if (!encoder) {
    recording_failed_ = true;
    ReportError(-1, "Metal clear render encoder creation failed.");
    return false;
  }
  // Each clear is encoded now, with independent values, even for interleaved images.
  encoder->endEncoding();
  return true;
}

bool MetalCommandContextImpl::CmdPresent(rhi::IWindow *window) {
  if (render_target_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before clear, present or submission.");
    return false;
  }
  const auto target = rhi::QueryInterfacePtr<IMetalWindow>(window).Get();
  if (!target || target->DeviceIdentity() != device_ || target->OwnerQueue() != OwnerQueue() ||
      !glfw::IsWindowThread()) {
    recording_failed_ = true;
    ReportError(-1, "CmdPresent requires a window from this device on the window thread.");
    return false;
  }
  for (const auto &present : presents_) {
    if (present.first == target) {
      recording_failed_ = true;
      ReportError(-1, "A context can present each window only once.");
      return false;
    }
  }
  ObjPtr<IImage> frame_image = window->GetNextImage();
  auto frame = rhi::QueryInterfacePtr<IMetalSwapChainImage>(frame_image.Get());
  if (!frame || !RetainResource(frame.Get()))
    return false;

  if (!target->RecordImGui(this, frame_image.Get())) {
    recording_failed_ = true;
    return false;
  }
  presents_.emplace_back(target, frame.Get());
  return true;
}

bool MetalCommandContextImpl::EndRecording() {
#if !defined(NDEBUG)
  if (render_target_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before clear, present or submission.");
    return false;
  }
#endif
  // All encoders end at their Cmd call. Metal tracks texture hazards and layouts
  // automatically; there is no persistent layout to restore between contexts.
  return true;
}

IMetalCommandContext::SubmitResult MetalCommandContextImpl::SubmitNative(const SubmitInfo &info, Event signal) {
#if !defined(NDEBUG)
  if (!presents_.empty() && !glfw::IsWindowThread()) {
    ReportError(-1, "Present must run on the window thread.");
    return SUBMIT_RESULT_REJECTED;
  }
  for (const auto &present : presents_) {
    if (!rhi::QueryInterfacePtr<IImage>(present.second)->IsValid())
      return SUBMIT_RESULT_REJECTED;
  }
#endif
  {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    if (!info.waits.empty()) {
      auto preamble = queue_->CreateCommandBuffer();
      if (!preamble)
        return SUBMIT_RESULT_REJECTED;
      for (const auto &wait : info.waits)
        preamble->encodeWait(rhi::QueryInterfacePtr<IMetalTimeline>(wait.timeline).Get()->NativeEvent(), wait.value);
      preamble->commit();
    }
    commands_->encodeSignalEvent(rhi::QueryInterfacePtr<IMetalTimeline>(signal.timeline).Get()->NativeEvent(),
                                 signal.value);
    for (const auto &present : presents_)
      commands_->presentDrawable(present.second->NativeDrawable());
    device_->ObserveCompletion(commands_.get(), signal);
    commands_->commit();
    MarkSubmitted();
    for (auto &entry : as_dependencies_)
      entry.first->CommitDependencies(std::move(entry.second));
    as_dependencies_.clear();
    for (const auto &present : presents_)
      present.second->DidPresent();
  }
  return SUBMIT_RESULT_SUBMITTED;
}

void MetalCommandContextImpl::UpdatePresentedImages() {
  for (const auto &present : presents_) {
    if (!present.first->UpdateNextImage()) {
      ReportError(-1, "Failed to prepare the next Metal drawable after submission.");
    }
  }
}

SubmissionStatus MetalCommandContextImpl::PollCompletion(Event signal, uint64_t completed, bool timeline_valid) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  if (commands_->status() == MTL::CommandBufferStatusError) {
    rhi::QueryInterfacePtr<IMetalTimeline>(signal.timeline).Get()->MarkFailed();
    device_->NotifyTimelineFailure();
    ReportMetalError(commands_->error(), "Metal command buffer failed",
                     commands_->errorOptions() & MTL::CommandBufferErrorOptionEncoderExecutionStatus);
    return SUBMISSION_STATUS_FAILED;
  }
  if (commands_->status() != MTL::CommandBufferStatusCompleted)
    return SUBMISSION_STATUS_PENDING;
  if (!timeline_valid)
    return SUBMISSION_STATUS_FAILED;
  if (completed < signal.value)
    return SUBMISSION_STATUS_PENDING;
  return SUBMISSION_STATUS_COMPLETED;
}

MetalCommandContextImpl::~MetalCommandContextImpl() {
  if (encoder_)
    encoder_->endEncoding();
}

bool MetalCommandContextImpl::CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) {
  auto native = rhi::QueryInterfacePtr<IMetalBuffer>(buffer).Get();
  if (!native || native->Owner() != device_ || sealed_ || render_target_) {
    recording_failed_ = true;
    return false;
  }
  if (!data || !size || offset > buffer->GetDesc().size || size > buffer->GetDesc().size - offset) {
    recording_failed_ = true;
    return false;
  }
  auto upload = device_->UploadPool().Acquire(size);
  if (!upload || !upload->Write(data, size)) {
    recording_failed_ = true;
    return false;
  }
  return CopyBuffer(upload.Get(), 0, native, offset, size);
}

bool MetalCommandContextImpl::CmdCopyBuffer(IBuffer *source,
                                            uint64_t source_offset,
                                            IBuffer *destination,
                                            uint64_t destination_offset,
                                            uint64_t size) {
  auto src = rhi::QueryInterfacePtr<IMetalBuffer>(source).Get();
  auto dst = rhi::QueryInterfacePtr<IMetalBuffer>(destination).Get();
  if (!src || !dst || src->Owner() != device_ || dst->Owner() != device_ || sealed_ || render_target_) {
    recording_failed_ = true;
    return false;
  }
  if (src == dst || !size || source_offset > source->GetDesc().size || size > source->GetDesc().size - source_offset ||
      destination_offset > destination->GetDesc().size || size > destination->GetDesc().size - destination_offset) {
    recording_failed_ = true;
    return false;
  }
  return CopyBuffer(src, source_offset, dst, destination_offset, size);
}

bool MetalCommandContextImpl::CopyBuffer(IMetalBuffer *src,
                                         uint64_t source_offset,
                                         IMetalBuffer *dst,
                                         uint64_t destination_offset,
                                         uint64_t size) {
  if (!RetainResource(src) || !RetainResource(dst))
    return false;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto blit = commands_->blitCommandEncoder();
  if (!blit) {
    recording_failed_ = true;
    return false;
  }
  blit->copyFromBuffer(src->NativeBuffer(), source_offset, dst->NativeBuffer(), destination_offset, size);
  blit->endEncoding();
  return true;
}

bool MetalCommandContextImpl::CmdUploadImage(IImage *image, const void *data, uint64_t row_pitch) {
  auto native = rhi::QueryInterfacePtr<IMetalImage>(image).Get();
  const uint64_t pitch = native ? uint64_t(image->GetDesc().width) * ImageFormatSize(image->GetDesc().format) : 0;
  if (sealed_ || encoder_ || !native || !image->IsValid() ||
      (native->DeviceIdentity() != device_ || !native->SupportsQueue(OwnerQueue())) || !data ||
      !IsColorFormat(image->GetDesc().format) || !pitch || row_pitch < pitch ||
      (image->GetDesc().height - 1) > (SIZE_MAX - pitch) / row_pitch) {
    recording_failed_ = true;
    return false;
  }

  const auto desc = image->GetDesc();
  const uint64_t size = pitch * desc.height;
  auto upload = device_->UploadPool().Acquire(size);
  if (!upload || !RetainResource(upload.Get()) || !RetainResource(native)) {
    recording_failed_ = true;
    return false;
  }
  // The lease is shared staging memory; snapshot rows directly into it.
  auto destination = static_cast<uint8_t *>(upload->NativeBuffer()->contents());
  for (uint32_t y = 0; y < desc.height; ++y)
    std::memcpy(destination + y * pitch, static_cast<const uint8_t *>(data) + y * row_pitch, pitch);
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto blit = commands_->blitCommandEncoder();
  if (!blit) {
    recording_failed_ = true;
    return false;
  }
  blit->copyFromBuffer(upload->NativeBuffer(), 0, pitch, size, MTL::Size::Make(desc.width, desc.height, 1),
                       native->NativeImage(), 0, 0, MTL::Origin::Make(0, 0, 0));
  blit->endEncoding();
  return true;
}

bool MetalCommandContextImpl::CmdBindGroup(uint32_t index, IBindingGroup *group) {
  auto native = rhi::QueryInterfacePtr<IMetalBindingGroup>(group).Get();
  if (sealed_ || !native_program_ || !native || native->Owner() != device_ || index >= groups_.size() ||
      !SameBindingLayout(group->GetLayout(), native_program_->BindingLayout().groups[index]) ||
      !native->CurrentVersion()->IsComplete()) {
    recording_failed_ = true;
    return false;
  }
  auto version = native->CurrentVersion();
  for (size_t binding = 0; binding < version->GetLayout().bindings.size(); ++binding) {
    const auto type = version->GetLayout().bindings[binding].type;
    if (type == BINDING_TEXTURE2D || type == BINDING_RW_TEXTURE2D) {
      for (const auto &image : version->GetReferences()[binding]) {
        if (!rhi::QueryInterfacePtr<IMetalImage>(image.Get()).Get()->SupportsQueue(OwnerQueue())) {
          recording_failed_ = true;
          return false;
        }
      }
    }
  }
  if (!RetainResource(version))
    return false;
  groups_[index] = version;
  if (encoder_)
    version->Bind(encoder_.get(), index);
  return true;
}

bool MetalCommandContextImpl::CmdBindComputeProgram(IComputeProgram *program) {
  auto native = rhi::QueryInterfacePtr<IMetalComputeProgram>(program).Get();
  if (sealed_ || encoder_ || !native || native->Owner() != device_) {
    recording_failed_ = true;
    return false;
  }
  if (!RetainResource(native))
    return false;
  groups_.assign(native->BindingLayout().groups.size(), nullptr);
  compute_program_ = native;
  native_program_ = native;
  program_ = nullptr;
  return true;
}

bool MetalCommandContextImpl::CmdDispatch(uint32_t x, uint32_t y, uint32_t z) {
  if (sealed_ || encoder_ || !compute_program_ || recording_failed_) {
    recording_failed_ = true;
    return false;
  }
  for (size_t i = 0; i < groups_.size(); ++i)
    if (!groups_[i] && !native_program_->BindingLayout().groups[i].bindings.empty()) {
      recording_failed_ = true;
      return false;
    }
  if (!x || !y || !z)
    return true;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto acceleration_resources = AccelerationResources();
  auto encoder = commands_->computeCommandEncoder();
  if (!encoder) {
    recording_failed_ = true;
    return false;
  }
  for (auto resource : acceleration_resources)
    encoder->useResource(resource, MTL::ResourceUsageRead);
  encoder->setComputePipelineState(compute_program_->NativePipeline());
  for (uint32_t i = 0; i < groups_.size(); ++i)
    if (groups_[i])
      groups_[i]->Bind(encoder, i);
  encoder->dispatchThreadgroups(MTL::Size::Make(x, y, z), compute_program_->ThreadGroupSize());
  // Separate serial encoders plus tracked resource residency synchronize dependent
  // dispatches, blits and rendering, including resources in argument buffers.
  encoder->endEncoding();
  return true;
}

bool MetalCommandContextImpl::CmdBindProgram(IProgram *program) {
  auto native = rhi::QueryInterfacePtr<IMetalProgram>(program).Get();
  if (!native || native->Owner() != device_ || sealed_) {
    recording_failed_ = true;
    return false;
  }
  if (!RetainResource(native))
    return false;

  vertex_buffers_.assign(program->GetDesc().vertex_bindings.size(), {});
  groups_.assign(program->GetDesc().pipeline_layout.groups.size(), nullptr);

  program_ = program;
  compute_program_ = nullptr;
  native_program_ = native;
  if (encoder_)
    ApplyBindings();
  return true;
}

void MetalCommandContextImpl::ApplyBindings() {
  if (program_) {
    auto native = rhi::QueryInterfacePtr<IMetalProgram>(program_);
    encoder_->setRenderPipelineState(native->NativePipeline());
    const auto &raster = program_->GetDesc().rasterizer;
    const MTL::CullMode cull_modes[] = {MTL::CullModeNone, MTL::CullModeFront, MTL::CullModeBack};
    encoder_->setCullMode(cull_modes[raster.cull_mode]);
    encoder_->setFrontFacingWinding(raster.front_face == FRONT_FACE_COUNTER_CLOCKWISE ? MTL::WindingCounterClockwise
                                                                                      : MTL::WindingClockwise);
    encoder_->setTriangleFillMode(raster.fill_mode == FILL_MODE_WIREFRAME ? MTL::TriangleFillModeLines
                                                                          : MTL::TriangleFillModeFill);
    encoder_->setDepthStencilState(native->NativeDepthStencil());
    encoder_->setStencilReferenceValue(program_->GetDesc().depth_stencil.stencil_reference);
  }
  for (uint32_t i = 0; i < groups_.size(); ++i)
    if (groups_[i])
      groups_[i]->Bind(encoder_.get(), i);
  for (size_t i = 0; i < vertex_buffers_.size(); ++i)
    if (vertex_buffers_[i].first)
      encoder_->setVertexBuffer(vertex_buffers_[i].first->NativeBuffer(), vertex_buffers_[i].second,
                                i + IMetalProgram::VertexBufferBase);
  if (viewport_set_)
    encoder_->setViewport(MTL::Viewport{viewport_.x, viewport_.y, viewport_.width, viewport_.height,
                                        viewport_.min_depth, viewport_.max_depth});
  if (scissor_set_)
    ApplyScissor();
}

void MetalCommandContextImpl::ApplyScissor() {
  // Metal requires the native scissor to fit the current attachment. Preserve
  // the API state and intersect it again when switching to a different target.
  const auto desc = render_target_->GetDesc();
  const auto x = std::min<uint64_t>(scissor_.x, desc.width);
  const auto y = std::min<uint64_t>(scissor_.y, desc.height);
  const auto right = std::min<uint64_t>(uint64_t(scissor_.x) + scissor_.width, desc.width);
  const auto bottom = std::min<uint64_t>(uint64_t(scissor_.y) + scissor_.height, desc.height);
  encoder_->setScissorRect(MTL::ScissorRect{x, y, right - x, bottom - y});
}

bool MetalCommandContextImpl::CmdBeginRendering(const RenderingInfo &info) {
  if (sealed_ || encoder_ || info.color_attachments.size() > 8 ||
      (info.color_attachments.empty() && !info.depth_stencil.image)) {
    recording_failed_ = true;
    return false;
  }
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto pass = MTL::RenderPassDescriptor::renderPassDescriptor();
  IImage *target = nullptr;
  uint32_t width = 0, height = 0;
  auto validate = [&](IImage *image, AttachmentLoadOp load, AttachmentStoreOp store) -> IMetalImage * {
    auto value = rhi::QueryInterfacePtr<IMetalImage>(image).Get();
    if (!value || !image->IsValid() || (value->DeviceIdentity() != device_ || !value->SupportsQueue(OwnerQueue())) ||
        load < ATTACHMENT_LOAD_OP_LOAD || load > ATTACHMENT_LOAD_OP_DISCARD || store < ATTACHMENT_STORE_OP_STORE ||
        store > ATTACHMENT_STORE_OP_DISCARD)
      return nullptr;
    auto desc = image->GetDesc();
    if (width && (width != desc.width || height != desc.height))
      return nullptr;
    width = desc.width;
    height = desc.height;
    return RetainResource(value) ? value : nullptr;
  };
  const MTL::LoadAction loads[] = {MTL::LoadActionLoad, MTL::LoadActionClear, MTL::LoadActionDontCare};
  auto store = [](AttachmentStoreOp op) {
    return op == ATTACHMENT_STORE_OP_STORE ? MTL::StoreActionStore : MTL::StoreActionDontCare;
  };

  color_formats_.clear();
  color_targets_.clear();
  for (size_t i = 0; i < info.color_attachments.size(); ++i) {
    auto &a = info.color_attachments[i];
    auto image = validate(a.color_attachment, a.load_op, a.store_op);
    if (!image || IsDepthFormat(a.color_attachment->GetDesc().format)) {
      recording_failed_ = true;
      return false;
    }
    for (size_t j = 0; j < i; ++j)
      if (info.color_attachments[j].color_attachment == a.color_attachment) {
        recording_failed_ = true;
        return false;
      }
    target = a.color_attachment;
    color_formats_.push_back(a.color_attachment->GetDesc().format);
    color_targets_.push_back(image);
    auto color = pass->colorAttachments()->object(i);
    color->setTexture(image->NativeImage());
    color->setLoadAction(loads[a.load_op]);
    color->setStoreAction(store(a.store_op));
    if (a.load_op == ATTACHMENT_LOAD_OP_CLEAR)
      color->setClearColor(NativeClearColor(a.color_attachment->GetDesc().format, a.clear_value));
  }

  depth_format_ = IMAGE_FORMAT_UNKNOWN;
  const auto &ds = info.depth_stencil;
  if (ds.image) {
    auto image = validate(ds.image, ds.depth_load_op, ds.depth_store_op);
    if (!image || !IsDepthFormat(ds.image->GetDesc().format) || ds.stencil_load_op < ATTACHMENT_LOAD_OP_LOAD ||
        ds.stencil_load_op > ATTACHMENT_LOAD_OP_DISCARD || ds.stencil_store_op < ATTACHMENT_STORE_OP_STORE ||
        ds.stencil_store_op > ATTACHMENT_STORE_OP_DISCARD) {
      recording_failed_ = true;
      return false;
    }
    target = ds.image;
    depth_format_ = ds.image->GetDesc().format;
    auto depth = pass->depthAttachment();
    depth->setTexture(image->NativeImage());
    depth->setLoadAction(loads[ds.depth_load_op]);
    depth->setStoreAction(store(ds.depth_store_op));
    depth->setClearDepth(ds.clear_value.depth);
    if (HasStencil(depth_format_)) {
      auto stencil = pass->stencilAttachment();
      stencil->setTexture(image->NativeImage());
      stencil->setLoadAction(loads[ds.stencil_load_op]);
      stencil->setStoreAction(store(ds.stencil_store_op));
      stencil->setClearStencil(ds.clear_value.stencil);
    }
  }
  encoder_ = NS::RetainPtr(commands_->renderCommandEncoder(pass));
  if (!encoder_) {
    recording_failed_ = true;
    return false;
  }
  render_target_ = target;
  ApplyBindings();
  return true;
}

bool MetalCommandContextImpl::CmdEndRendering() {
#if !defined(NDEBUG)
  if (sealed_ || !render_target_) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (recording_failed_)
    return false;
  encoder_->endEncoding();
  encoder_.reset();
  render_target_ = nullptr;
  color_targets_.clear();
  rendering_ = {};
  return true;
}

bool MetalCommandContextImpl::CmdBindVertexBuffer(uint32_t binding, IBuffer *buffer, uint64_t offset) {
  auto native = rhi::QueryInterfacePtr<IMetalBuffer>(buffer).Get();
  if (!native || native->Owner() != device_ || sealed_) {
    recording_failed_ = true;
    return false;
  }
  if (!RetainResource(native))
    return false;
  if (!program_ || binding >= vertex_buffers_.size() || offset >= buffer->GetDesc().size) {
    recording_failed_ = true;
    return false;
  }
  vertex_buffers_[binding] = {native, offset};
  if (encoder_)
    encoder_->setVertexBuffer(native->NativeBuffer(), offset, binding + IMetalProgram::VertexBufferBase);
  return true;
}

bool MetalCommandContextImpl::CmdBindIndexBuffer(IBuffer *buffer, IndexFormat format, uint64_t offset) {
  auto native = rhi::QueryInterfacePtr<IMetalBuffer>(buffer).Get();
  if (!native || native->Owner() != device_ || sealed_) {
    recording_failed_ = true;
    return false;
  }
  if (!RetainResource(native))
    return false;
  if ((format != INDEX_FORMAT_UINT16 && format != INDEX_FORMAT_UINT32) || offset >= buffer->GetDesc().size ||
      offset % (format == INDEX_FORMAT_UINT16 ? 2 : 4)) {
    recording_failed_ = true;
    return false;
  }
  index_ = native;
  index_offset_ = offset;
  index_format_ = format;
  return true;
}

bool MetalCommandContextImpl::CmdSetViewport(const Viewport &viewport) {
#if !defined(NDEBUG)
  if (sealed_ || viewport.width <= 0 || viewport.height <= 0) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (recording_failed_)
    return false;
  viewport_ = viewport;
  viewport_set_ = true;
  if (encoder_)
    encoder_->setViewport(
        MTL::Viewport{viewport.x, viewport.y, viewport.width, viewport.height, viewport.min_depth, viewport.max_depth});
  return true;
}

bool MetalCommandContextImpl::CmdSetScissor(const Scissor &scissor) {
#if !defined(NDEBUG)
  if (sealed_ || !scissor.width || !scissor.height || scissor.x < 0 || scissor.y < 0) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (recording_failed_)
    return false;
  scissor_ = scissor;
  scissor_set_ = true;
  if (encoder_)
    ApplyScissor();
  return true;
}

bool MetalCommandContextImpl::ValidateDraw(bool indexed) {
#if !defined(NDEBUG)
  if (sealed_ || !program_ || !render_target_ || !viewport_set_ || !scissor_set_ || (indexed && !index_)) {
    recording_failed_ = true;
    return false;
  }
  if (program_->GetDesc().color_formats != color_formats_ ||
      program_->GetDesc().depth_stencil.format != depth_format_ ||
      std::any_of(vertex_buffers_.begin(), vertex_buffers_.end(), [](const auto &binding) { return !binding.first; })) {
    recording_failed_ = true;
    return false;
  }
#endif
  for (size_t i = 0; i < groups_.size(); ++i) {
    if (!groups_[i] && !program_->GetDesc().pipeline_layout.groups[i].bindings.empty()) {
      recording_failed_ = true;
      return false;
    }
    if (groups_[i]) {
      auto &layout = groups_[i]->GetLayout();
      for (size_t b = 0; b < layout.bindings.size(); ++b) {
        auto type = layout.bindings[b].type;
        if (type != BINDING_TEXTURE2D && type != BINDING_RW_TEXTURE2D)
          continue;
        for (auto &image : groups_[i]->GetReferences()[b])
          for (auto target : color_targets_)
            if (image.Get() == target) {
              recording_failed_ = true;
              return false;
            }
      }
    }
  }
  for (auto resource : AccelerationResources())
    encoder_->useResource(resource, MTL::ResourceUsageRead, MTL::RenderStageVertex | MTL::RenderStageFragment);
  return !recording_failed_;
}

bool MetalCommandContextImpl::CmdDraw(uint32_t vertex_count,
                                      uint32_t instance_count,
                                      uint32_t first_vertex,
                                      uint32_t first_instance) {
  if (!ValidateDraw(false))
    return false;
  encoder_->drawPrimitives(NativePrimitiveTopology(program_->GetDesc().topology), first_vertex, vertex_count,
                           instance_count, first_instance);
  return true;
}

bool MetalCommandContextImpl::CmdDrawIndexed(uint32_t index_count,
                                             uint32_t instance_count,
                                             uint32_t first_index,
                                             int32_t vertex_offset,
                                             uint32_t first_instance) {
  if (!ValidateDraw(true))
    return false;
  encoder_->drawIndexedPrimitives(
      NativePrimitiveTopology(program_->GetDesc().topology), index_count,
      index_format_ == INDEX_FORMAT_UINT16 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32, index_->NativeBuffer(),
      index_offset_ + uint64_t(first_index) * (index_format_ == INDEX_FORMAT_UINT16 ? 2 : 4), instance_count,
      vertex_offset, first_instance);
  return true;
}

IDownloadToken *MetalCommandContextImpl::CmdDownloadBuffer(IBuffer *buffer, uint64_t offset, uint64_t size) {
  auto native = rhi::QueryInterfacePtr<IMetalBuffer>(buffer).Get();
  if (sealed_ || render_target_ || !native || native->Owner() != device_) {
    recording_failed_ = true;
    return nullptr;
  }
  if (recording_failed_ || !size || offset > buffer->GetDesc().size || size > buffer->GetDesc().size - offset) {
    recording_failed_ = true;
    return nullptr;
  }
  auto lease = device_->DownloadPool().Acquire(size);
  if (!lease || !CopyBuffer(native, offset, lease.Get(), 0, size)) {
    recording_failed_ = true;
    return nullptr;
  }
  return RegisterDownload(ObjectPtr<IMetalDownloadToken>::Adopt(CreateMetalDownloadToken(std::move(lease), size)));
}

IDownloadToken *MetalCommandContextImpl::CmdDownloadImage(IImage *image, const ImageDownloadRegion &region) {
  auto native = rhi::QueryInterfacePtr<IMetalImage>(image).Get();
  if (sealed_ || render_target_ || !native ||
      (native->DeviceIdentity() != device_ || !native->SupportsQueue(OwnerQueue()))) {
    recording_failed_ = true;
    return nullptr;
  }
  if (image->GetDesc().format == IMAGE_FORMAT_D24_UNORM_S8_UINT) {
    recording_failed_ = true;
    ReportError(-1, "CmdDownloadImage does not support D24S8 readback; use a depth-only or color image.");
    return nullptr;
  }
  if (recording_failed_ || !image->IsValid()) {
    recording_failed_ = true;
    return nullptr;
  }
  const auto desc = image->GetDesc();
  if (region.x >= desc.width || region.y >= desc.height) {
    recording_failed_ = true;
    return nullptr;
  }
  const uint32_t width = region.width ? region.width : desc.width - region.x;
  const uint32_t height = region.height ? region.height : desc.height - region.y;
  const uint64_t packed = uint64_t(width) * ImageFormatSize(desc.format);
  if (width > desc.width - region.x || height > desc.height - region.y || !packed || height > SIZE_MAX / packed) {
    recording_failed_ = true;
    return nullptr;
  }
  const uint64_t pitch = packed;
  const bool stencil = HasStencil(desc.format);
  auto lease = device_->DownloadPool().Acquire(pitch * height + (stencil ? uint64_t(width) * height : 0));
  if (!lease || !RetainResource(native) || !RetainResource(lease.Get())) {
    recording_failed_ = true;
    return nullptr;
  }
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto blit = commands_->blitCommandEncoder();
  if (!blit) {
    recording_failed_ = true;
    return nullptr;
  }
  if (stencil) {
    blit->copyFromTexture(native->NativeImage(), 0, 0, MTL::Origin::Make(region.x, region.y, 0),
                          MTL::Size::Make(width, height, 1), lease->NativeBuffer(), 0, pitch, pitch * height,
                          MTL::BlitOptionDepthFromDepthStencil);
    blit->copyFromTexture(native->NativeImage(), 0, 0, MTL::Origin::Make(region.x, region.y, 0),
                          MTL::Size::Make(width, height, 1), lease->NativeBuffer(), pitch * height, width,
                          uint64_t(width) * height, MTL::BlitOptionStencilFromDepthStencil);
  } else {
    blit->copyFromTexture(native->NativeImage(), 0, 0, MTL::Origin::Make(region.x, region.y, 0),
                          MTL::Size::Make(width, height, 1), lease->NativeBuffer(), 0, pitch, pitch * height);
  }
  blit->endEncoding();
  return RegisterDownload(ObjectPtr<IMetalDownloadToken>::Adopt(CreateMetalDownloadToken(
      std::move(lease), DownloadLayout{0, pitch, packed, height, stencil ? pitch * height : 0, stencil ? width : 0})));
}

ICommandContext *CreateMetalCommandContext(IMetalQueue *queue) {
  return MetalCommandContextImpl::Create(queue);
}

bool MetalCommandContextImpl::CmdDrawImGui(IWindow *window, IImage *target) {
  auto native = rhi::QueryInterfacePtr<IMetalWindow>(window).Get();
  auto image = rhi::QueryInterfacePtr<IMetalImage>(target).Get();
  if (!native || !image || native->DeviceIdentity() != DeviceIdentity() ||
      (image->DeviceIdentity() != DeviceIdentity() || !image->SupportsQueue(OwnerQueue())) || !glfw::IsWindowThread() ||
      sealed_ || !rendering_.color_attachments.empty() || rendering_.depth_stencil.image) {
    recording_failed_ = true;
    ReportError(-1,
                "CmdDrawImGui requires a window and target from this device outside rendering on the window thread.");
    return false;
  }
  const auto format = target->GetDesc().format;
  if (!IsColorFormat(format) || IsSignedIntegerFormat(format) || IsUnsignedIntegerFormat(format)) {
    recording_failed_ = true;
    ReportError(-1, "CmdDrawImGui requires a non-integer color target.");
    return false;
  }
  if (!RetainResource(target))
    return false;
  if (!native->RecordImGui(this, target, COLOR_SPACE_BT709_LINEAR, 1.0f)) {
    recording_failed_ = true;
    return false;
  }
  return true;
}

bool MetalCommandContextImpl::CmdPresentImage(IWindow *window, IImage *frame_img, ColorSpace frame_color_space) {
  auto native = rhi::QueryInterfacePtr<IMetalWindow>(window).Get();
  auto image = rhi::QueryInterfacePtr<IMetalImage>(frame_img).Get();
  if (!native || !image || native->DeviceIdentity() != DeviceIdentity() ||
      (image->DeviceIdentity() != DeviceIdentity() || !image->SupportsQueue(OwnerQueue())) || !glfw::IsWindowThread() ||
      sealed_ || !rendering_.color_attachments.empty() || rendering_.depth_stencil.image) {
    recording_failed_ = true;
    ReportError(-1,
                "CmdPresentImage requires a window and image from this device outside rendering on the window thread.");
    return false;
  }
  if (!RetainResource(frame_img))
    return false;
  if (!native->RecordPresentImage(this, frame_img, frame_color_space)) {
    recording_failed_ = true;
    return false;
  }
  return CmdPresent(window);
}

bool MetalCommandContextImpl::CmdBindRayTracingProgram(IRayTracingProgram *) {
  recording_failed_ = true;
  return false;
}

bool MetalCommandContextImpl::CmdDispatchRays(uint32_t, uint32_t, uint32_t, IShaderBindingTable *) {
  recording_failed_ = true;
  return false;
}

bool MetalCommandContextImpl::PrepareRendering(const RenderingInfo &info) {
  auto next = info;
  if (next.color_attachments.size() > 8 || (next.color_attachments.empty() && !next.depth_stencil.image)) {
    recording_failed_ = true;
    return false;
  }
  std::vector<IImage *> images;
  auto validate = [&](IImage *image, bool depth, AttachmentLoadOp load, AttachmentStoreOp store) {
    auto native = rhi::QueryInterfacePtr<IMetalImage>(image).Get();
    if (!native || !image->IsValid() ||
        (native->DeviceIdentity() != DeviceIdentity() || !native->SupportsQueue(OwnerQueue())) ||
        IsDepthFormat(image->GetDesc().format) != depth || load < ATTACHMENT_LOAD_OP_LOAD ||
        load > ATTACHMENT_LOAD_OP_DISCARD || store < ATTACHMENT_STORE_OP_STORE || store > ATTACHMENT_STORE_OP_DISCARD)
      return false;
    for (auto previous : images)
      if (previous == image || previous->GetDesc().width != image->GetDesc().width ||
          previous->GetDesc().height != image->GetDesc().height)
        return false;
    images.push_back(image);
    return RetainResource(image);
  };
  for (auto color : next.color_attachments)
    if (!validate(color.color_attachment, false, color.load_op, color.store_op)) {
      recording_failed_ = true;
      return false;
    }
  const auto &ds = next.depth_stencil;
  if (ds.image &&
      (!validate(ds.image, true, ds.depth_load_op, ds.depth_store_op) || ds.stencil_load_op < ATTACHMENT_LOAD_OP_LOAD ||
       ds.stencil_load_op > ATTACHMENT_LOAD_OP_DISCARD || ds.stencil_store_op < ATTACHMENT_STORE_OP_STORE ||
       ds.stencil_store_op > ATTACHMENT_STORE_OP_DISCARD ||
       !(ds.clear_value.depth >= 0 && ds.clear_value.depth <= 1))) {
    recording_failed_ = true;
    return false;
  }
  rendering_ = std::move(next);
  return true;
}

bool MetalCommandContextImpl::RenderingMatches(const IProgram *program) const {
  const auto &desc = program->GetDesc();
  if (desc.color_formats.size() != rendering_.color_attachments.size())
    return false;
  for (size_t i = 0; i < desc.color_formats.size(); ++i)
    if (desc.color_formats[i] != rendering_.color_attachments[i].color_attachment->GetDesc().format)
      return false;
  return desc.depth_stencil.format ==
         (rendering_.depth_stencil.image ? rendering_.depth_stencil.image->GetDesc().format : IMAGE_FORMAT_UNKNOWN);
}

void MetalCommandContextImpl::InitializeCommandContext(IMetalQueue *queue) {
  queue_ = queue;
}

rhi::ISubmissionToken *MetalCommandContextImpl::Submit(const rhi::SubmitInfo &info) {
  try {
    auto submission = queue_->Submit(ObjectPtr<IMetalCommandContext>::Retain(this), info);
    return rhi::QueryInterfacePtr<ISubmissionToken>(submission.Get()).Detach();
  } catch (const std::system_error &error) {
    ReportError(error.code().value(), error.what());
  }
  return nullptr;
}

void MetalCommandContextImpl::BindDownloads(rhi::ISubmissionToken *submission) {
  for (const auto &token : downloads_)
    token->BindSubmission(submission);
}

void MetalCommandContextImpl::ReleaseDownloads() {
  downloads_.clear();
}

IDownloadToken *MetalCommandContextImpl::RegisterDownload(ObjectPtr<IMetalDownloadToken> token) {
  downloads_.push_back(token);
  return rhi::QueryInterfacePtr<IDownloadToken>(token.Get()).Detach();
}

bool MetalCommandContextImpl::RetainResource(rhi::IObject *resource) {
#if !defined(NDEBUG)
  if (sealed_)
    return false;
#endif
  if (recording_failed_)
    return false;
  if (resources_.find(resource) != resources_.end())
    return true;
  resources_.emplace(resource, ObjectPtr<rhi::IObject>::Retain(resource));

  return true;
}

bool MetalCommandContextImpl::Seal() {
  if (recording_failed_)
    return false;
#if !defined(NDEBUG)
  if (sealed_) {
    ReportError(-1, "Cannot submit a command context twice.");
    return false;
  }
#endif
  sealed_ = true;
  if (!EndRecording()) {
    recording_failed_ = true;
    return false;
  }
  return true;
}

IMetalQueue *MetalCommandContextImpl::OwnerQueue() const {
  return queue_;
}

const void *MetalCommandContextImpl::DeviceIdentity() const {
  return queue_->Owner();
}

void MetalCommandContextImpl::MarkSubmitted() {
  submitted_ = true;
}

bool MetalCommandContextImpl::WasSubmitted() const {
  return submitted_;
}
}  // namespace grassland::rhi::backend::metal
