#include "grassland/rhi/backend/metal/metal_command_context.h"

#include <cstring>

#include "grassland/rhi/backend/metal/metal_compute_program.h"
#include "grassland/rhi/backend/metal/metal_program.h"
#include "grassland/rhi/backend/metal/metal_swap_chain_image.h"
#include "grassland/rhi/backend/metal/metal_window.h"

namespace grassland::rhi::backend::metal {

MetalCommandContext::MetalCommandContext(MetalDevice *device) : CommandContext(device), device_(device) {
}

MetalCommandContext *MetalCommandContext::Create(MetalDevice *device) {
  auto object = std::unique_ptr<MetalCommandContext>(new MetalCommandContext(device));
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  // Metal command buffers are single-use. The persistent native queue manages
  // their backing storage; completed buffers cannot be reset like Vulkan/D3D12.
  object->commands_ = NS::RetainPtr(object->device_->NativeQueue()->commandBuffer());
  if (!object->commands_) {
    ReportError(-1, "Metal command buffer creation failed.");
    return nullptr;
  }
  return object.release();
}

bool MetalCommandContext::CmdClearColor(rhi::IImage *image, const std::array<float, 4> &color) {
#if !defined(NDEBUG)
  if (render_target_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before clear, present or submission.");
    return false;
  }
  const auto native = dynamic_cast<MetalImage *>(image);
  if (!native || !native->IsValid() || native->DeviceIdentity() != device_ || IsDepthFormat(native->GetDesc().format)) {
    recording_failed_ = true;
    ReportError(-1, "CmdClearColor requires a valid image from this device.");
    return false;
  }
#else
  const auto native = static_cast<MetalImage *>(image);
#endif
  if (!RetainResource(native))
    return false;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto pass = MTL::RenderPassDescriptor::renderPassDescriptor();
  if (!pass) {
    recording_failed_ = true;
    return false;
  }
  auto attachment = pass->colorAttachments()->object(0);
  attachment->setTexture(native->NativeImage());
  attachment->setLoadAction(MTL::LoadActionClear);
  attachment->setStoreAction(MTL::StoreActionStore);
  attachment->setClearColor(MTL::ClearColor::Make(color[0], color[1], color[2], color[3]));
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

bool MetalCommandContext::CmdPresent(rhi::IWindow *window) {
#if !defined(NDEBUG)
  if (render_target_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before clear, present or submission.");
    return false;
  }
  const auto target = dynamic_cast<MetalWindow *>(window);
  if (!target || target->DeviceIdentity() != device_ || !glfw::IsWindowThread()) {
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
#else
  const auto target = static_cast<MetalWindow *>(window);
#endif
  auto frame = ObjectPtr<MetalSwapChainImage>::Adopt(static_cast<MetalSwapChainImage *>(target->GetNextImage()));
  if (!frame || !RetainResource(frame.Get()))
    return false;

  presents_.emplace_back(target, frame.Get());
  return true;
}

bool MetalCommandContext::EndRecording() {
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

CommandContext::SubmitResult MetalCommandContext::SubmitNative() {
#if !defined(NDEBUG)
  if (!presents_.empty() && !glfw::IsWindowThread()) {
    ReportError(-1, "Present must run on the window thread.");
    return SUBMIT_RESULT_REJECTED;
  }
  for (const auto &present : presents_) {
    if (!present.second->IsValid())
      return SUBMIT_RESULT_REJECTED;
  }
#endif
  {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    std::lock_guard<std::mutex> lock(device_->QueueMutex());
    for (const auto &present : presents_)
      commands_->presentDrawable(present.second->NativeDrawable());
    commands_->commit();
    MarkSubmitted();
    for (auto &entry : as_dependencies_)
      entry.first->CommitDependencies(std::move(entry.second));
    as_dependencies_.clear();
    for (const auto &present : presents_)
      present.second->DidPresent();
  }
  for (const auto &present : presents_) {
    if (!present.first->UpdateNextImage()) {
      submission_failed_ = true;
      ReportError(-1, "Failed to prepare the next Metal drawable after submission.");
      return SUBMIT_RESULT_FAILED_IN_FLIGHT;
    }
  }
  return SUBMIT_RESULT_SUBMITTED;
}

SubmissionStatus MetalCommandContext::WaitForProgress() {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  commands_->waitUntilCompleted();
  if (commands_->status() == MTL::CommandBufferStatusError) {
    auto error = commands_->error();
    ReportError(-1, error ? error->localizedDescription()->utf8String() : "Metal command buffer failed.");
    return SUBMISSION_STATUS_FAILED;
  }
  if (commands_->status() != MTL::CommandBufferStatusCompleted)
    return SUBMISSION_STATUS_PENDING;
  return submission_failed_ ? SUBMISSION_STATUS_FAILED : SUBMISSION_STATUS_COMPLETED;
}

MetalCommandContext::~MetalCommandContext() {
  if (encoder_)
    encoder_->endEncoding();
}

bool MetalCommandContext::CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) {
#if !defined(NDEBUG)
  auto native = dynamic_cast<MetalBuffer *>(buffer);
  if (!native || native->Owner() != device_ || sealed_ || render_target_) {
    recording_failed_ = true;
    return false;
  }
#else
  auto native = static_cast<MetalBuffer *>(buffer);
#endif
  if (!data || !size || offset > native->GetDesc().size || size > native->GetDesc().size - offset) {
    recording_failed_ = true;
    return false;
  }
  auto upload = device_->UploadPool().Acquire(size);
  if (!upload || !upload->Write(data, size)) {
    recording_failed_ = true;
    return false;
  }
  return CmdCopyBuffer(upload.Get(), 0, native, offset, size);
}

bool MetalCommandContext::CmdCopyBuffer(IBuffer *source,
                                        uint64_t source_offset,
                                        IBuffer *destination,
                                        uint64_t destination_offset,
                                        uint64_t size) {
#if !defined(NDEBUG)
  auto src = dynamic_cast<MetalBuffer *>(source);
  auto dst = dynamic_cast<MetalBuffer *>(destination);
  if (!src || !dst || src->Owner() != device_ || dst->Owner() != device_ || sealed_ || render_target_) {
    recording_failed_ = true;
    return false;
  }
#else
  auto src = static_cast<MetalBuffer *>(source);
  auto dst = static_cast<MetalBuffer *>(destination);
#endif
  if (src == dst || !size || source_offset > src->GetDesc().size || size > src->GetDesc().size - source_offset ||
      destination_offset > dst->GetDesc().size || size > dst->GetDesc().size - destination_offset) {
    recording_failed_ = true;
    return false;
  }
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

bool MetalCommandContext::CmdUploadImage(IImage *image, const void *data, uint64_t row_pitch) {
  auto native = dynamic_cast<MetalImage *>(image);
  if (sealed_ || encoder_ || !native || !native->IsValid() || native->DeviceIdentity() != device_ || !data ||
      IsDepthFormat(native->GetDesc().format) || row_pitch < uint64_t(native->GetDesc().width) * 4 ||
      (native->GetDesc().height - 1) > (SIZE_MAX - uint64_t(native->GetDesc().width) * 4) / row_pitch) {
    recording_failed_ = true;
    return false;
  }

  const auto desc = native->GetDesc();
  const uint64_t pitch = uint64_t(desc.width) * 4;
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

bool MetalCommandContext::CmdBindGroup(uint32_t index, IBindingGroup *group) {
  auto native = dynamic_cast<MetalBindingGroup *>(group);
  if (sealed_ || !native_program_ || !native || native->Owner() != device_ || index >= groups_.size() ||
      !SameBindingLayout(native->GetLayout(), native_program_->BindingLayout().groups[index]) ||
      !native->CurrentVersion()->IsComplete()) {
    recording_failed_ = true;
    return false;
  }
  auto version = static_cast<MetalBindingGroup::Version *>(native->CurrentVersion());
  if (!RetainResource(version))
    return false;
  groups_[index] = version;
  if (encoder_)
    version->Bind(encoder_.get(), index);
  return true;
}

bool MetalCommandContext::CmdBindComputeProgram(IComputeProgram *program) {
  auto native = dynamic_cast<MetalComputeProgram *>(program);
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

bool MetalCommandContext::CmdDispatch(uint32_t x, uint32_t y, uint32_t z) {
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

bool MetalCommandContext::CmdBindProgram(IProgram *program) {
#if !defined(NDEBUG)
  auto native = dynamic_cast<MetalProgram *>(program);
  if (!native || native->Owner() != device_ || sealed_) {
    recording_failed_ = true;
    return false;
  }
#else
  auto native = static_cast<MetalProgram *>(program);
#endif
  if (!RetainResource(native))
    return false;

  vertex_buffers_.assign(native->GetDesc().vertex_bindings.size(), {});
  groups_.assign(native->GetDesc().pipeline_layout.groups.size(), nullptr);

  program_ = native;
  compute_program_ = nullptr;
  native_program_ = native;
  if (encoder_)
    ApplyBindings();
  return true;
}

void MetalCommandContext::ApplyBindings() {
  if (program_) {
    encoder_->setRenderPipelineState(program_->NativePipeline());
    encoder_->setDepthStencilState(program_->NativeDepthStencil());
    encoder_->setStencilReferenceValue(program_->GetDesc().depth_stencil.stencil_reference);
  }
  for (uint32_t i = 0; i < groups_.size(); ++i)
    if (groups_[i])
      groups_[i]->Bind(encoder_.get(), i);
  for (size_t i = 0; i < vertex_buffers_.size(); ++i)
    if (vertex_buffers_[i].first)
      encoder_->setVertexBuffer(vertex_buffers_[i].first->NativeBuffer(), vertex_buffers_[i].second,
                                i + MetalProgram::VertexBufferBase);
  if (viewport_set_)
    encoder_->setViewport(MTL::Viewport{viewport_.x, viewport_.y, viewport_.width, viewport_.height,
                                        viewport_.min_depth, viewport_.max_depth});
  if (scissor_set_)
    ApplyScissor();
}

void MetalCommandContext::ApplyScissor() {
  // Metal requires the native scissor to fit the current attachment. Preserve
  // the API state and intersect it again when switching to a different target.
  const auto desc = render_target_->GetDesc();
  const auto x = std::min<uint64_t>(scissor_.x, desc.width);
  const auto y = std::min<uint64_t>(scissor_.y, desc.height);
  const auto right = std::min<uint64_t>(uint64_t(scissor_.x) + scissor_.width, desc.width);
  const auto bottom = std::min<uint64_t>(uint64_t(scissor_.y) + scissor_.height, desc.height);
  encoder_->setScissorRect(MTL::ScissorRect{x, y, right - x, bottom - y});
}

bool MetalCommandContext::CmdBeginRendering(const RenderingInfo &info) {
  if (sealed_ || encoder_ || info.color_attachments.size() > 8 ||
      (info.color_attachments.empty() && !info.depth_stencil.image)) {
    recording_failed_ = true;
    return false;
  }
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto pass = MTL::RenderPassDescriptor::renderPassDescriptor();
  MetalImage *native = nullptr;
  uint32_t width = 0, height = 0;
  auto validate = [&](IImage *image, AttachmentLoadOp load, AttachmentStoreOp store) -> MetalImage * {
    auto value = dynamic_cast<MetalImage *>(image);
    if (!value || !value->IsValid() || value->DeviceIdentity() != device_ || load < ATTACHMENT_LOAD_OP_LOAD ||
        load > ATTACHMENT_LOAD_OP_DISCARD || store < ATTACHMENT_STORE_OP_STORE || store > ATTACHMENT_STORE_OP_DISCARD)
      return nullptr;
    auto desc = value->GetDesc();
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
    if (!image || IsDepthFormat(image->GetDesc().format)) {
      recording_failed_ = true;
      return false;
    }
    for (size_t j = 0; j < i; ++j)
      if (info.color_attachments[j].color_attachment == a.color_attachment) {
        recording_failed_ = true;
        return false;
      }
    native = image;
    color_formats_.push_back(image->GetDesc().format);
    color_targets_.push_back(image);
    auto color = pass->colorAttachments()->object(i);
    color->setTexture(image->NativeImage());
    color->setLoadAction(loads[a.load_op]);
    color->setStoreAction(store(a.store_op));
    color->setClearColor(MTL::ClearColor::Make(a.clear_color[0], a.clear_color[1], a.clear_color[2], a.clear_color[3]));
  }

  depth_format_ = IMAGE_FORMAT_UNKNOWN;
  const auto &ds = info.depth_stencil;
  if (ds.image) {
    auto image = validate(ds.image, ds.depth_load_op, ds.depth_store_op);
    if (!image || !IsDepthFormat(image->GetDesc().format) || ds.stencil_load_op < ATTACHMENT_LOAD_OP_LOAD ||
        ds.stencil_load_op > ATTACHMENT_LOAD_OP_DISCARD || ds.stencil_store_op < ATTACHMENT_STORE_OP_STORE ||
        ds.stencil_store_op > ATTACHMENT_STORE_OP_DISCARD) {
      recording_failed_ = true;
      return false;
    }
    native = image;
    depth_format_ = image->GetDesc().format;
    auto depth = pass->depthAttachment();
    depth->setTexture(image->NativeImage());
    depth->setLoadAction(loads[ds.depth_load_op]);
    depth->setStoreAction(store(ds.depth_store_op));
    depth->setClearDepth(ds.clear_depth);
    if (HasStencil(depth_format_)) {
      auto stencil = pass->stencilAttachment();
      stencil->setTexture(image->NativeImage());
      stencil->setLoadAction(loads[ds.stencil_load_op]);
      stencil->setStoreAction(store(ds.stencil_store_op));
      stencil->setClearStencil(ds.clear_stencil);
    }
  }
  encoder_ = NS::RetainPtr(commands_->renderCommandEncoder(pass));
  if (!encoder_) {
    recording_failed_ = true;
    return false;
  }
  render_target_ = native;
  ApplyBindings();
  return true;
}

bool MetalCommandContext::CmdEndRendering() {
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
  return true;
}

bool MetalCommandContext::CmdBindVertexBuffer(uint32_t binding, IBuffer *buffer, uint64_t offset) {
#if !defined(NDEBUG)
  auto native = dynamic_cast<MetalBuffer *>(buffer);
  if (!native || native->Owner() != device_ || sealed_) {
    recording_failed_ = true;
    return false;
  }
#else
  auto native = static_cast<MetalBuffer *>(buffer);
#endif
  if (!RetainResource(native))
    return false;
  if (!program_ || binding >= vertex_buffers_.size() || offset >= native->GetDesc().size) {
    recording_failed_ = true;
    return false;
  }
  vertex_buffers_[binding] = {native, offset};
  if (encoder_)
    encoder_->setVertexBuffer(native->NativeBuffer(), offset, binding + MetalProgram::VertexBufferBase);
  return true;
}

bool MetalCommandContext::CmdBindIndexBuffer(IBuffer *buffer, IndexFormat format, uint64_t offset) {
#if !defined(NDEBUG)
  auto native = dynamic_cast<MetalBuffer *>(buffer);
  if (!native || native->Owner() != device_ || sealed_) {
    recording_failed_ = true;
    return false;
  }
#else
  auto native = static_cast<MetalBuffer *>(buffer);
#endif
  if (!RetainResource(native))
    return false;
  if ((format != INDEX_FORMAT_UINT16 && format != INDEX_FORMAT_UINT32) || offset >= native->GetDesc().size ||
      offset % (format == INDEX_FORMAT_UINT16 ? 2 : 4)) {
    recording_failed_ = true;
    return false;
  }
  index_ = native;
  index_offset_ = offset;
  index_format_ = format;
  return true;
}

bool MetalCommandContext::CmdSetViewport(const Viewport &viewport) {
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

bool MetalCommandContext::CmdSetScissor(const Scissor &scissor) {
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

bool MetalCommandContext::ValidateDraw(bool indexed) {
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

bool MetalCommandContext::CmdDraw(uint32_t vertex_count,
                                  uint32_t instance_count,
                                  uint32_t first_vertex,
                                  uint32_t first_instance) {
  if (!ValidateDraw(false))
    return false;
  encoder_->drawPrimitives(MTL::PrimitiveTypeTriangle, first_vertex, vertex_count, instance_count, first_instance);
  return true;
}

bool MetalCommandContext::CmdDrawIndexed(uint32_t index_count,
                                         uint32_t instance_count,
                                         uint32_t first_index,
                                         int32_t vertex_offset,
                                         uint32_t first_instance) {
  if (!ValidateDraw(true))
    return false;
  encoder_->drawIndexedPrimitives(
      MTL::PrimitiveTypeTriangle, index_count,
      index_format_ == INDEX_FORMAT_UINT16 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32, index_->NativeBuffer(),
      index_offset_ + uint64_t(first_index) * (index_format_ == INDEX_FORMAT_UINT16 ? 2 : 4), instance_count,
      vertex_offset, first_instance);
  return true;
}
}  // namespace grassland::rhi::backend::metal
