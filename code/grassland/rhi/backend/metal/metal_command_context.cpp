#include "grassland/rhi/backend/metal/metal_command_context.h"

#include "grassland/rhi/backend/metal/metal_swap_chain_image.h"
#include "grassland/rhi/backend/metal/metal_window.h"

namespace grassland::rhi::backend::metal {

MetalCommandContext::MetalCommandContext(MetalDevice *device) : CommandContext(device), device_(device) {
}

bool MetalCommandContext::Initialize() {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  // Metal command buffers are single-use. The persistent native queue manages
  // their backing storage; completed buffers cannot be reset like Vulkan/D3D12.
  commands_ = NS::RetainPtr(device_->NativeQueue()->commandBuffer());
  if (!commands_)
    ReportError(-1, "Metal command buffer creation failed.");
  return bool(commands_);
}

bool MetalCommandContext::CmdClearColor(rhi::IImage *image, const std::array<float, 4> &color) {
#if !defined(NDEBUG)
  const auto native = dynamic_cast<MetalImage *>(image);
  if (!native || !native->IsValid() || native->DeviceIdentity() != device_) {
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
  try {
    presents_.emplace_back(target, frame.Get());
    return true;
  } catch (const std::bad_alloc &) {
    recording_failed_ = true;
    return false;
  }
}

bool MetalCommandContext::EndRecording() {
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

}  // namespace grassland::rhi::backend::metal
