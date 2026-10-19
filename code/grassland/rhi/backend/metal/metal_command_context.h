#pragma once
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
class MetalWindow;
class MetalSwapChainImage;

class MetalCommandContext final : public CommandContext {
 public:
  explicit MetalCommandContext(MetalDevice *device);
  bool Initialize();
  bool CmdClearColor(rhi::IImage *image, const std::array<float, 4> &color) override;
  bool CmdPresent(rhi::IWindow *window) override;
  SubmitResult SubmitNative() override;
  SubmissionStatus WaitForProgress() override;

 private:
  bool EndRecording() override;
  MetalDevice *device_;
  NS::SharedPtr<MTL::CommandBuffer> commands_;
  std::vector<std::pair<MetalWindow *, MetalSwapChainImage *>> presents_;
  bool submission_failed_{};
};

}  // namespace grassland::rhi::backend::metal
