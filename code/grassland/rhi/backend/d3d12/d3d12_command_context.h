#pragma once
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {

class D3D12CommandContext final : public CommandContext {
 public:
  D3D12CommandContext(D3D12Device *device, QueueType type);
  ~D3D12CommandContext() override;
  bool Initialize();
  SubmitResult SubmitNative() override;
  SubmissionStatus WaitForProgress() override;

 private:
  bool RecordClear(Image *image, const std::array<float, 4> &color, bool first_use) override;
  bool RecordPresent(Image *image) override;
  bool EndRecording() override;
  D3D12Device *device_;
  ComPtr<ID3D12CommandAllocator> allocator_;
  ComPtr<ID3D12GraphicsCommandList> list_;
  HANDLE event_{};
  uint64_t value_{};
  bool failed_{};
};

}  // namespace grassland::rhi::backend::d3d12
