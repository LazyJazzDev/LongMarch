#pragma once
#include <vector>

#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Image;
class D3D12Window;
class D3D12SwapChainImage;

class D3D12CommandContext final : public CommandContext {
 public:
  D3D12CommandContext(D3D12Device *device);
  ~D3D12CommandContext() override;
  bool Initialize();
  bool CmdClearColor(rhi::IImage *image, const std::array<float, 4> &color) override;
  bool CmdPresent(rhi::IWindow *window) override;
  SubmitResult SubmitNative() override;
  SubmissionStatus WaitForProgress() override;

 private:
  bool EndRecording() override;
  bool TransitionResource(ID3D12Resource *resource, D3D12_RESOURCE_STATES state, D3D12_RESOURCE_STATES default_state);

  struct ResourceState {
    D3D12_RESOURCE_STATES current;
    D3D12_RESOURCE_STATES default_state;
  };

  std::unordered_map<ID3D12Resource *, ResourceState> resource_states_;
  std::vector<std::pair<D3D12Window *, D3D12SwapChainImage *>> presents_;
  D3D12Device *device_;
  ComPtr<ID3D12CommandAllocator> allocator_;
  ComPtr<ID3D12GraphicsCommandList> list_;
  HANDLE event_{};
  uint64_t value_{};
  bool submission_failed_{};
};

}  // namespace grassland::rhi::backend::d3d12
