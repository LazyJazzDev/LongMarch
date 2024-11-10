#pragma once
#include "grassland/d3d12/d3d12util.h"
#include "grassland/util/double_ptr.h"

namespace grassland::d3d12 {
class DXGIFactory {
 public:
  explicit DXGIFactory(const ComPtr<IDXGIFactory4> &factory);

  IDXGIFactory4 *Handle() const {
    return factory_.Get();
  }

  std::vector<Adapter> EnumerateAdapters() const;

  HRESULT CreateDevice(
      const DeviceFeatureRequirement &device_feature_requirement,
      int device_index,
      double_ptr<Device> pp_device);

  HRESULT CreateSwapChain(const CommandQueue &command_queue,
                          HWND hwnd,
                          const DXGI_SWAP_CHAIN_DESC1 &desc,
                          double_ptr<SwapChain> pp_swap_chain);

  HRESULT CreateSwapChain(const CommandQueue &command_queue,
                          HWND hwnd,
                          UINT buffer_count,
                          double_ptr<SwapChain> pp_swap_chain);

 private:
  ComPtr<IDXGIFactory4> factory_;
};

HRESULT CreateDXGIFactory(double_ptr<DXGIFactory> pp_factory);

}  // namespace grassland::d3d12
