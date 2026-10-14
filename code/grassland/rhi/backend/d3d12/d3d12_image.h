#pragma once
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {

class D3D12Window;

class D3D12Image final : public IImage {
 public:
  D3D12Image(ImageDesc desc,
             D3D12Window *window,
             uint64_t generation,
             ID3D12Resource *image,
             D3D12_CPU_DESCRIPTOR_HANDLE rtv);
  ~D3D12Image() override;
  bool IsValid() const override;
  const void *DeviceIdentity() const override;
  ID3D12Resource *NativeImage() const;
  D3D12_CPU_DESCRIPTOR_HANDLE RenderTarget() const;
  HRESULT Present();

 private:
  D3D12Window *window_;
  uint64_t generation_;
  ID3D12Resource *image_;
  D3D12_CPU_DESCRIPTOR_HANDLE rtv_;
  std::atomic<bool> presented_{};
};

}  // namespace grassland::rhi::backend::d3d12
