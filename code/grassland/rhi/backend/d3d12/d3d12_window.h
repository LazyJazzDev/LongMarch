#pragma once
#include <array>

#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12Window,
                        grassland::rhi::backend::IBackendWindow,
                        "1babd07b-773d-4d2b-9392-ec5113e23fdf")

namespace grassland::rhi::backend::d3d12 {

class ID3D12Window : public IBackendWindow {
 public:
  virtual ID3D12Queue *OwnerQueue() const = 0;
  virtual bool RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image) = 0;
  virtual bool RecordImGui(rhi::ICommandContext *commands,
                           rhi::IImage *image,
                           ColorSpace color_space,
                           float white_scale) = 0;
  virtual bool RecordPresentImage(ICommandContext *commands, IImage *image, ColorSpace color_space) = 0;
  virtual const void *DeviceIdentity() const = 0;
  virtual bool UpdateNextImage() = 0;
  virtual void ReleaseNextImage() = 0;
  virtual ObjectPtr<ID3D12Image> AcquireNativeFrame() = 0;
  virtual bool RebuildNativeSwapChain(PresentMode mode) = 0;
  static rhi::IWindow *Create(ID3D12Device *device, const WindowDesc &desc);

  virtual bool IsFrameValid(uint64_t generation) const = 0;
  virtual HRESULT PresentFrame(uint64_t generation) = 0;
};

}  // namespace grassland::rhi::backend::d3d12
