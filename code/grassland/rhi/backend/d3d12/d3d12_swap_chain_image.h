#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12SwapChainImage,
                        grassland::rhi::backend::d3d12::ID3D12Image,
                        "f03d8d79-9e19-41ca-b585-8f6f5bd64143")

namespace grassland::rhi::backend::d3d12 {

class ID3D12SwapChainImage : public ID3D12Image {
 public:
  static ID3D12SwapChainImage *Create(ImageDesc desc,
                                      ID3D12Window *window,
                                      uint64_t generation,
                                      ID3D12Resource *image,
                                      ObjectPtr<D3D12DescriptorPool::Lease> attachment);
  virtual HRESULT Present() = 0;
};

}  // namespace grassland::rhi::backend::d3d12

namespace grassland::rhi::backend::d3d12 {
class D3D12SwapChainImageImpl final : public Object<rhi::IImage, ID3D12SwapChainImage> {
 public:
  D3D12SwapChainImageImpl(ImageDesc desc,
                          ID3D12Window *window,
                          uint64_t generation,
                          ID3D12Resource *image,
                          ObjectPtr<D3D12DescriptorPool::Lease> attachment);
  ~D3D12SwapChainImageImpl() override;
  bool IsValid() const override;
  HRESULT Present();

 private:
  ID3D12Window *window_;
  uint64_t generation_;
#if !defined(NDEBUG)
  bool present_requested_{};
#endif

  void InitializeImage(ImageDesc desc, ID3D12Queue *queue = nullptr);

 public:
  bool SupportsQueue(const ID3D12Queue *queue) const override;

  const ImageDesc &GetDesc() const override;

 private:
  ImageDesc desc_{};
  ID3D12Queue *queue_{};

  void InitializeD3D12Image(ImageDesc desc,
                            const void *device,
                            ID3D12Resource *image,
                            ObjectPtr<D3D12DescriptorPool::Lease> attachment,
                            D3D12_RESOURCE_STATES default_state = D3D12_RESOURCE_STATE_COMMON,
                            ID3D12Queue *queue = nullptr);

 public:
  D3D12_CPU_DESCRIPTOR_HANDLE AttachmentView() const override;

  bool SupportsStorage() const override;

  const void *DeviceIdentity() const override;

  ID3D12Resource *NativeImage() const override;

  D3D12_RESOURCE_STATES DefaultState() const override;

 private:
  ObjectPtr<D3D12DescriptorPool::Lease> attachment_{};
  const void *device_{};
  ID3D12Resource *image_{};
  D3D12_RESOURCE_STATES default_state_{};
};
}  // namespace grassland::rhi::backend::d3d12
