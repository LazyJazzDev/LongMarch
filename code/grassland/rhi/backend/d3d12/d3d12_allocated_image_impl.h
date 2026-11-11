#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_allocated_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12AllocatedImageImpl final : public Object<rhi::IImage, ID3D12Image> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IImage, ID3D12Image, IBackendImage, rhi::IImage)
  static IImage *Create(ID3D12Device *device, const ImageDesc &desc);
  ~D3D12AllocatedImageImpl() override;
  bool IsValid() const override;

 private:
  D3D12AllocatedImageImpl(ID3D12Device *device,
                          ImageDesc desc,
                          ComPtr<ID3D12Resource> image,
                          ComPtr<D3D12MA::Allocation> allocation,
                          ObjectPtr<D3D12DescriptorPool::Lease> attachment);
  ComPtr<D3D12MA::Allocation> allocation_;
  ComPtr<ID3D12Resource> image_;

  void InitializeImage(ImageDesc desc, IBackendQueue *queue = nullptr);

 public:
  bool SupportsQueue(const IBackendQueue *queue) const override;

  const ImageDesc &GetDesc() const override;

 private:
  ImageDesc desc_{};
  IBackendQueue *queue_{};

  void InitializeD3D12Image(ImageDesc desc,
                            const void *device,
                            ID3D12Resource *image,
                            ObjectPtr<D3D12DescriptorPool::Lease> attachment,
                            D3D12_RESOURCE_STATES default_state = D3D12_RESOURCE_STATE_COMMON,
                            IBackendQueue *queue = nullptr);

 public:
  D3D12_CPU_DESCRIPTOR_HANDLE AttachmentView() const override;

  bool SupportsStorage() const override;

  const void *DeviceIdentity() const override;

  ID3D12Resource *NativeImage() const override;

  D3D12_RESOURCE_STATES DefaultState() const override;

 private:
  ObjectPtr<D3D12DescriptorPool::Lease> attachment_{};
  const void *device_{};

  D3D12_RESOURCE_STATES default_state_{};
};
}  // namespace grassland::rhi::backend::d3d12
