#include "grassland/rhi/backend/d3d12/d3d12_swap_chain_image.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_window.h"

namespace grassland::rhi::backend::d3d12 {

D3D12SwapChainImageImpl::D3D12SwapChainImageImpl(ImageDesc desc,
                                                 ID3D12Window *window,
                                                 uint64_t generation,
                                                 ID3D12Resource *image,
                                                 ObjectPtr<D3D12DescriptorPool::Lease> attachment)
    : window_(window),
      generation_(generation) {
  InitializeD3D12Image(desc, window->DeviceIdentity(), image, std::move(attachment), D3D12_RESOURCE_STATE_COMMON,
                       window->OwnerQueue());
}

D3D12SwapChainImageImpl::~D3D12SwapChainImageImpl() {
}

bool D3D12SwapChainImageImpl::IsValid() const {
#if !defined(NDEBUG)
  if (present_requested_)
    return false;
#endif
  return window_->IsFrameValid(generation_);
}

HRESULT D3D12SwapChainImageImpl::Present() {
#if !defined(NDEBUG)
  present_requested_ = true;
#endif
  return window_->PresentFrame(generation_);
}

ID3D12SwapChainImage *ID3D12SwapChainImage::Create(ImageDesc desc,
                                                   ID3D12Window *window,
                                                   uint64_t generation,
                                                   ID3D12Resource *image,
                                                   ObjectPtr<D3D12DescriptorPool::Lease> attachment) {
  return new D3D12SwapChainImageImpl(desc, window, generation, image, std::move(attachment));
}

void D3D12SwapChainImageImpl::InitializeImage(ImageDesc desc, ID3D12Queue *queue) {
  desc_ = desc;
  queue_ = queue;
}

bool D3D12SwapChainImageImpl::SupportsQueue(const ID3D12Queue *queue) const {
  return !queue_ || queue_ == queue;
}

const ImageDesc &D3D12SwapChainImageImpl::GetDesc() const {
  return desc_;
}

void D3D12SwapChainImageImpl::InitializeD3D12Image(ImageDesc desc,
                                                   const void *device,
                                                   ID3D12Resource *image,
                                                   ObjectPtr<D3D12DescriptorPool::Lease> attachment,
                                                   D3D12_RESOURCE_STATES default_state,
                                                   ID3D12Queue *queue) {
  InitializeImage(desc, queue);
  attachment_ = std::move(attachment);
  device_ = device;
  image_ = image;
  default_state_ = default_state;
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12SwapChainImageImpl::AttachmentView() const {
  return attachment_->CPU();
}

bool D3D12SwapChainImageImpl::SupportsStorage() const {
  return (image_->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0;
}

const void *D3D12SwapChainImageImpl::DeviceIdentity() const {
  return device_;
}

ID3D12Resource *D3D12SwapChainImageImpl::NativeImage() const {
  return image_;
}

D3D12_RESOURCE_STATES D3D12SwapChainImageImpl::DefaultState() const {
  return default_state_;
}
}  // namespace grassland::rhi::backend::d3d12
