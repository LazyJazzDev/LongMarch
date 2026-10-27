#include "grassland/rhi/backend/backend_headless_window.h"

#include "grassland/rhi/backend/backend_device.h"
#if defined(LONGMARCH_METAL_ENABLED)
#include "grassland/rhi/backend/metal/metal_image.h"
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/rhi/backend/vulkan/vulkan_image.h"
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_image.h"
#endif

namespace grassland::rhi::backend {
namespace {
ImageFormat FrameFormat(ColorSpace space) {
  switch (space) {
    case COLOR_SPACE_SRGB_NONLINEAR:
      return IMAGE_FORMAT_R8G8B8A8_UNORM;
    case COLOR_SPACE_EXTENDED_SRGB_LINEAR:
    case COLOR_SPACE_BT709_LINEAR:
      return IMAGE_FORMAT_R16G16B16A16_SFLOAT;
    case COLOR_SPACE_HDR10_ST2084:
      return IMAGE_FORMAT_R10G10B10A2_UNORM;
    default:
      return IMAGE_FORMAT_UNKNOWN;
  }
}

#if defined(LONGMARCH_METAL_ENABLED)
class MetalFrameLease final : public metal::MetalImage {
 public:
  MetalFrameLease(metal::MetalImage *image, const std::shared_ptr<HeadlessFramePool> &pool)
      : MetalImage(image->GetDesc(), image->DeviceIdentity(), image->NativeImage()),
        reservation_(pool),
        image_(ObjectPtr<metal::MetalImage>::Retain(image)) {
  }

  bool IsValid() const override {
    return image_->IsValid();
  }

  bool SupportsStorage() const override {
    return image_->SupportsStorage();
  }

 private:
  HeadlessFramePool::Reservation reservation_;
  ObjectPtr<metal::MetalImage> image_;
};
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
class VulkanFrameLease final : public vulkan::VulkanImage {
 public:
  VulkanFrameLease(vulkan::VulkanImage *image, const std::shared_ptr<HeadlessFramePool> &pool)
      : VulkanImage(image->GetDesc(), image->DeviceIdentity(), image->DefaultLayout()),
        reservation_(pool),
        image_(ObjectPtr<vulkan::VulkanImage>::Retain(image)) {
  }

  bool IsValid() const override {
    return image_->IsValid();
  }

  bool SupportsStorage() const override {
    return image_->SupportsStorage();
  }

  vulkan::VulkanImage *StateIdentity() override {
    return image_->StateIdentity();
  }

  VkImage NativeImage() const override {
    return image_->NativeImage();
  }

  VkImageView AttachmentView() const override {
    return image_->AttachmentView();
  }

  VkImageView BindingView() const override {
    return image_->BindingView();
  }

 private:
  HeadlessFramePool::Reservation reservation_;
  ObjectPtr<vulkan::VulkanImage> image_;
};
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
class D3D12FrameLease final : public d3d12::D3D12Image {
 public:
  D3D12FrameLease(d3d12::D3D12Image *image,
                  const std::shared_ptr<HeadlessFramePool> &pool,
                  ObjectPtr<d3d12::D3D12DescriptorPool::Lease> attachment)
      : D3D12Image(image->GetDesc(),
                   image->DeviceIdentity(),
                   image->NativeImage(),
                   std::move(attachment),
                   image->DefaultState()),
        reservation_(pool),
        image_(ObjectPtr<d3d12::D3D12Image>::Retain(image)) {
  }

  bool IsValid() const override {
    return image_->IsValid();
  }

 private:
  HeadlessFramePool::Reservation reservation_;
  ObjectPtr<d3d12::D3D12Image> image_;
};
#endif
}  // namespace

HeadlessFramePool::Reservation::Reservation(std::shared_ptr<HeadlessFramePool> pool) : pool_(std::move(pool)) {
  std::unique_lock<std::mutex> lock(pool_->mutex_);
  pool_->available_.wait(lock, [&] { return pool_->in_use_ < 3; });
  ++pool_->in_use_;
}

HeadlessFramePool::Reservation::~Reservation() {
  {
    std::lock_guard<std::mutex> lock(pool_->mutex_);
    --pool_->in_use_;
  }
  pool_->available_.notify_one();
}

size_t HeadlessFramePool::InUse() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return in_use_;
}

HeadlessWindow::HeadlessWindow(Device *device, BackendAPI api, const WindowDesc &desc)
    : Window(device),
      device_(device),
      api_(api),
      width_(desc.width),
      height_(desc.height),
      color_space_(desc.color_space),
      pool_(std::make_shared<HeadlessFramePool>()) {
}

HeadlessWindow *HeadlessWindow::Create(Device *device, BackendAPI api, const WindowDesc &desc) {
  if (desc.width <= 0 || desc.height <= 0 || FrameFormat(desc.color_space) == IMAGE_FORMAT_UNKNOWN)
    return nullptr;
  auto window = ObjectPtr<HeadlessWindow>::Adopt(new HeadlessWindow(device, api, desc));
  window->frame_image_ = ObjectPtr<Image>::Adopt(
      static_cast<Image *>(device->CreateImage({window->width_, window->height_, FrameFormat(desc.color_space)})));
  return window->frame_image_ ? window.Detach() : nullptr;
}

ObjectPtr<Image> HeadlessWindow::CreateFrameLease() {
  switch (api_) {
#if defined(LONGMARCH_METAL_ENABLED)
    case BACKEND_API_METAL:
      return ObjectPtr<Image>::Adopt(new MetalFrameLease(static_cast<metal::MetalImage *>(frame_image_.Get()), pool_));
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    case BACKEND_API_VULKAN:
      return ObjectPtr<Image>::Adopt(
          new VulkanFrameLease(static_cast<vulkan::VulkanImage *>(frame_image_.Get()), pool_));
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
    case BACKEND_API_D3D12: {
      auto image = static_cast<d3d12::D3D12Image *>(frame_image_.Get());
      auto attachment =
          d3d12::D3D12Image::CreateAttachmentView(static_cast<d3d12::D3D12Device *>(device_), image->NativeImage());
      if (!attachment)
        return {};
      return ObjectPtr<Image>::Adopt(new D3D12FrameLease(image, pool_, std::move(attachment)));
    }
#endif
    default:
      return {};
  }
}

IImage *HeadlessWindow::GetNextImage() {
  if (!current_)
    current_ = CreateFrameLease();
  if (current_)
    current_->AddRef();
  return current_.Get();
}

void HeadlessWindow::AdvanceFrame() {
  current_.Reset();
}

bool HeadlessWindow::EnableImGui(bool, const char *, float) {
  return true;
}

bool HeadlessWindow::UpdateImGui(const std::function<void()> &) {
  return true;
}

GLFWwindow *HeadlessWindow::GLFWWindow() const {
  return nullptr;
}

bool HeadlessWindow::ShouldClose() const {
  return false;
}

bool HeadlessWindow::IsKeyDown(int) const {
  return false;
}

bool HeadlessWindow::SetTitle(const std::string &) {
  return true;
}

bool HeadlessWindow::IsSwapChainInvalid() const {
  return false;
}

bool HeadlessWindow::RebuildSwapChain() {
  return true;
}

bool HeadlessWindow::UpdateNextImage() {
  return true;
}

bool HeadlessWindow::SetPresentMode(PresentMode) {
  return true;
}

PresentMode HeadlessWindow::GetPresentMode() const {
  return PRESENT_MODE_CONTINUOUS;
}

bool HeadlessWindow::SetColorSpace(ColorSpace space) {
  if (FrameFormat(space) == IMAGE_FORMAT_UNKNOWN)
    return false;
  if (space == color_space_)
    return true;
  // Replacement cannot change the backing image while callers/recordings own leases.
  if ((current_ && !current_->IsUniquelyOwned()) || pool_->InUse() > (current_ ? 1u : 0u))
    return false;
  auto image =
      ObjectPtr<Image>::Adopt(static_cast<Image *>(device_->CreateImage({width_, height_, FrameFormat(space)})));
  if (!image)
    return false;
  current_.Reset();
  frame_image_ = std::move(image);
  color_space_ = space;
  return true;
}

ColorSpace HeadlessWindow::GetColorSpace() const {
  return color_space_;
}

bool HeadlessWindow::GetHDRReferenceWhiteNits(float &) const {
  return false;
}

float HeadlessWindow::GetHDRReferenceWhiteScale() const {
  return 1.0f;
}

float HeadlessWindow::GetHDRHeadroom() const {
  return 10.0f;
}

std::vector<WindowPresentationOption> HeadlessWindow::GetSupportedPresentationOptions() const {
  std::vector<WindowPresentationOption> result;
  for (auto space : {COLOR_SPACE_SRGB_NONLINEAR, COLOR_SPACE_EXTENDED_SRGB_LINEAR, COLOR_SPACE_BT709_LINEAR,
                     COLOR_SPACE_HDR10_ST2084})
    result.push_back({PRESENT_MODE_CONTINUOUS, FrameFormat(space), space});
  return result;
}

ObjectPtr<Image> HeadlessWindow::AcquireNativeFrame() {
  return {};
}

bool HeadlessWindow::RebuildNativeSwapChain(PresentMode) {
  return false;
}
}  // namespace grassland::rhi::backend
