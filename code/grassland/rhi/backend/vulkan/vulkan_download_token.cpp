#include "grassland/rhi/backend/vulkan/vulkan_download_token.h"

namespace grassland::rhi::backend::vulkan {
VulkanDownloadToken::VulkanDownloadToken(ObjectPtr<VulkanTemporalBufferLease> lease, uint64_t size)
    : VulkanDownloadToken(std::move(lease), DownloadLayout{0, size, size, 1}) {
}

VulkanDownloadToken::VulkanDownloadToken(ObjectPtr<VulkanTemporalBufferLease> lease, const DownloadLayout &layout)
    : DownloadToken(layout.row_bytes * layout.rows),
      lease_(std::move(lease)),
      layout_(layout) {
}

VulkanDownloadToken::~VulkanDownloadToken() = default;

bool VulkanDownloadToken::CopyData(void *destination) const {
  auto output = static_cast<uint8_t *>(destination);
  if (layout_.row_pitch == layout_.row_bytes)
    return lease_->Read(output, GetSize(), layout_.offset);
  for (uint32_t y = 0; y < layout_.rows; ++y) {
    if (!lease_->Read(output + y * layout_.row_bytes, layout_.row_bytes, layout_.offset + y * layout_.row_pitch))
      return false;
  }
  return true;
}
}  // namespace grassland::rhi::backend::vulkan
