#include "grassland/rhi/backend/vulkan/vulkan_download_token.h"

#include <vector>

#include "grassland/rhi/backend/vulkan/vulkan_download_token_impl.h"

namespace grassland::rhi::backend::vulkan {
VulkanDownloadTokenImpl::VulkanDownloadTokenImpl(ObjectPtr<IVulkanTemporalBufferLease> lease, uint64_t size)
    : VulkanDownloadTokenImpl(std::move(lease), DownloadLayout{0, size, size, 1}) {
}

VulkanDownloadTokenImpl::VulkanDownloadTokenImpl(ObjectPtr<IVulkanTemporalBufferLease> lease,
                                                 const DownloadLayout &layout)
    : lease_(std::move(lease)),
      layout_(layout) {
  InitializeDownloadToken(layout.row_bytes * layout.rows);
}

VulkanDownloadTokenImpl::~VulkanDownloadTokenImpl() {
}

bool VulkanDownloadTokenImpl::CopyData(void *destination) const {
  auto output = static_cast<uint8_t *>(destination);
  if (!layout_.stencil_pitch && layout_.row_pitch == layout_.row_bytes)
    return lease_->Read(output, GetSize(), layout_.offset);
  std::vector<uint8_t> stencil(layout_.stencil_pitch ? layout_.row_bytes / 4 : 0);
  for (uint32_t y = 0; y < layout_.rows; ++y) {
    auto row = output + y * layout_.row_bytes;
    if (!lease_->Read(row, layout_.row_bytes, layout_.offset + y * layout_.row_pitch))
      return false;
    if (layout_.stencil_pitch) {
      if (!lease_->Read(stencil.data(), stencil.size(), layout_.stencil_offset + y * layout_.stencil_pitch))
        return false;
      for (size_t x = 0; x < stencil.size(); ++x)
        row[x * 4 + 3] = stencil[x];
    }
  }
  return true;
}

IVulkanDownloadToken *CreateVulkanDownloadToken(ObjectPtr<IVulkanTemporalBufferLease> lease, uint64_t size) {
  return new VulkanDownloadTokenImpl(lease, size);
}

IVulkanDownloadToken *CreateVulkanDownloadToken(ObjectPtr<IVulkanTemporalBufferLease> lease,
                                                const DownloadLayout &layout) {
  return new VulkanDownloadTokenImpl(lease, layout);
}
}  // namespace grassland::rhi::backend::vulkan
