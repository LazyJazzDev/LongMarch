#include "grassland/rhi/backend/metal/metal_download_token.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace grassland::rhi::backend::metal {
MetalDownloadToken::MetalDownloadToken(ObjectPtr<MetalTemporalBufferLease> lease, uint64_t size)
    : MetalDownloadToken(std::move(lease), DownloadLayout{0, size, size, 1}) {
}

MetalDownloadToken::MetalDownloadToken(ObjectPtr<MetalTemporalBufferLease> lease, const DownloadLayout &layout)
    : DownloadToken(layout.row_bytes * layout.rows),
      lease_(std::move(lease)),
      layout_(layout) {
}

MetalDownloadToken::~MetalDownloadToken() = default;

bool MetalDownloadToken::CopyData(void *destination) const {
  auto output = static_cast<uint8_t *>(destination);
  if (!layout_.stencil_pitch && layout_.row_pitch == layout_.row_bytes)
    return lease_->Read(output, GetSize(), layout_.offset);
  std::vector<uint8_t> stencil(layout_.stencil_pitch ? layout_.row_bytes / 4 : 0);
  std::vector<float> depths(layout_.stencil_pitch ? layout_.row_bytes / 4 : 0);
  for (uint32_t y = 0; y < layout_.rows; ++y) {
    auto row = output + y * layout_.row_bytes;
    if (layout_.stencil_pitch) {
      if (!lease_->Read(depths.data(), layout_.row_bytes, layout_.offset + y * layout_.row_pitch) ||
          !lease_->Read(stencil.data(), stencil.size(), layout_.stencil_offset + y * layout_.stencil_pitch))
        return false;
      for (size_t x = 0; x < stencil.size(); ++x) {
        const uint32_t value =
            uint32_t(std::clamp(double(depths[x]), 0.0, 1.0) * 16777215.0 + 0.5) | (uint32_t(stencil[x]) << 24);
        std::memcpy(row + x * 4, &value, 4);
      }
    } else if (!lease_->Read(row, layout_.row_bytes, layout_.offset + y * layout_.row_pitch))
      return false;
  }
  return true;
}
}  // namespace grassland::rhi::backend::metal
