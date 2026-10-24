#include "grassland/rhi/backend/d3d12/d3d12_download_token.h"

#include <vector>

namespace grassland::rhi::backend::d3d12 {
D3D12DownloadToken::D3D12DownloadToken(ObjectPtr<D3D12TemporalBufferLease> lease, uint64_t size)
    : D3D12DownloadToken(std::move(lease), DownloadLayout{0, size, size, 1}) {
}

D3D12DownloadToken::D3D12DownloadToken(ObjectPtr<D3D12TemporalBufferLease> lease, const DownloadLayout &layout)
    : DownloadToken(layout.row_bytes * layout.rows),
      lease_(std::move(lease)),
      layout_(layout) {
}

D3D12DownloadToken::~D3D12DownloadToken() = default;

bool D3D12DownloadToken::CopyData(void *destination) const {
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
}  // namespace grassland::rhi::backend::d3d12
