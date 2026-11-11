#include "grassland/rhi/backend/d3d12/d3d12_download_token.h"

#include <vector>

#include "grassland/rhi/backend/d3d12/d3d12_download_token_impl.h"

namespace grassland::rhi::backend::d3d12 {
D3D12DownloadTokenImpl::D3D12DownloadTokenImpl(ObjectPtr<ID3D12TemporalBufferLease> lease, uint64_t size)
    : D3D12DownloadTokenImpl(std::move(lease), DownloadLayout{0, size, size, 1}) {
}

D3D12DownloadTokenImpl::D3D12DownloadTokenImpl(ObjectPtr<ID3D12TemporalBufferLease> lease, const DownloadLayout &layout)
    : lease_(std::move(lease)),
      layout_(layout) {
  InitializeDownloadToken(layout.row_bytes * layout.rows);
}

D3D12DownloadTokenImpl::~D3D12DownloadTokenImpl() {
}

bool D3D12DownloadTokenImpl::CopyData(void *destination) const {
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

LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::ID3D12DownloadToken, "c5315cf9-1ed2-46ac-b619-f12be68cb0ae")

namespace grassland::rhi::backend::d3d12 {
ID3D12DownloadToken *ID3D12DownloadToken::Create(ObjectPtr<ID3D12TemporalBufferLease> lease, uint64_t size) {
  return new D3D12DownloadTokenImpl(lease, size);
}

ID3D12DownloadToken *ID3D12DownloadToken::Create(ObjectPtr<ID3D12TemporalBufferLease> lease,
                                                 const DownloadLayout &layout) {
  return new D3D12DownloadTokenImpl(lease, layout);
}
}  // namespace grassland::rhi::backend::d3d12
