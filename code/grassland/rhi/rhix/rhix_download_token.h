#pragma once
#include "grassland/rhi/rhi_download_token.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
class DownloadToken : public ObjectContainer {
 public:
  DownloadToken() noexcept;
  DownloadToken(rhi::IDownloadToken *object) noexcept;
  rhi::IDownloadToken *Get() const noexcept;
  rhi::SubmissionStatus GetStatus() const;
  bool WaitHost() const;
  uint64_t GetSize() const;
  bool CopyTo(void *destination, uint64_t size) const;
};
}  // namespace grassland::rhix
