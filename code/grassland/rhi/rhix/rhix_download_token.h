#pragma once
#include "grassland/rhi/rhi_download_token.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
class DownloadToken : public ObjectContainer {
 public:
  DownloadToken() noexcept = default;

  DownloadToken(rhi::IDownloadToken *object) noexcept : ObjectContainer(object) {
  }

  rhi::IDownloadToken *Get() const noexcept {
    return static_cast<rhi::IDownloadToken *>(GetObject());
  }

  rhi::SubmissionStatus GetStatus() const {
    return *this ? Get()->GetStatus() : rhi::SUBMISSION_STATUS_CANCELED;
  }

  uint64_t GetSize() const {
    return *this ? Get()->GetSize() : 0;
  }

  bool CopyTo(void *destination, uint64_t size) const {
    return *this && Get()->CopyTo(destination, size);
  }
};
}  // namespace grassland::rhix
