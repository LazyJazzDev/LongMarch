#include "grassland/rhi/rhix/rhix_download_token.h"

namespace grassland::rhix {
DownloadToken::DownloadToken() noexcept = default;

DownloadToken::DownloadToken(rhi::IDownloadToken *object) noexcept : ObjectContainer(object) {
}

rhi::IDownloadToken *DownloadToken::Get() const noexcept {
  return static_cast<rhi::IDownloadToken *>(GetObject());
}

rhi::SubmissionStatus DownloadToken::GetStatus() const {
  return *this ? Get()->GetStatus() : rhi::SUBMISSION_STATUS_CANCELED;
}

bool DownloadToken::WaitHost() const {
  return *this && Get()->WaitHost();
}

uint64_t DownloadToken::GetSize() const {
  return *this ? Get()->GetSize() : 0;
}

bool DownloadToken::CopyTo(void *destination, uint64_t size) const {
  return *this && Get()->CopyTo(destination, size);
}
}  // namespace grassland::rhix
