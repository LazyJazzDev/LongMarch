#pragma once
#include <functional>
#include <mutex>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_download_token.h"

namespace grassland::rhi::backend {
class DownloadToken final : public Object<IDownloadToken> {
 public:
  DownloadToken(uint64_t size, std::function<bool(void *)> copy) : size_(size), copy_(std::move(copy)) {
  }

  SubmissionStatus GetStatus() const override {
    return status_.load(std::memory_order_acquire);
  }

  uint64_t GetSize() const override {
    return size_;
  }

  bool CopyTo(void *destination, uint64_t size) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return destination && size >= size_ && GetStatus() == SUBMISSION_STATUS_COMPLETED && copy_(destination);
  }

  void Complete(SubmissionStatus status) {
    status_.store(status, std::memory_order_release);
  }

 private:
  mutable std::mutex mutex_;
  uint64_t size_;
  std::function<bool(void *)> copy_;
  std::atomic<SubmissionStatus> status_{SUBMISSION_STATUS_PENDING};
};
}  // namespace grassland::rhi::backend
