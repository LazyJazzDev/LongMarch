#pragma once
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
struct QueueState final {
  QueueState();
  ~QueueState();

  struct Submission {
    ObjectPtr<IBackendCommandContext> commands;
    ObjectPtr<IBackendSubmissionToken> completion;
  };

  std::list<Submission> pending_{};
  IBackendDevice *device_{};
  QueueType type_{};
  // Logical queues that share a native queue also share its host synchronization.
  std::shared_ptr<std::mutex> mutex_{};
  uint64_t serial_{};
};
}  // namespace grassland::rhi::backend
