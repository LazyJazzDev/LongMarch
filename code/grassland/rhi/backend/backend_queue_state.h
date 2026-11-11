#pragma once
#include "grassland/rhi/backend/backend_queue.h"

namespace grassland::rhi::backend {
struct QueueState final {
  QueueState();
  ~QueueState();

  struct Submission {
    ObjectPtr<CommandContext> commands;
    ObjectPtr<SubmissionToken> completion;
  };

  std::list<Submission> pending_{};
  Device *device_{};
  QueueType type_{};
  // Logical queues that share a native queue also share its host synchronization.
  std::shared_ptr<std::mutex> mutex_{};
  uint64_t serial_{};
};
}  // namespace grassland::rhi::backend
