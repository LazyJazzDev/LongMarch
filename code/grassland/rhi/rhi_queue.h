#pragma once
#include "grassland/rhi/rhi_util.h"

namespace grassland::rhi {
class SubmissionToken {
 public:
  SubmissionStatus GetStatus() const;
  bool WaitHost() const;
  explicit operator bool() const noexcept;

 private:
  // A token owns only completion state, never a native queue or command payload.
  using Interface = IObject;
  ObjectRef object_;
  friend struct backend::Access;
};

struct SubmitInfo {
  std::vector<SubmissionToken> waits;
};

class RecordedCommands;

class Queue {
 public:
  // Always consumes commands. An empty token reports submission failure.
  SubmissionToken Submit(RecordedCommands &&commands, const SubmitInfo &info = {}) const;
  QueueType GetType() const;
  bool SharesPhysicalQueue(const Queue &other) const;
  explicit operator bool() const noexcept;

 private:
  using Interface = backend::IQueue;
  ObjectRef object_;
  friend struct backend::Access;
};

}  // namespace grassland::rhi
