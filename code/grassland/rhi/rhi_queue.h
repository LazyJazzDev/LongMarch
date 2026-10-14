#pragma once
#include "grassland/rhi/rhi_object_container.h"
#include "grassland/rhi/rhi_util.h"

namespace grassland::rhi {
class SubmissionToken : public ObjectContainer {
 public:
  SubmissionToken() noexcept;
  SubmissionStatus GetStatus() const;
  bool WaitHost() const;

 private:
  // A token owns only completion state, never a native queue or command payload.
  using Interface = IObject;
  friend struct backend::Access;
};

struct SubmitInfo {
  std::vector<SubmissionToken> waits;
};

class RecordedCommands;

class Queue : public ObjectContainer {
 public:
  Queue() noexcept;
  // Always consumes commands. An empty token reports submission failure.
  SubmissionToken Submit(RecordedCommands &&commands, const SubmitInfo &info = {}) const;
  QueueType GetType() const;
  bool SharesPhysicalQueue(const Queue &other) const;

 private:
  using Interface = backend::IQueue;
  friend struct backend::Access;
};

}  // namespace grassland::rhi
