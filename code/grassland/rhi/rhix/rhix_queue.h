#pragma once
#include "grassland/rhi/rhi.h"
#include "grassland/rhi/rhi_util.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
class SubmissionToken : public ObjectContainer {
 public:
  SubmissionToken(rhi::ISubmissionToken *object) noexcept;
  rhi::ISubmissionToken *Get() const noexcept;
  SubmissionToken() noexcept;
  rhi::SubmissionStatus GetStatus() const;
  bool WaitHost() const;
};

struct SubmitInfo {
  std::vector<SubmissionToken> waits;
};

class CommandContext;

class Queue : public ObjectContainer {
 public:
  Queue(rhi::IQueue *object) noexcept;
  rhi::IQueue *Get() const noexcept;
  Queue() noexcept;
  // Ends recording and consumes the context, including on failure.
  // An empty token reports submission failure.
  SubmissionToken Submit(CommandContext &&commands, const SubmitInfo &info = {}) const;
  rhi::QueueType GetType() const;
  bool SharesPhysicalQueue(const Queue &other) const;
};

}  // namespace grassland::rhix
