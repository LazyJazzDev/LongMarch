#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace grassland::graphics {

class Submission;

namespace detail {
class InFlightSubmission;

// Host access to a queue and its tickets must be externally serialized.
class SubmissionQueue {
 public:
  SubmissionQueue();
  ~SubmissionQueue();
  SubmissionQueue(const SubmissionQueue &) = delete;
  SubmissionQueue &operator=(const SubmissionQueue &) = delete;

  Submission Track(std::function<bool()> poll,
                   std::function<void()> wait,
                   std::vector<std::function<void()>> callbacks,
                   std::function<void()> submit = {});
  void Collect();
  void WaitAll();
  // Call while the native device is still alive. Never throws from destruction.
  void Shutdown() noexcept;

 private:
  uint64_t queue_id_;
  uint64_t next_value_{1};
  std::vector<std::shared_ptr<InFlightSubmission>> pending_;
};
}  // namespace detail

// A copyable completion ticket, independent of presentation or frame numbers.
// A default ticket is invalid; querying/waiting on it throws logic_error.
// GPU or completion-callback errors are retained and rethrown on every query.
class Submission {
 public:
  Submission() = default;
  bool IsValid() const;
  bool IsComplete() const;
  void Wait() const;
  uint64_t QueueId() const;
  uint64_t Value() const;

 private:
  friend Submission detail::SubmissionQueue::Track(std::function<bool()>,
                                                   std::function<void()>,
                                                   std::vector<std::function<void()>>,
                                                   std::function<void()>);
  explicit Submission(std::shared_ptr<detail::InFlightSubmission> state);
  detail::InFlightSubmission &State() const;
  std::shared_ptr<detail::InFlightSubmission> state_;
};

}  // namespace grassland::graphics
