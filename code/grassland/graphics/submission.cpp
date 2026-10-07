#include "grassland/graphics/submission.h"

#include <atomic>
#include <exception>
#include <stdexcept>
#include <utility>

namespace grassland::graphics {
namespace detail {
class InFlightSubmission {
 public:
  InFlightSubmission(uint64_t queue,
                     uint64_t value,
                     std::function<bool()> poll,
                     std::function<void()> wait,
                     std::vector<std::function<void()>> callbacks)
      : queue_(queue),
        value_(value),
        poll_(std::move(poll)),
        wait_(std::move(wait)),
        callbacks_(std::move(callbacks)) {
  }

  bool Poll(bool wait) {
    if (!finished_) {
      try {
        if (wait)
          wait_();
        else if (!poll_())
          return false;
      } catch (...) {
        error_ = std::current_exception();
      }
      finished_ = true;
      poll_ = {};
      wait_ = {};
      // Run every callback exactly once, even when another callback throws.
      // GPU failure only releases captures; success callbacks must not run.
      auto callbacks = std::move(callbacks_);
      if (!error_) {
        for (auto &callback : callbacks) {
          try {
            callback();
          } catch (...) {
            if (!error_)
              error_ = std::current_exception();
          }
        }
      }
    }
    if (error_)
      std::rethrow_exception(error_);
    return true;
  }

  uint64_t Queue() const {
    return queue_;
  }

  uint64_t Value() const {
    return value_;
  }

 private:
  uint64_t queue_, value_;
  bool finished_{false};
  std::exception_ptr error_;
  std::function<bool()> poll_;
  std::function<void()> wait_;
  std::vector<std::function<void()>> callbacks_;
};

SubmissionQueue::SubmissionQueue() {
  static std::atomic<uint64_t> next_queue{1};
  queue_id_ = next_queue.fetch_add(1);
}

SubmissionQueue::~SubmissionQueue() {
  // Each backend must call Shutdown before releasing its native device.
  // Core also owns this object, so no native callbacks may survive destruction.
  Shutdown();
}

Submission SubmissionQueue::Track(std::function<bool()> poll,
                                  std::function<void()> wait,
                                  std::vector<std::function<void()>> callbacks,
                                  std::function<void()> submit) {
  auto state = std::make_shared<InFlightSubmission>(queue_id_, next_value_++, std::move(poll), std::move(wait),
                                                    std::move(callbacks));
  pending_.push_back(state);
  try {
    if (submit)
      submit();
  } catch (...) {
    pending_.pop_back();
    throw;
  }
  return Submission(std::move(state));
}

void SubmissionQueue::Collect() {
  std::exception_ptr error;
  for (auto it = pending_.begin(); it != pending_.end();) {
    bool finished = false;
    try {
      finished = (*it)->Poll(false);
    } catch (...) {
      finished = true;
      if (!error)
        error = std::current_exception();
    }
    if (finished)
      it = pending_.erase(it);
    else
      ++it;
  }
  if (error)
    std::rethrow_exception(error);
}

void SubmissionQueue::WaitAll() {
  auto pending = std::move(pending_);
  std::exception_ptr error;
  for (auto &state : pending) {
    try {
      state->Poll(true);
    } catch (...) {
      if (!error)
        error = std::current_exception();
    }
  }
  if (error)
    std::rethrow_exception(error);
}

void SubmissionQueue::Shutdown() noexcept {
  try {
    WaitAll();
  } catch (...) {
    // Errors remain in the tickets; destruction itself cannot throw.
  }
}

void InFlightSlot::Wait() const {
  if (last_use_.IsValid())
    last_use_.Wait();
}

void InFlightSlot::Retire(Submission submission) {
  if (!submission.IsValid())
    throw std::invalid_argument("invalid slot submission");
  if (last_use_.IsValid() && !last_use_.IsComplete())
    throw std::logic_error("in-flight slot reused before completion");
  last_use_ = std::move(submission);
}
}  // namespace detail

Submission::Submission(std::shared_ptr<detail::InFlightSubmission> state) : state_(std::move(state)) {
}

detail::InFlightSubmission &Submission::State() const {
  if (!state_)
    throw std::logic_error("invalid submission ticket");
  return *state_;
}

bool Submission::IsValid() const {
  return bool(state_);
}

bool Submission::IsComplete() const {
  return State().Poll(false);
}

void Submission::Wait() const {
  State().Poll(true);
}

uint64_t Submission::QueueId() const {
  return State().Queue();
}

uint64_t Submission::Value() const {
  return State().Value();
}
}  // namespace grassland::graphics
