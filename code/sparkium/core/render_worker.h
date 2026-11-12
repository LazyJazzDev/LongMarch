#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

#include "sparkium/core/film.h"

namespace sparkium {

// Backend owns mutable rendering state on this worker thread. It supplies
// Settings (a value snapshot), Configure, GetFilm and Submit(samples, image, hdr,
// dependencies). UI code only exchanges snapshots and completed frame leases.
// Destroy the worker before its borrowed device. Main/Assist may share a native queue.
template <class Backend>
class RenderWorker {
 public:
  using Settings = typename Backend::Settings;
  using Factory = std::function<std::unique_ptr<Backend>()>;

  struct Frame {
    std::shared_ptr<const Film::Output> output;
    rhi::ObjPtr<rhi::ISubmissionToken> rendered;
    uint64_t revision{}, sequence{};
    // Accepted samples for this revision; Film may retain earlier history.
    int samples{};
    bool hdr{};

    rhi::IImage *Image() const {
      return output ? output->Image() : nullptr;
    }
  };

  struct State {
    uint64_t revision{};
    int completed_samples{};
    std::string error;
  };

  RenderWorker(Factory factory,
               const Settings &settings,
               int sample_limit = 0,
               bool hdr = false,
               int samples_per_batch = 4)
      : factory_(std::move(factory)),
        requested_{1, settings, sample_limit, hdr, samples_per_batch},
        thread_([this] {
          try {
            Run();
          } catch (const std::exception &error) {
            Fail(error.what());
          }
        }) {
  }

  ~RenderWorker() {
    Stop();
  }

  void Stop() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    if (thread_.joinable())
      thread_.join();
  }

  // A zero batch size retains the previous request's scheduling policy.
  uint64_t Configure(const Settings &settings, int sample_limit = 0, bool hdr = false, int samples_per_batch = 0) {
    std::lock_guard lock(mutex_);
    requested_ = {requested_.revision + 1, settings, sample_limit, hdr,
                  samples_per_batch ? samples_per_batch : requested_.samples_per_batch};
    condition_.notify_all();
    return requested_.revision;
  }

  State GetState() const {
    std::lock_guard lock(mutex_);
    return state_;
  }

  std::shared_ptr<const Frame> Latest() const {
    std::lock_guard lock(mutex_);
    return latest_;
  }

  // For offline rendering only; interactive frontends use Latest without waiting.
  std::shared_ptr<const Frame> WaitForFrame(uint64_t revision, int samples) {
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [&] {
      return stopping_.load() || !state_.error.empty() ||
             (latest_ && latest_->revision >= revision && latest_->samples >= samples);
    });
    return state_.error.empty() && latest_ && latest_->revision >= revision && latest_->samples >= samples ? latest_
                                                                                                           : nullptr;
  }

  // Calls are ordered on the frontend's presentation queue. Keep the frame lease
  // until this call so it cannot be recycled before its last reader is registered.
  void Presented(const std::shared_ptr<const Frame> &frame, rhi::ObjPtr<rhi::ISubmissionToken> completion) {
    std::lock_guard lock(mutex_);
    for (auto &slot : slots_)
      if (slot.frame == frame) {
        slot.displayed = std::move(completion);
        condition_.notify_all();
        break;
      }
  }

 private:
  struct Request {
    uint64_t revision{};
    Settings settings;
    int sample_limit{};
    bool hdr{};
    int samples_per_batch{};
  };

  struct Slot {
    std::shared_ptr<Frame> frame;
    rhi::ObjPtr<rhi::ISubmissionToken> displayed;
  };

  void Fail(const std::string &error) {
    std::lock_guard lock(mutex_);
    state_.revision = requested_.revision;
    state_.error = error;
    condition_.notify_all();
  }

  void Run() {
    auto backend = factory_();
    if (!backend || !backend->GetFilm() || !backend->GetFilm()->IsValid()) {
      Fail("Cannot create render backend");
      return;
    }
    Request active;
    uint64_t sequence{};
    int accepted_in_revision{};

    struct Batch {
      rhi::ObjPtr<rhi::ISubmissionToken> submitted;
      std::shared_ptr<Frame> frame;
      uint64_t revision;
      int samples;
    };

    std::deque<Batch> pending;
    auto retire = [&]() {
      auto batch = std::move(pending.front());
      pending.pop_front();
      if (!batch.submitted->WaitHost()) {
        Fail("GPU render submission failed");
        return false;
      }
      std::lock_guard lock(mutex_);
      if (batch.revision == requested_.revision) {
        state_.completed_samples = batch.samples;
        if (batch.frame && (!latest_ || batch.frame->sequence > latest_->sequence))
          latest_ = batch.frame;
      }
      condition_.notify_all();
      return true;
    };
    while (!stopping_.load()) {
      Request request;
      {
        std::lock_guard lock(mutex_);
        request = requested_;
      }
      if (request.revision != active.revision) {
        while (!pending.empty())
          if (!retire())
            return;
        if (request.samples_per_batch <= 0 || request.samples_per_batch > 32) {
          Fail("Invalid render batch size");
          return;
        }
        if (request.sample_limit < 0 || !backend->Configure(request.settings)) {
          Fail("Invalid render configuration");
          return;
        }
        active = request;
        accepted_in_revision = 0;
        std::lock_guard lock(mutex_);
        if (requested_.revision == active.revision)
          state_ = {active.revision, 0, {}};
      }
      auto *film = backend->GetFilm();
      if (pending.size() == 2 || (active.sample_limit && accepted_in_revision >= active.sample_limit)) {
        if (!pending.empty()) {
          if (!retire())
            return;
        } else {
          std::unique_lock lock(mutex_);
          condition_.wait(lock, [&] { return stopping_.load() || requested_.revision != active.revision; });
        }
        continue;
      }
      std::shared_ptr<Frame> frame;
      rhi::ObjPtr<rhi::ISubmissionToken> displayed;
      {
        std::lock_guard lock(mutex_);
        for (auto &slot : slots_) {
          if (slot.frame && slot.frame.use_count() != 1)
            continue;
          if (slot.displayed && slot.displayed->GetStatus() == rhi::SUBMISSION_STATUS_PENDING)
            continue;
          if (slot.displayed && slot.displayed->GetStatus() != rhi::SUBMISSION_STATUS_COMPLETED) {
            state_.error = "Presentation submission failed";
            condition_.notify_all();
            return;
          }
          if (!slot.frame)
            slot.frame = std::make_shared<Frame>();
          else
            slot.frame->output.reset();
          frame = slot.frame;
          displayed = slot.displayed;
          break;
        }
      }
      if (frame) {
        frame->output = film->AcquireOutput(active.hdr);
        if (!frame->output)
          frame.reset();
      }
      if (!film->IsValid()) {
        Fail("Cannot acquire Film output image");
        return;
      }
      const int samples = active.sample_limit
                              ? std::min(active.samples_per_batch, active.sample_limit - accepted_in_revision)
                              : active.samples_per_batch;
      // A held display image must not stall accumulation. Intermediate batches
      // may omit development; the final finite batch always publishes an image.
      const bool final_batch = active.sample_limit && accepted_in_revision + samples == active.sample_limit;
      if (!frame && final_batch) {
        if (!pending.empty()) {
          if (!retire())
            return;
        } else {
          std::unique_lock lock(mutex_);
          condition_.wait_for(lock, std::chrono::milliseconds(1));
        }
        continue;
      }
      rhi::SubmitInfo dependencies;
      if (displayed)
        dependencies.waits.push_back(displayed->GetEvent());
      const int accumulated_before = film->info.accumulated_samples;
      auto submitted = backend->Submit(samples, frame ? frame->Image() : nullptr, active.hdr, dependencies);
      if (!submitted) {
        Fail("Cannot submit rendering");
        return;
      }
      const int accepted_samples = film->info.accumulated_samples - accumulated_before;
      if (accepted_samples <= 0 || accepted_samples > samples) {
        Fail("Render submission did not accept the requested sample progress");
        return;
      }
      // A configuration may preserve temporal accumulation. Its finite budget
      // counts newly accepted samples, independently of the Film history.
      accepted_in_revision += accepted_samples;
      if (frame) {
        frame->revision = active.revision;
        frame->sequence = ++sequence;
        frame->hdr = active.hdr;
        frame->rendered = submitted;
        frame->samples = accepted_in_revision;
      }
      pending.push_back({std::move(submitted), std::move(frame), active.revision, accepted_in_revision});
    }
    while (!pending.empty())
      if (!retire())
        return;
  }

  Factory factory_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  Request requested_;
  State state_;
  std::array<Slot, 3> slots_;
  std::shared_ptr<Frame> latest_;
  std::atomic<bool> stopping_{};
  std::thread thread_;
};
}  // namespace sparkium
