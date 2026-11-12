#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

namespace demo {

// Opt-in frame-loop measurements, excluding initialization and 120 warmup frames.
// Call Finish only after draining submitted GPU work. This measures application
// throughput (including event handling and presentation), not GPU timestamps.
class FrameBenchmark {
 public:
  static bool PresentFrames() {
    static const bool present = std::getenv("LONGMARCH_DEMO_OFFSCREEN") == nullptr;
    return present;
  }

  bool BeginFrame(int frame) {
    if (!enabled_)
      return false;
    if (frame < 120) {
      // Let startup animations settle even on GPUs that render 120 frames in
      // a few milliseconds. None of this delay belongs to measured samples.
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      return false;
    }
    frame_start_ = Clock::now();
    if (samples_.empty())
      start_ = frame_start_;
    return true;
  }

  void EndFrame(bool measured) {
    if (measured)
      samples_.push_back(Milliseconds(Clock::now() - frame_start_));
  }

  void Finish() {
    if (samples_.empty())
      return;
    const double elapsed = Milliseconds(Clock::now() - start_);
    std::sort(samples_.begin(), samples_.end());
    std::cout << "DEMO_BENCHMARK {\"frames\":" << samples_.size() << ",\"elapsed_ms\":" << elapsed
              << ",\"fps\":" << samples_.size() * 1000.0 / elapsed
              << ",\"cpu_p50_ms\":" << samples_[samples_.size() / 2]
              << ",\"cpu_p95_ms\":" << samples_[samples_.size() * 95 / 100] << "}\n";
  }

 private:
  using Clock = std::chrono::steady_clock;

  static double Milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
  }

  bool enabled_{std::getenv("LONGMARCH_DEMO_BENCHMARK") != nullptr};
  Clock::time_point start_{}, frame_start_{};
  std::vector<double> samples_;
};

}  // namespace demo
