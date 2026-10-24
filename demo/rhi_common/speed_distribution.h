#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <iostream>

#include "grassland/rhi/rhi.h"

namespace rhi_demo {
class SpeedDistribution {
 public:
  explicit SpeedDistribution(uint32_t count) : velocities_(count) {
  }

  bool Poll() {
    using namespace grassland::rhi;
    if (!download_ || download_->GetStatus() == SUBMISSION_STATUS_PENDING)
      return true;
    if (!download_->CopyTo(velocities_.data(), velocities_.size() * sizeof(glm::vec3)))
      return false;
    download_.Release();
    completion_.Release();
    maximum_ = 0;
    for (const auto &v : velocities_) {
      const float speed = glm::length(v);
      if (!std::isfinite(speed)) {
        std::cerr << "nbody: non-finite particle speed\n";
        return false;
      }
      maximum_ = std::max(maximum_, speed);
    }
    counts_.fill(0);
    for (const auto &v : velocities_) {
      const size_t bin =
          maximum_ > 0 ? std::min(size_t(glm::length(v) / maximum_ * counts_.size()), counts_.size() - 1) : 0;
      ++counts_[bin];
    }
    const auto peak = *std::max_element(counts_.begin(), counts_.end());
    for (size_t i = 0; i < counts_.size(); ++i)
      normalized_[i] = peak ? float(counts_[i]) / peak : 0;
    ready_ = true;
    return true;
  }

  bool Record(grassland::rhi::ICommandContext *commands, grassland::rhi::IBuffer *velocities) {
    if (download_)
      return true;
    download_ = commands->CmdDownloadBuffer(velocities, 0, velocities_.size() * sizeof(glm::vec3));
    recorded_ = bool(download_);
    return recorded_;
  }

  void Submitted(grassland::rhi::ISubmissionToken *completion) {
    if (recorded_) {
      completion_ = grassland::rhi::ObjPtr<grassland::rhi::ISubmissionToken>::Retain(completion);
      recorded_ = false;
    }
  }

  bool Finish(const std::string &directory) {
    if (completion_ && !completion_->WaitHost())
      return false;
    if (!Poll())
      return false;
    if (!ready_)
      return true;
    std::cout << "nbody: downloaded " << velocities_.size() << " particle velocities, max speed " << maximum_
              << ", 100 histogram bins\n";
    if (!directory.empty()) {
      std::error_code error;
      const auto path = std::filesystem::u8path(directory);
      std::filesystem::create_directories(path, error);
      if (error)
        return false;
      std::ofstream file(path / "nbody_speeds.csv");
      file << "speed_min,speed_max,count,normalized\n";
      for (size_t i = 0; i < counts_.size(); ++i)
        file << maximum_ * i / counts_.size() << ',' << maximum_ * (i + 1) / counts_.size() << ',' << counts_[i] << ','
             << normalized_[i] << '\n';
      file.close();
      return bool(file);
    }
    return true;
  }

  bool Ready() const {
    return ready_;
  }

  float Maximum() const {
    return maximum_;
  }

  const std::array<float, 100> &Values() const {
    return normalized_;
  }

 private:
  std::vector<glm::vec3> velocities_;
  std::array<uint32_t, 100> counts_{};
  std::array<float, 100> normalized_{};
  grassland::rhi::ObjPtr<grassland::rhi::IDownloadToken> download_;
  grassland::rhi::ObjPtr<grassland::rhi::ISubmissionToken> completion_;
  float maximum_{};
  bool ready_{}, recorded_{};
};
}  // namespace rhi_demo
