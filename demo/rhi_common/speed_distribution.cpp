#include "speed_distribution.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <glm/geometric.hpp>
#include <iostream>

namespace rhi_demo {
SpeedDistribution::SpeedDistribution(uint32_t count) : velocities_(count) {
}

bool SpeedDistribution::Poll() {
  using namespace grassland::rhi;
  if (!download_ || download_->GetStatus() == SUBMISSION_STATUS_PENDING)
    return true;
  if (!download_->CopyTo(velocities_.data(), velocities_.size() * sizeof(glm::vec3)))
    return false;
  download_.Release();
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

bool SpeedDistribution::Record(grassland::rhi::ICommandContext *commands, grassland::rhi::IBuffer *velocities) {
  if (download_)
    return true;
  download_ = commands->CmdDownloadBuffer(velocities, 0, velocities_.size() * sizeof(glm::vec3));
  return bool(download_);
}

bool SpeedDistribution::Finish(const std::string &directory) {
  if (download_ && !download_->WaitHost())
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

bool SpeedDistribution::Ready() const {
  return ready_;
}

float SpeedDistribution::Maximum() const {
  return maximum_;
}

const std::array<float, 100> &SpeedDistribution::Values() const {
  return normalized_;
}
}  // namespace rhi_demo
