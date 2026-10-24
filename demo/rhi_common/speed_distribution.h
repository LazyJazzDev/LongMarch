#pragma once
#include <array>
#include <glm/vec3.hpp>

#include "grassland/rhi/rhi.h"

namespace rhi_demo {
class SpeedDistribution {
 public:
  explicit SpeedDistribution(uint32_t count);
  bool Poll();
  bool Record(grassland::rhi::ICommandContext *commands, grassland::rhi::IBuffer *velocities);
  bool Finish(const std::string &directory);
  bool Ready() const;
  float Maximum() const;
  const std::array<float, 100> &Values() const;

 private:
  std::vector<glm::vec3> velocities_;
  std::array<uint32_t, 100> counts_{};
  std::array<float, 100> normalized_{};
  grassland::rhi::ObjPtr<grassland::rhi::IDownloadToken> download_;
  float maximum_{};
  bool ready_{};
};
}  // namespace rhi_demo
