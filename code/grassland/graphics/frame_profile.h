#pragma once

#include <chrono>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "grassland/graphics/core.h"
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/graphics/backend/vulkan/vulkan_command_context.h"
#endif

namespace grassland::graphics {

// Opt-in, single-thread frame profiling. GPU results must be read after WaitGPU.
// CPU scopes may nest: inclusive CPU and GPU durations must not be added together.
class FrameProfile {
 public:
  explicit FrameProfile(graphics::Core *core, bool gpu_timestamps = true) {
#if defined(LONGMARCH_VULKAN_ENABLED)
    auto vk = dynamic_cast<graphics::backend::VulkanCore *>(core);
    if (!vk)
      throw std::runtime_error("frame GPU profiling currently requires Vulkan");
    device_ = vk->Device()->Handle();
    const auto properties = vk->Device()->PhysicalDevice().GetPhysicalDeviceProperties();
    device_name = properties.deviceName;
    if (!gpu_timestamps)
      return;
    period_ = properties.limits.timestampPeriod;
    bits_ = vk->Device()
                ->PhysicalDevice()
                .GetQueueFamilyProperties()[vk->GraphicsQueue()->QueueFamilyIndex()]
                .timestampValidBits;
    if (!bits_ || period_ <= 0)
      throw std::runtime_error("GPU queue does not support timestamps");
    VkQueryPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = kQueries;
    if (vkCreateQueryPool(device_, &info, nullptr, &pool_) != VK_SUCCESS)
      throw std::runtime_error("could not create timestamp query pool");
#else
    throw std::runtime_error("frame GPU profiling currently requires Vulkan");
#endif
  }
  ~FrameProfile() {
    if (active == this)
      active = nullptr;
#if defined(LONGMARCH_VULKAN_ENABLED)
    if (pool_) {
      vkDeviceWaitIdle(device_);
      vkDestroyQueryPool(device_, pool_, nullptr);
    }
#endif
  }
  FrameProfile(const FrameProfile &) = delete;
  FrameProfile &operator=(const FrameProfile &) = delete;
  void Begin(bool gpu_timestamps = true) {
    if (active)
      throw std::runtime_error("frame profile already active");
    cpu_ms.clear();
    gpu_ms.clear();
    counters.clear();
    names_.clear();
    timestamps_enabled_ = gpu_timestamps;
#if defined(LONGMARCH_VULKAN_ENABLED)
    counters["gpu_timestamps"] = pool_ && timestamps_enabled_;
#endif
    active = this;
  }
  void Finish() {
    active = nullptr;
#if defined(LONGMARCH_VULKAN_ENABLED)
    std::vector<uint64_t> values(names_.size() * 2);
    if (!values.empty() && vkGetQueryPoolResults(device_, pool_, 0, values.size(), values.size() * sizeof(uint64_t),
                                                 values.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
      throw std::runtime_error("GPU timestamps unavailable; Finish requires GPU completion");
    const uint64_t mask = bits_ == 64 ? ~uint64_t(0) : (uint64_t(1) << bits_) - 1;
    for (size_t i = 0; i < names_.size(); ++i)
      gpu_ms[names_[i]] += ((values[i * 2 + 1] - values[i * 2]) & mask) * period_ / 1e6;
#endif
  }
  int BeginGpu(graphics::CommandContext *commands, const std::string &name) {
#if defined(LONGMARCH_VULKAN_ENABLED)
    if (!pool_ || !timestamps_enabled_)
      return -1;
    auto vk = dynamic_cast<graphics::backend::VulkanCommandContext *>(commands);
    if (!vk || (names_.size() + 1) * 2 > kQueries)
      throw std::runtime_error("invalid GPU profiling scope");
    const int index = names_.size() * 2;
    names_.push_back(name);
    vk->CmdTimestamp(pool_, index, index == 0 ? kQueries : 0);
    return index;
#else
    return 0;
#endif
  }
  void EndGpu(graphics::CommandContext *commands, int index) {
#if defined(LONGMARCH_VULKAN_ENABLED)
    if (index < 0)
      return;
    dynamic_cast<graphics::backend::VulkanCommandContext *>(commands)->CmdTimestamp(pool_, index + 1);
#endif
  }
  inline static thread_local FrameProfile *active = nullptr;
  std::map<std::string, double> cpu_ms, gpu_ms;
  std::map<std::string, uint64_t> counters;
  std::string device_name;

 private:
  static constexpr uint32_t kQueries = 64;
  std::vector<std::string> names_;
  bool timestamps_enabled_{true};
#if defined(LONGMARCH_VULKAN_ENABLED)
  VkDevice device_{};
  VkQueryPool pool_{};
  double period_{};
  uint32_t bits_{};
#endif
};

class CpuProfileScope {
 public:
  explicit CpuProfileScope(const char *name) : profile_(FrameProfile::active), name_(name) {
    if (profile_)
      start_ = std::chrono::steady_clock::now();
  }
  ~CpuProfileScope() {
    End();
  }
  void End() {
    if (profile_) {
      profile_->cpu_ms[name_] +=
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
      profile_ = nullptr;
    }
  }

 private:
  FrameProfile *profile_;
  const char *name_;
  std::chrono::steady_clock::time_point start_;
};

class GpuProfileScope {
 public:
  GpuProfileScope(graphics::CommandContext *commands, const char *name)
      : profile_(FrameProfile::active), commands_(commands) {
    if (profile_)
      index_ = profile_->BeginGpu(commands_, name);
  }
  ~GpuProfileScope() {
    End();
  }
  void End() {
    if (profile_) {
      profile_->EndGpu(commands_, index_);
      profile_ = nullptr;
    }
  }

 private:
  FrameProfile *profile_;
  graphics::CommandContext *commands_;
  int index_{};
};
}  // namespace grassland::graphics
