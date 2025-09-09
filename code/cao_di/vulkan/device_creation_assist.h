#pragma once

#include "cao_di/vulkan/instance.h"
#include "cao_di/vulkan/physical_device.h"
#include "cao_di/vulkan/validation_layer.h"
#include "map"
#include "memory"

namespace CD::vulkan {

struct DeviceFeatureRequirement {
  bool enable_raytracing_extension{false};

  class DeviceCreateInfo GenerateRecommendedDeviceCreateInfo(const PhysicalDevice &physical_device) const;

  VmaAllocatorCreateFlags GetVmaAllocatorCreateFlags() const;
};

struct DeviceFeatureContainerBase {
  virtual ~DeviceFeatureContainerBase() = default;
  virtual void *LinkNext(void *pNext) = 0;
  [[nodiscard]] virtual std::unique_ptr<DeviceFeatureContainerBase> Duplicate() const = 0;
};

template <class FeatureType>
struct DeviceFeatureContainer : public DeviceFeatureContainerBase {
  FeatureType feature;

  DeviceFeatureContainer(const FeatureType &feature) : feature(feature) {
  }

  void *LinkNext(void *pNext) override {
    feature.pNext = pNext;
    return reinterpret_cast<void *>(&feature);
  }

  std::unique_ptr<DeviceFeatureContainerBase> Duplicate() const override {
    return std::make_unique<DeviceFeatureContainer<FeatureType>>(feature);
  }
};

struct DeviceCreateInfo {
  std::vector<std::unique_ptr<DeviceFeatureContainerBase>> features;
  std::vector<const char *> extensions;
  std::map<int, std::vector<float>> queue_families;

 private:
  std::vector<VkDeviceQueueCreateInfo> queue_create_infos_;
  VkPhysicalDeviceFeatures physical_device_features_{};
  std::vector<const char *> enabled_layers_;

 public:
  DeviceCreateInfo(const DeviceCreateInfo &other) {
    for (auto &feature : other.features) {
      features.emplace_back(feature->Duplicate());
    }
    extensions = other.extensions;
    queue_families = other.queue_families;
  }

  DeviceCreateInfo() = default;

  DeviceCreateInfo(DeviceCreateInfo &&) noexcept = default;

  template <class FeatureType>
  void AddFeature(const FeatureType &feature) {
    features.emplace_back(std::make_unique<DeviceFeatureContainer<FeatureType>>(feature));
  }

  void AddExtension(const char *extension) {
    extensions.push_back(extension);
  }

  void AddQueueFamily(int family_index, const std::vector<float> &priorities) {
    queue_families[family_index] = priorities;
  }

  VkDeviceCreateInfo CompileVkDeviceCreateInfo(bool enable_validation_layers, const PhysicalDevice &physical_device);
};
}  // namespace CD::vulkan
