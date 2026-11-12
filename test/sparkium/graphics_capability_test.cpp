#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>

#include "grassland/graphics/backend/backend.h"
#include "grassland/graphics/graphics.h"
using namespace grassland;
namespace vulkan = grassland::graphics::backend::vulkan;

class GraphicsCapabilities : public testing::TestWithParam<graphics::BackendAPI> {};

TEST_P(GraphicsCapabilities, RayQueryCapabilityMatchesDevice) {
  std::unique_ptr<graphics::Core> graphics;
  ASSERT_EQ(graphics::CreateCore(GetParam(), {}, &graphics), 0);
  ASSERT_EQ(graphics->InitializeLogicalDeviceAutoSelect(false), 0);
#if defined(LONGMARCH_VULKAN_ENABLED)
  if (auto *vk = dynamic_cast<graphics::backend::VulkanCore *>(graphics.get())) {
    const auto physical = vk->PhysicalDevice();
    EXPECT_EQ(graphics->DeviceRayQuerySupport(), vulkan::SupportRayQuery(physical));
    if (vulkan::SupportRayQuery(physical)) {
      vulkan::DeviceFeatureRequirement requirement{};
      requirement.enable_rayquery_extension = true;
      auto info = requirement.GenerateRecommendedDeviceCreateInfo(physical);
      auto has_extension = [&](const char *name) {
        return std::any_of(info.extensions.begin(), info.extensions.end(),
                           [&](const char *extension) { return std::strcmp(extension, name) == 0; });
      };
      EXPECT_TRUE(has_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME));
      EXPECT_TRUE(has_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME));
      EXPECT_FALSE(has_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME));
      EXPECT_NE(requirement.GetVmaAllocatorCreateFlags() & VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT, 0);
      VkDevice query_device = VK_NULL_HANDLE;
      auto create_info = info.CompileVkDeviceCreateInfo(false, physical);
      ASSERT_EQ(vkCreateDevice(physical, &create_info, nullptr, &query_device), VK_SUCCESS);
      EXPECT_NE(vkGetDeviceProcAddr(query_device, "vkCmdBuildAccelerationStructuresKHR"), nullptr);
      EXPECT_EQ(vkGetDeviceProcAddr(query_device, "vkCmdTraceRaysKHR"), nullptr);
      vkDestroyDevice(query_device, nullptr);
    }
  }
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
  if (auto *dx = dynamic_cast<graphics::backend::D3D12Core *>(graphics.get())) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options{};
    const auto result = dx->Device()->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options, sizeof(options));
    EXPECT_EQ(graphics->DeviceRayQuerySupport(),
              SUCCEEDED(result) && options.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1);
  }
#endif
}

#if defined(LONGMARCH_VULKAN_ENABLED)
INSTANTIATE_TEST_SUITE_P(Vulkan, GraphicsCapabilities, testing::Values(graphics::BACKEND_API_VULKAN));
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
INSTANTIATE_TEST_SUITE_P(D3D12, GraphicsCapabilities, testing::Values(graphics::BACKEND_API_D3D12));
#endif
