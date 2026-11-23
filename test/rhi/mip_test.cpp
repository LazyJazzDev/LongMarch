#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "grassland/rhi/rhi.h"
#if defined(LONGMARCH_D3D12_ENABLED)
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#endif

namespace grassland::rhi {
namespace {
std::vector<BackendAPI> MipBackends() {
  std::vector<BackendAPI> result;
#if defined(LONGMARCH_D3D12_ENABLED)
  result.push_back(BACKEND_API_D3D12);
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
  result.push_back(BACKEND_API_VULKAN);
#endif
  return result;
}
}  // namespace

class RhiMip : public testing::TestWithParam<BackendAPI> {};

TEST(RhiMipDescription, LegacySamplerConstructionAndIndependentOverrides) {
  SamplerDesc legacy{SAMPLER_FILTER_NEAREST, SAMPLER_ADDRESS_CLAMP_TO_EDGE, SAMPLER_ADDRESS_MIRRORED_REPEAT};
  EXPECT_EQ(legacy.MinFilter(), SAMPLER_FILTER_NEAREST);
  EXPECT_EQ(legacy.MagFilter(), SAMPLER_FILTER_NEAREST);
  EXPECT_EQ(legacy.MipFilter(), SAMPLER_FILTER_NEAREST);
  EXPECT_EQ(legacy.address_w, SAMPLER_ADDRESS_MIRRORED_REPEAT);
  legacy.filter = SAMPLER_FILTER_LINEAR;
  EXPECT_EQ(legacy.MinFilter(), SAMPLER_FILTER_LINEAR);
  legacy.min_filter = SAMPLER_FILTER_NEAREST;
  EXPECT_EQ(legacy.MinFilter(), SAMPLER_FILTER_NEAREST);
  EXPECT_EQ(legacy.MagFilter(), SAMPLER_FILTER_LINEAR);
  SamplerDesc detailed{SAMPLER_FILTER_NEAREST, SAMPLER_FILTER_LINEAR, SAMPLER_FILTER_NEAREST};
  EXPECT_EQ(detailed.MinFilter(), SAMPLER_FILTER_NEAREST);
  EXPECT_EQ(detailed.MagFilter(), SAMPLER_FILTER_LINEAR);
  EXPECT_EQ(detailed.MipFilter(), SAMPLER_FILTER_NEAREST);
}

TEST_P(RhiMip, EnumeratedCapabilitiesAgreeWithNativeDevice) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  const auto devices = core->GetPhysicalDevices();
  ASSERT_FALSE(devices.empty());
  for (size_t i = 0; i < devices.size(); ++i) {
    ObjPtr<IDevice> device = core->CreateDevice(static_cast<int32_t>(i));
    ASSERT_TRUE(device);
    const auto &mip = devices[i].mip_feature;
    EXPECT_GT(mip.max_image_dimension_2d, 0u);
    EXPECT_GE(mip.max_anisotropy, 1u);
    EXPECT_GT(mip.formats[IMAGE_FORMAT_R8G8B8A8_UNORM].max_mip_levels, 1u);
    EXPECT_TRUE(mip.formats[IMAGE_FORMAT_R8G8B8A8_UNORM].linear_filter);
    EXPECT_FALSE(mip.formats[IMAGE_FORMAT_R32_UINT].linear_filter);
    EXPECT_FALSE(mip.formats[IMAGE_FORMAT_UNKNOWN].sampled);
#if defined(LONGMARCH_D3D12_ENABLED)
    if (GetParam() == BACKEND_API_D3D12) {
      auto native = QueryInterfacePtr<backend::d3d12::ID3D12Device>(device.Get());
      ASSERT_TRUE(native);
      EXPECT_EQ(mip.max_anisotropy, D3D12_MAX_MAXANISOTROPY);
      for (int index = IMAGE_FORMAT_UNKNOWN + 1; index < IMAGE_FORMAT_COUNT; ++index) {
        auto format = backend::d3d12::NativeImageFormat(static_cast<ImageFormat>(index));
        if (format == DXGI_FORMAT_UNKNOWN) {
          EXPECT_EQ(mip.formats[index].max_mip_levels, 0u);
          continue;
        }
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format};
        if (FAILED(
                native->NativeDevice()->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))))
          continue;
        if (support.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D) {
          EXPECT_EQ(mip.formats[index].max_mip_levels > 1, (support.Support1 & D3D12_FORMAT_SUPPORT1_MIP) != 0);
          EXPECT_EQ(mip.formats[index].storage,
                    bool((support.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) &&
                         (support.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE)));
        }
      }
    }
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    if (GetParam() == BACKEND_API_VULKAN) {
      auto native = QueryInterfacePtr<backend::vulkan::IVulkanDevice>(device.Get());
      ASSERT_TRUE(native);
      VkPhysicalDeviceFeatures features{};
      VkPhysicalDeviceProperties properties{};
      vkGetPhysicalDeviceFeatures(native->NativePhysicalDevice(), &features);
      vkGetPhysicalDeviceProperties(native->NativePhysicalDevice(), &properties);
      EXPECT_EQ(mip.max_anisotropy, features.samplerAnisotropy ? uint32_t(properties.limits.maxSamplerAnisotropy) : 1u);
      EXPECT_FLOAT_EQ(mip.max_lod_bias, properties.limits.maxSamplerLodBias);
      EXPECT_EQ(mip.max_image_dimension_2d, properties.limits.maxImageDimension2D);
      for (int index = IMAGE_FORMAT_UNKNOWN + 1; index < IMAGE_FORMAT_COUNT; ++index) {
        auto format = backend::vulkan::NativeImageFormat(static_cast<ImageFormat>(index));
        if (format == VK_FORMAT_UNDEFINED || !mip.formats[index].max_mip_levels)
          continue;
        VkFormatProperties format_properties{};
        vkGetPhysicalDeviceFormatProperties(native->NativePhysicalDevice(), format, &format_properties);
        EXPECT_EQ(mip.formats[index].storage,
                  (format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0);
        EXPECT_EQ(mip.formats[index].linear_filter,
                  bool((format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) &&
                       (format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)));
      }
    }
#endif
  }
}

TEST_P(RhiMip, SamplerLimitsAndInvalidDescriptionsAreRejected) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  const auto devices = core->GetPhysicalDevices();
  ASSERT_FALSE(devices.empty());
  ObjPtr<IDevice> device = core->CreateDevice(0);
  ASSERT_TRUE(device);
  const auto &features = devices[0].mip_feature;
  SamplerDesc desc{SAMPLER_FILTER_LINEAR, SAMPLER_FILTER_LINEAR, SAMPLER_FILTER_LINEAR};
  desc.max_anisotropy = features.max_anisotropy;
  desc.min_lod = 1;
  desc.max_lod = 5;
  desc.mip_lod_bias = features.max_lod_bias;
  ObjPtr<ISampler> sampler = device->CreateSampler(desc);
  ASSERT_TRUE(sampler);
  EXPECT_FLOAT_EQ(sampler->GetDesc().min_lod, 1);
  EXPECT_FLOAT_EQ(sampler->GetDesc().max_lod, 5);
  ++desc.max_anisotropy;
  EXPECT_FALSE(device->CreateSampler(desc));
  desc.max_anisotropy = 0;
  EXPECT_FALSE(device->CreateSampler(desc));
  desc.max_anisotropy = 1;
  desc.mip_lod_bias = features.max_lod_bias + 1;
  EXPECT_FALSE(device->CreateSampler(desc));
  desc.mip_lod_bias = features.min_lod_bias - 1;
  EXPECT_FALSE(device->CreateSampler(desc));
  desc.mip_lod_bias = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(device->CreateSampler(desc));
  desc.mip_lod_bias = 0;
  desc.max_lod = 0;
  EXPECT_FALSE(device->CreateSampler(desc));
  desc.min_lod = 0;
  desc.max_lod = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(device->CreateSampler(desc));
  desc.max_lod = 0;
  desc.address_w = static_cast<SamplerAddressMode>(99);
  EXPECT_FALSE(device->CreateSampler(desc));
  desc.address_w = SAMPLER_ADDRESS_CLAMP_TO_EDGE;
  desc.mag_filter = SAMPLER_FILTER_NEAREST;
  desc.mip_filter = SAMPLER_FILTER_LINEAR;
  sampler = device->CreateSampler(desc);
  ASSERT_TRUE(sampler);
  if (features.max_anisotropy > 1) {
    desc.max_anisotropy = 2;
    EXPECT_FALSE(device->CreateSampler(desc));
  }
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiMip, testing::ValuesIn(MipBackends()));
}  // namespace grassland::rhi
