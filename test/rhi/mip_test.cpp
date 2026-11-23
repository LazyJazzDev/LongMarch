#include <gtest/gtest.h>

#include <algorithm>
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

TEST_P(RhiMip, CreationResolvesChainsAndRejectsInvalidLevels) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IImage> legacy = device->CreateImage({19, 7, IMAGE_FORMAT_R8G8B8A8_UNORM});
  ASSERT_TRUE(legacy);
  EXPECT_EQ(legacy->GetDesc().mip_levels, 1u);
  ObjPtr<IImage> chain = device->CreateImage({19, 7, IMAGE_FORMAT_R8G8B8A8_UNORM, 0});
  ASSERT_TRUE(chain);
  EXPECT_EQ(chain->GetDesc().mip_levels, 5u);
  ObjPtr<IImage> partial = device->CreateImage({19, 7, IMAGE_FORMAT_R8G8B8A8_UNORM, 3});
  ASSERT_TRUE(partial);
  EXPECT_EQ(partial->GetDesc().mip_levels, 3u);
  EXPECT_FALSE(device->CreateImage({19, 7, IMAGE_FORMAT_R8G8B8A8_UNORM, 6}));
  EXPECT_FALSE(device->CreateImage({0, 7, IMAGE_FORMAT_R8G8B8A8_UNORM, 0}));
  EXPECT_FALSE(device->CreateImage({1, 1, IMAGE_FORMAT_R8G8B8A8_UNORM, UINT32_MAX}));
  EXPECT_FALSE(device->CreateImage({1, 1, static_cast<ImageFormat>(-1), 1}));
}

TEST_P(RhiMip, PaddedBatchUploadsSnapshotEveryLevelAndSingleUpdatesAreIsolated) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ObjPtr<IQueue> assist = device->GetQueue(QUEUE_TYPE_ASSIST);
  ASSERT_TRUE(queue && assist);
  for (const auto dimensions : {std::array<uint32_t, 2>{19, 7}, {1, 9}, {8, 1}, {1, 1}}) {
    ObjPtr<IImage> image = device->CreateImage({dimensions[0], dimensions[1], IMAGE_FORMAT_R8G8B8A8_UNORM, 0});
    ASSERT_TRUE(image);
    const auto levels = image->GetDesc().mip_levels;
    std::vector<std::vector<uint8_t>> padded(levels), expected(levels);
    std::vector<ImageUploadInfo> uploads;
    for (uint32_t mip = 0; mip < levels; ++mip) {
      ImageDesc level;
      ASSERT_TRUE(ImageMipDesc(image->GetDesc(), mip, level));
      const size_t row = level.width * 4;
      padded[mip].resize((row + 12) * level.height, 0xee);
      expected[mip].resize(row * level.height);
      for (uint32_t y = 0; y < level.height; ++y)
        for (size_t x = 0; x < row; ++x) {
          const auto value = uint8_t(17 * mip + 13 * y + x);
          padded[mip][y * (row + 12) + x] = value;
          expected[mip][y * row + x] = value;
        }
      uploads.push_back({padded[mip].data(), row + 12, mip});
    }
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    ASSERT_TRUE(commands);
    ASSERT_TRUE(commands->CmdUploadImage(image.Get(), uploads));
    for (auto &bytes : padded)
      std::fill(bytes.begin(), bytes.end(), 0);
    std::vector<ObjPtr<IDownloadToken>> downloads;
    for (uint32_t mip = 0; mip < levels; ++mip) {
      downloads.emplace_back(commands->CmdDownloadImage(image.Get(), {0, 0, 0, 0, mip}));
      ASSERT_TRUE(downloads.back());
    }
    ObjPtr<ISubmissionToken> initial = commands->Submit();
    ASSERT_TRUE(initial);
    ASSERT_TRUE(initial->WaitHost());
    for (uint32_t mip = 0; mip < levels; ++mip) {
      std::vector<uint8_t> actual(expected[mip].size());
      EXPECT_EQ(downloads[mip]->GetSize(), actual.size());
      ASSERT_TRUE(downloads[mip]->CopyTo(actual.data(), actual.size()));
      EXPECT_EQ(actual, expected[mip]) << "mip " << mip;
    }
    const uint32_t changed = levels / 2;
    ImageDesc level;
    ASSERT_TRUE(ImageMipDesc(image->GetDesc(), changed, level));
    std::fill(expected[changed].begin(), expected[changed].end(), 0x7b);
    commands = assist->CreateCommandContext();
    ASSERT_TRUE(commands);
    ASSERT_TRUE(commands->CmdUploadImage(image.Get(), expected[changed].data(), level.width * 4, changed));
    ObjPtr<ISubmissionToken> updated = commands->Submit({{initial->GetEvent()}});
    ASSERT_TRUE(updated);
    // The next context is recorded without a host wait. Submission dependencies
    // and restored per-mip defaults must make the cross-queue readback correct.
    commands = queue->CreateCommandContext();
    downloads.clear();
    for (uint32_t mip = 0; mip < levels; ++mip) {
      downloads.emplace_back(commands->CmdDownloadImage(image.Get(), {0, 0, 0, 0, mip}));
      ASSERT_TRUE(downloads.back());
    }
    ObjPtr<IDownloadToken> subrect = commands->CmdDownloadImage(image.Get(), {0, 0, 1, 1, changed});
    ASSERT_TRUE(subrect);
    ObjPtr<ISubmissionToken> done = commands->Submit({{updated->GetEvent()}});
    ASSERT_TRUE(done && done->WaitHost());
    ASSERT_TRUE(updated->WaitHost());
    for (uint32_t mip = 0; mip < levels; ++mip) {
      std::vector<uint8_t> actual(expected[mip].size());
      ASSERT_TRUE(downloads[mip]->CopyTo(actual.data(), actual.size()));
      EXPECT_EQ(actual, expected[mip]) << "mip " << mip;
    }
    std::array<uint8_t, 4> pixel{};
    ASSERT_EQ(subrect->GetSize(), pixel.size());
    ASSERT_TRUE(subrect->CopyTo(pixel.data(), pixel.size()));
    EXPECT_EQ(pixel, (std::array<uint8_t, 4>{0x7b, 0x7b, 0x7b, 0x7b}));
  }
}

TEST_P(RhiMip, InvalidBatchIsRejectedBeforeAnyUpload) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IImage> image = device->CreateImage({4, 4, IMAGE_FORMAT_R8G8B8A8_UNORM, 0});
  ASSERT_TRUE(image);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ASSERT_TRUE(queue);
  ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
  std::vector<uint8_t> before(64, 12), after(64, 33);
  ASSERT_TRUE(commands->CmdUploadImage(image.Get(), before.data(), 16));
  EXPECT_FALSE(commands->CmdUploadImage(image.Get(), {{after.data(), 16, 0}, {after.data(), 4, 3}}));
  EXPECT_FALSE(commands->CmdUploadImage(image.Get(), std::vector<ImageUploadInfo>{}));
  EXPECT_FALSE(commands->CmdUploadImage(image.Get(), {{after.data(), 7, 1}}));
  ObjPtr<IDownloadToken> download = commands->CmdDownloadImage(image.Get());
  ASSERT_TRUE(download);
  ObjPtr<ISubmissionToken> done = commands->Submit();
  ASSERT_TRUE(done && done->WaitHost());
  std::vector<uint8_t> actual(64);
  ASSERT_TRUE(download->CopyTo(actual.data(), actual.size()));
  EXPECT_EQ(actual, before);
  commands = queue->CreateCommandContext();
  EXPECT_FALSE(commands->CmdUploadImage(image.Get(), before.data(), 16, 3));
  EXPECT_FALSE(commands->Submit());
  commands = queue->CreateCommandContext();
  EXPECT_FALSE(commands->CmdDownloadImage(image.Get(), {0, 0, 0, 0, 3}));
  EXPECT_FALSE(commands->Submit());
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiMip, testing::ValuesIn(MipBackends()));
}  // namespace grassland::rhi
