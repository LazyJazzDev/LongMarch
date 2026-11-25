#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>

#include "../../demo/rhi_hello/modules/mip/scene.h"
#include "grassland/rhi/rhi.h"
#if defined(LONGMARCH_D3D12_ENABLED)
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_image.h"
#endif
#if defined(LONGMARCH_METAL_ENABLED)
#include "grassland/rhi/backend/metal/metal_device.h"
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
#if defined(LONGMARCH_METAL_ENABLED)
  result.push_back(BACKEND_API_METAL);
#endif
  return result;
}
}  // namespace

class RhiMip : public testing::TestWithParam<BackendAPI> {
 protected:
  // Keep validation enabled for every image test, including newly added cases.
  ICore *CreateCore(BackendAPI api, const CoreDesc & = {}) {
    auto core = grassland::rhi::CreateCore(api, {true});
    validation_core_ = core;
    if (core)
      core->AddRef();
    return core;
  }

  void Track(IDevice *device) {
#if defined(LONGMARCH_VULKAN_ENABLED)
    if (GetParam() == BACKEND_API_VULKAN && !messenger_) {
      instance_ = rhi::QueryInterfacePtr<backend::vulkan::IVulkanDevice>(device)->Instance();
      auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
          vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
      ASSERT_TRUE(create);
      VkDebugUtilsMessengerCreateInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
      info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      info.pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
                                const VkDebugUtilsMessengerCallbackDataEXT *data, void *) -> VkBool32 {
        ADD_FAILURE() << data->pMessage;
        return VK_FALSE;
      };
      ASSERT_EQ(create(instance_, &info, nullptr, &messenger_), VK_SUCCESS);
    }
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
    if (GetParam() == BACKEND_API_D3D12)
      ASSERT_TRUE(
          SUCCEEDED(rhi::QueryInterfacePtr<backend::d3d12::ID3D12Device>(device).Get()->NativeDevice()->QueryInterface(
              IID_PPV_ARGS(&messages_))));
#endif
  }

  void TearDown() override {
#if defined(LONGMARCH_VULKAN_ENABLED)
    if (messenger_) {
      auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
          vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
      destroy(instance_, messenger_, nullptr);
    }
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
    if (messages_)
      for (UINT64 i = 0; i < messages_->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
        SIZE_T size = 0;
        messages_->GetMessage(i, nullptr, &size);
        std::vector<uint8_t> storage(size);
        auto message = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
        messages_->GetMessage(i, message, &size);
        if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR)
          ADD_FAILURE() << message->pDescription;
      }
    messages_.Reset();
#endif
  }
#if defined(LONGMARCH_D3D12_ENABLED)
  backend::d3d12::ComPtr<ID3D12InfoQueue> messages_;
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
  VkInstance instance_{};
  VkDebugUtilsMessengerEXT messenger_{};
#endif
  ObjPtr<ICore> validation_core_;
};

TEST(RhiMipDescription, SamplerConstructionSetsIndependentFilters) {
  SamplerDesc defaults;
  EXPECT_EQ(defaults.min_filter, SAMPLER_FILTER_LINEAR);
  EXPECT_EQ(defaults.mag_filter, SAMPLER_FILTER_LINEAR);
  EXPECT_EQ(defaults.mip_filter, SAMPLER_FILTER_LINEAR);
  SamplerDesc uniform{SAMPLER_FILTER_NEAREST, SAMPLER_ADDRESS_CLAMP_TO_EDGE, SAMPLER_ADDRESS_MIRRORED_REPEAT};
  EXPECT_EQ(uniform.min_filter, SAMPLER_FILTER_NEAREST);
  EXPECT_EQ(uniform.mag_filter, SAMPLER_FILTER_NEAREST);
  EXPECT_EQ(uniform.mip_filter, SAMPLER_FILTER_NEAREST);
  EXPECT_EQ(uniform.address_w, SAMPLER_ADDRESS_MIRRORED_REPEAT);
  uniform.min_filter = SAMPLER_FILTER_LINEAR;
  EXPECT_EQ(uniform.min_filter, SAMPLER_FILTER_LINEAR);
  EXPECT_EQ(uniform.mag_filter, SAMPLER_FILTER_NEAREST);
  EXPECT_EQ(uniform.mip_filter, SAMPLER_FILTER_NEAREST);
  SamplerDesc detailed{SAMPLER_FILTER_NEAREST, SAMPLER_FILTER_LINEAR, SAMPLER_FILTER_NEAREST};
  EXPECT_EQ(detailed.min_filter, SAMPLER_FILTER_NEAREST);
  EXPECT_EQ(detailed.mag_filter, SAMPLER_FILTER_LINEAR);
  EXPECT_EQ(detailed.mip_filter, SAMPLER_FILTER_NEAREST);
}

TEST_P(RhiMip, EnumeratedCapabilitiesAgreeWithNativeDevice) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  const auto devices = core->GetPhysicalDevices();
  ASSERT_FALSE(devices.empty());
  for (size_t i = 0; i < devices.size(); ++i) {
    ObjPtr<IDevice> device = core->CreateDevice(static_cast<int32_t>(i));
    ASSERT_TRUE(device);
    Track(device.Get());
    const auto &mip = devices[i].mip_feature;
    EXPECT_GT(mip.max_image_dimension_2d, 0u);
    EXPECT_GE(mip.max_anisotropy, 1u);
    EXPECT_GT(devices[i].format_features[IMAGE_FORMAT_R8G8B8A8_UNORM].max_mip_levels, 1u);
    EXPECT_TRUE(devices[i].format_features[IMAGE_FORMAT_R8G8B8A8_UNORM].linear_filter);
    EXPECT_FALSE(devices[i].format_features[IMAGE_FORMAT_R32_UINT].linear_filter);
    EXPECT_FALSE(devices[i].format_features[IMAGE_FORMAT_UNKNOWN].sampled);
#if defined(LONGMARCH_D3D12_ENABLED)
    if (GetParam() == BACKEND_API_D3D12) {
      auto native = QueryInterfacePtr<backend::d3d12::ID3D12Device>(device.Get());
      ASSERT_TRUE(native);
      EXPECT_EQ(mip.max_anisotropy, D3D12_MAX_MAXANISOTROPY);
      for (int index = IMAGE_FORMAT_UNKNOWN + 1; index < IMAGE_FORMAT_COUNT; ++index) {
        auto format = backend::d3d12::NativeImageFormat(static_cast<ImageFormat>(index));
        if (format == DXGI_FORMAT_UNKNOWN) {
          EXPECT_EQ(devices[i].format_features[index].max_mip_levels, 0u);
          continue;
        }
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format};
        if (FAILED(
                native->NativeDevice()->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))))
          continue;
        if (support.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D) {
          EXPECT_EQ(devices[i].format_features[index].max_mip_levels > 1,
                    (support.Support1 & D3D12_FORMAT_SUPPORT1_MIP) != 0);
          EXPECT_EQ(devices[i].format_features[index].storage,
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
        if (format == VK_FORMAT_UNDEFINED || !devices[i].format_features[index].max_mip_levels)
          continue;
        VkFormatProperties format_properties{};
        vkGetPhysicalDeviceFormatProperties(native->NativePhysicalDevice(), format, &format_properties);
        EXPECT_EQ(devices[i].format_features[index].storage,
                  (format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0);
        EXPECT_EQ(devices[i].format_features[index].linear_filter,
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
  Track(device.Get());
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
  desc.border_color = static_cast<SamplerBorderColor>(-1);
  EXPECT_FALSE(device->CreateSampler(desc));
  desc.border_color = SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK;
  desc.mag_filter = SAMPLER_FILTER_NEAREST;
  desc.mip_filter = SAMPLER_FILTER_LINEAR;
  sampler = device->CreateSampler(desc);
  ASSERT_TRUE(sampler);
  if (features.max_anisotropy > 1) {
    desc.max_anisotropy = 2;
    EXPECT_FALSE(device->CreateSampler(desc));
  }
}

TEST_P(RhiMip, BorderColorsAreSampledAndFilteredOutsideTexture) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
#if defined(LONGMARCH_METAL_ENABLED)
  if (GetParam() == BACKEND_API_METAL) {
    auto native = QueryInterfacePtr<backend::metal::IMetalDevice>(device.Get())->NativeDevice();
    if (!native->supportsFamily(MTL::GPUFamilyApple7) && !native->supportsFamily(MTL::GPUFamilyMac2)) {
      EXPECT_FALSE(device->CreateSampler({SAMPLER_FILTER_LINEAR, SAMPLER_ADDRESS_CLAMP_TO_BORDER}));
      return;
    }
  }
#endif
  ObjPtr<IImage> image = device->CreateImage({1, 1, IMAGE_FORMAT_R8G8B8A8_UNORM});
  ObjPtr<IBuffer> output = device->CreateBuffer({6 * 4 * sizeof(float)});
  ASSERT_TRUE(image && output);
  ObjPtr<IShader> shader = device->CreateShaderFromSource(R"(
    RHI_TEXTURE2D(0,0,float4,image);
    RHI_SAMPLER(0,1,sampling);
    RHI_RW_STRUCTURED_BUFFER(0,2,float4,output);
    [numthreads(1,1,1)] void Main(uint3 id:SV_DispatchThreadID) {
      float2 uv[6] = {float2(.5,.5), float2(-1,.5), float2(2,.5),
                      float2(.5,-1), float2(.5,2), float2(0,.5)};
      output[id.x] = image.SampleLevel(sampling, uv[id.x], 0);
    }
  )",
                                                          "Main", SHADER_STAGE_COMPUTE);
  ASSERT_TRUE(shader);
  BindingLayoutDesc layout{{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_COMPUTE},
                            {1, BINDING_SAMPLER, 1, SHADER_VISIBILITY_COMPUTE},
                            {2, BINDING_RW_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_COMPUTE, 16}}};
  ObjPtr<IComputeProgram> program = device->CreateComputeProgram({shader.Get(), {{layout}}});
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(layout);
  ASSERT_TRUE(program && group);
  ASSERT_TRUE(group->BindResource(0, image.Get()));
  ASSERT_TRUE(group->BindResource(2, output.Get()));
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_COMPUTE);
  ASSERT_TRUE(queue);
  const uint8_t pixel[]{64, 128, 192, 255};
  const float colors[][4] = {{0, 0, 0, 0}, {0, 0, 0, 1}, {1, 1, 1, 1}};
  for (auto mode : {SAMPLER_ADDRESS_CLAMP_TO_BORDER, SAMPLER_ADDRESS_CLAMP_TO_EDGE}) {
    for (auto filter : {SAMPLER_FILTER_NEAREST, SAMPLER_FILTER_LINEAR}) {
      for (auto color : {SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK, SAMPLER_BORDER_COLOR_OPAQUE_BLACK,
                         SAMPLER_BORDER_COLOR_OPAQUE_WHITE}) {
        SCOPED_TRACE(testing::Message() << "mode " << mode << ", filter " << filter << ", color " << color);
        SamplerDesc desc{filter, mode, mode};
        desc.border_color = color;
        ObjPtr<ISampler> sampler = device->CreateSampler(desc);
        ASSERT_TRUE(sampler);
        ASSERT_TRUE(group->BindResource(1, sampler.Get()));
        ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
        ASSERT_TRUE(commands);
        ASSERT_TRUE(commands->CmdUploadImage(image.Get(), pixel, sizeof(pixel)));
        ASSERT_TRUE(commands->CmdBindComputeProgram(program.Get()));
        ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
        ASSERT_TRUE(commands->CmdDispatch(6));
        ObjPtr<IDownloadToken> download = commands->CmdDownloadBuffer(output.Get(), 0, 6 * 4 * sizeof(float));
        ASSERT_TRUE(download);
        ObjPtr<ISubmissionToken> done = commands->Submit();
        ASSERT_TRUE(done && done->WaitHost());
        float actual[6][4]{};
        ASSERT_TRUE(download->CopyTo(actual, sizeof(actual)));
        for (int sample = 0; sample < 6; ++sample) {
          for (int channel = 0; channel < 4; ++channel) {
            float expected = pixel[channel] / 255.0f;
            if (mode == SAMPLER_ADDRESS_CLAMP_TO_BORDER) {
              if (sample >= 1 && sample <= 4)
                expected = colors[color][channel];
              else if (sample == 5 && filter == SAMPLER_FILTER_LINEAR)
                expected = (expected + colors[color][channel]) * 0.5f;
            }
            EXPECT_NEAR(actual[sample][channel], expected, 0.002f) << "sample " << sample << ", channel " << channel;
          }
        }
      }
    }
  }
}

TEST_P(RhiMip, CreationResolvesChainsAndRejectsInvalidLevels) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
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

TEST_P(RhiMip, PaddedMipUploadsSnapshotEveryLevelAndSingleUpdatesAreIsolated) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ObjPtr<IQueue> assist = device->GetQueue(QUEUE_TYPE_ASSIST);
  ASSERT_TRUE(queue && assist);
  for (const auto dimensions : {std::array<uint32_t, 2>{19, 7}, {1, 9}, {8, 1}, {1, 1}}) {
    ObjPtr<IImage> image = device->CreateImage({dimensions[0], dimensions[1], IMAGE_FORMAT_R8G8B8A8_UNORM, 0});
    ASSERT_TRUE(image);
    const auto levels = image->GetDesc().mip_levels;
    std::vector<std::vector<uint8_t>> padded(levels), expected(levels);
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
    }
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    ASSERT_TRUE(commands);
    // Batch policy belongs to the caller; the RHI only records single-level uploads.
    const auto upload_levels = [&] {
      for (uint32_t mip = 0; mip < levels; ++mip) {
        ImageDesc level;
        if (!ImageMipDesc(image->GetDesc(), mip, level) ||
            !commands->CmdUploadImage(image.Get(), padded[mip].data(), level.width * 4 + 12, mip))
          return false;
      }
      return true;
    };
    ASSERT_TRUE(upload_levels());
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

TEST_P(RhiMip, InvalidMipTransfersInvalidateRecording) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  ObjPtr<IImage> image = device->CreateImage({4, 4, IMAGE_FORMAT_R8G8B8A8_UNORM, 0});
  ASSERT_TRUE(image);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ASSERT_TRUE(queue);
  ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
  std::vector<uint8_t> before(64, 12);
  EXPECT_FALSE(commands->CmdUploadImage(image.Get(), before.data(), 16, 3));
  EXPECT_FALSE(commands->Submit());
  commands = queue->CreateCommandContext();
  EXPECT_FALSE(commands->CmdDownloadImage(image.Get(), {0, 0, 0, 0, 3}));
  EXPECT_FALSE(commands->Submit());
  commands = queue->CreateCommandContext();
  EXPECT_FALSE(commands->CmdUploadImage(image.Get(), before.data(), 7, 1));
  EXPECT_FALSE(commands->Submit());
}

TEST_P(RhiMip, MipViewsCaptureRangesAndSynchronizeDisjointReadWrites) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  ObjPtr<IImage> image = device->CreateImage({8, 8, IMAGE_FORMAT_R8G8B8A8_UNORM, 0});
  ASSERT_TRUE(image);
  const std::string source = R"(
    RHI_TEXTURE2D(0,0,float4,sourceImage);
    RHI_RW_TEXTURE2D(0,1,float4,targetImage);
    [numthreads(8,8,1)] void Main(uint3 id:SV_DispatchThreadID) {
      uint width,height; targetImage.GetDimensions(width,height);
      if (id.x < width && id.y < height)
        targetImage[id.xy] = sourceImage.Load(int3(id.xy*2,0)) + 32.0/255.0;
    }
  )";
  ObjPtr<IShader> shader = device->CreateShaderFromSource(source, "Main", SHADER_STAGE_COMPUTE);
  ASSERT_TRUE(shader);
  BindingLayoutDesc layout{
      {{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_COMPUTE}, {1, BINDING_RW_TEXTURE2D, 1, SHADER_VISIBILITY_COMPUTE}}};
  ObjPtr<IComputeProgram> program = device->CreateComputeProgram({shader.Get(), {{layout}}});
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(layout);
  ASSERT_TRUE(program && group);
  EXPECT_FALSE(group->BindResource(0, ImageView{image.Get(), 4, 1}));
  EXPECT_FALSE(group->BindResource(0, ImageView{image.Get(), 2, 3}));
  EXPECT_FALSE(group->BindResource(1, ImageView{image.Get(), 0, 2}));
  EXPECT_FALSE(group->BindResource(0, ImageView{nullptr}));
  for (auto role : {QUEUE_TYPE_MAIN, QUEUE_TYPE_COMPUTE}) {
    ObjPtr<IQueue> queue = device->GetQueue(role);
    ASSERT_TRUE(queue);
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    std::vector<uint8_t> pixels(8 * 8 * 4, 32);
    ASSERT_TRUE(commands->CmdUploadImage(image.Get(), pixels.data(), 32));
    ASSERT_TRUE(commands->CmdBindComputeProgram(program.Get()));
    for (uint32_t mip = 1; mip < 4; ++mip) {
      ASSERT_TRUE(group->BindResource(0, ImageView{image.Get(), mip - 1, 1}));
      ASSERT_TRUE(group->BindResource(1, ImageView{image.Get(), mip, 0}));
      ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
      // Changing both ranges after capture must not change the pending dispatch.
      ASSERT_TRUE(group->BindResource(0, ImageView{image.Get(), mip, 1}));
      ASSERT_TRUE(group->BindResource(1, ImageView{image.Get(), mip - 1, 1}));
      ASSERT_TRUE(commands->CmdDispatch(1, 1, 1));
    }
    std::vector<ObjPtr<IDownloadToken>> downloads;
    for (uint32_t mip = 0; mip < 4; ++mip)
      downloads.emplace_back(commands->CmdDownloadImage(image.Get(), {0, 0, 0, 0, mip}));
    ObjPtr<ISubmissionToken> done = commands->Submit();
    ASSERT_TRUE(done && done->WaitHost());
    for (uint32_t mip = 0; mip < 4; ++mip) {
      ASSERT_TRUE(downloads[mip]);
      std::vector<uint8_t> actual(downloads[mip]->GetSize());
      ASSERT_TRUE(downloads[mip]->CopyTo(actual.data(), actual.size()));
      for (auto value : actual)
        EXPECT_NEAR(int(value), 32 * (mip + 1), 1) << "mip " << mip;
    }
    commands = queue->CreateCommandContext();
    ASSERT_TRUE(commands->CmdBindComputeProgram(program.Get()));
    ASSERT_TRUE(group->BindResource(0, image.Get()));  // Full chain overlaps mip 1.
    ASSERT_TRUE(group->BindResource(1, ImageView{image.Get(), 1, 1}));
    ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
    EXPECT_FALSE(commands->CmdDispatch(1, 1, 1));
    EXPECT_FALSE(commands->Submit());
  }
}

TEST_P(RhiMip, GpuGenerationMatchesLinearReferenceAndSupportsPartialChains) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_COMPUTE);
  ASSERT_TRUE(queue);
  for (auto dimensions : {std::array<uint32_t, 2>{8, 8}, {19, 7}, {1, 9}, {8, 1}, {1, 1}}) {
    ObjPtr<IImage> image = device->CreateImage({dimensions[0], dimensions[1], IMAGE_FORMAT_R32G32B32A32_SFLOAT, 0});
    ASSERT_TRUE(image);
    std::vector<std::vector<float>> expected(image->GetDesc().mip_levels);
    expected[0].resize(dimensions[0] * dimensions[1] * 4);
    for (size_t i = 0; i < expected[0].size(); ++i)
      expected[0][i] = float((i * 37) % 113) / 31.0f;
    for (uint32_t mip = 1; mip < expected.size(); ++mip) {
      ImageDesc src, dst;
      ASSERT_TRUE(ImageMipDesc(image->GetDesc(), mip - 1, src));
      ASSERT_TRUE(ImageMipDesc(image->GetDesc(), mip, dst));
      expected[mip].resize(dst.width * dst.height * 4);
      for (uint32_t y = 0; y < dst.height; ++y)
        for (uint32_t x = 0; x < dst.width; ++x) {
          const float sx = (x + 0.5f) * src.width / dst.width - 0.5f;
          const float sy = (y + 0.5f) * src.height / dst.height - 0.5f;
          const int ix = int(std::floor(sx)), iy = int(std::floor(sy));
          const float fx = sx - ix, fy = sy - iy;
          for (uint32_t c = 0; c < 4; ++c) {
            float value = 0;
            for (int j = 0; j < 2; ++j)
              for (int i = 0; i < 2; ++i) {
                const auto px = std::clamp(ix + i, 0, int(src.width) - 1);
                const auto py = std::clamp(iy + j, 0, int(src.height) - 1);
                value += expected[mip - 1][(py * src.width + px) * 4 + c] * (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
              }
            expected[mip][(y * dst.width + x) * 4 + c] = value;
          }
        }
    }
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    ASSERT_TRUE(commands->CmdUploadImage(image.Get(), expected[0].data(), dimensions[0] * 16));
    ASSERT_TRUE(commands->CmdGenerateMipmaps(image.Get()));
    std::vector<ObjPtr<IDownloadToken>> downloads;
    for (uint32_t mip = 0; mip < expected.size(); ++mip)
      downloads.emplace_back(commands->CmdDownloadImage(image.Get(), {0, 0, 0, 0, mip}));
    ObjPtr<ISubmissionToken> done = commands->Submit();
    ASSERT_TRUE(done && done->WaitHost());
    for (uint32_t mip = 0; mip < expected.size(); ++mip) {
      ASSERT_TRUE(downloads[mip]);
      std::vector<float> actual(expected[mip].size());
      ASSERT_TRUE(downloads[mip]->CopyTo(actual.data(), actual.size() * sizeof(float)));
      for (size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[mip][i], 0.025f) << "mip " << mip;
    }
    if (expected.size() < 3)
      continue;
    commands = queue->CreateCommandContext();
    ImageDesc level;
    ASSERT_TRUE(ImageMipDesc(image->GetDesc(), 1, level));
    std::vector<float> replacement(level.width * level.height * 4, 2.25f);
    ASSERT_TRUE(commands->CmdUploadImage(image.Get(), replacement.data(), level.width * 16, 1));
    ASSERT_TRUE(commands->CmdGenerateMipmaps(image.Get(), 1, 2));
    downloads.clear();
    for (uint32_t mip = 0; mip < expected.size(); ++mip)
      downloads.emplace_back(commands->CmdDownloadImage(image.Get(), {0, 0, 0, 0, mip}));
    done = commands->Submit();
    ASSERT_TRUE(done && done->WaitHost());
    for (uint32_t mip = 0; mip < expected.size(); ++mip) {
      std::vector<float> actual(expected[mip].size());
      ASSERT_TRUE(downloads[mip]->CopyTo(actual.data(), actual.size() * sizeof(float)));
      for (size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], mip == 1 || mip == 2 ? 2.25f : expected[mip][i], 0.025f);
    }
    commands = queue->CreateCommandContext();
    EXPECT_FALSE(commands->CmdGenerateMipmaps(image.Get(), image->GetDesc().mip_levels));
    EXPECT_FALSE(commands->Submit());
  }
  ObjPtr<IImage> integer = device->CreateImage({4, 4, IMAGE_FORMAT_R32_UINT, 0});
  ASSERT_TRUE(integer);
  ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
  EXPECT_FALSE(commands->CmdGenerateMipmaps(integer.Get()));
  EXPECT_FALSE(commands->Submit());
}

TEST_P(RhiMip, DemoDrawsNineAnimatedCubesWithDistinctFiltering) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  rhi_mip_demo::Scene scene;
  ASSERT_TRUE(scene.Initialize(device.Get(), IMAGE_FORMAT_R8G8B8A8_UNORM));
  constexpr uint32_t width = 1296, height = 936, cell_width = 432, cell_height = 288;
  ObjPtr<IImage> target = device->CreateImage({width, height, IMAGE_FORMAT_R8G8B8A8_UNORM});
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ASSERT_TRUE(target && queue);
  std::vector<uint8_t> previous;
  for (int frame = 0; frame < 2; ++frame) {
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    ASSERT_TRUE(commands);
    ASSERT_TRUE(scene.Draw(commands.Get(), target.Get(), float(frame)));
    ObjPtr<IDownloadToken> download = commands->CmdDownloadImage(target.Get());
    ASSERT_TRUE(download);
    ObjPtr<ISubmissionToken> done = commands->Submit();
    ASSERT_TRUE(done && done->WaitHost());
    std::vector<uint8_t> pixels(width * height * 4);
    ASSERT_TRUE(download->CopyTo(pixels.data(), pixels.size()));
    uint32_t occupied[3][3]{};
    uint64_t differences[2]{};
    for (uint32_t row = 0; row < 3; ++row)
      for (uint32_t y = 0; y < cell_height; ++y)
        for (uint32_t x = 0; x < cell_width; ++x) {
          for (uint32_t col = 0; col < 3; ++col) {
            const auto offset = ((72 + row * cell_height + y) * width + col * cell_width + x) * 4;
            if (pixels[offset] > 25 || pixels[offset + 1] > 25 || pixels[offset + 2] > 30)
              ++occupied[row][col];
            if (col < 2)
              for (uint32_t c = 0; c < 3; ++c)
                differences[col] += std::abs(int(pixels[offset + c]) - int(pixels[offset + cell_width * 4 + c]));
          }
        }
    for (uint32_t col = 0; col < 3; ++col) {
      EXPECT_GT(occupied[0][col], 500u);
      EXPECT_GT(occupied[1][col], occupied[0][col] * 2);
      EXPECT_GT(occupied[2][col], occupied[1][col] * 2);
    }
    EXPECT_GT(differences[0], 100000u);  // No mip versus trilinear.
    EXPECT_GT(differences[1], 10000u);   // Trilinear versus anisotropic on oblique faces.
    if (!previous.empty())
      EXPECT_NE(previous, pixels);
    previous = pixels;
    // Optional test artifact: raw rendered scene, never presented as a window screenshot.
    if (frame == 0)
      if (const char *directory = std::getenv("LONGMARCH_RHI_CAPTURE_DIR")) {
        const auto filename = std::string(directory) + "/mip-" + BackendAPIName(GetParam()) + ".ppm";
        std::ofstream output(filename, std::ios::binary);
        ASSERT_TRUE(output);
        output << "P6\n" << width << " " << height << "\n255\n";
        for (size_t i = 0; i < pixels.size(); i += 4)
          output.write(reinterpret_cast<const char *>(pixels.data() + i), 3);
        ASSERT_TRUE(output);
      }
  }
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiMip, testing::ValuesIn(MipBackends()));
}  // namespace grassland::rhi
#if defined(LONGMARCH_VULKAN_ENABLED)

namespace grassland::rhi {
TEST(RhiVulkanImageView, ReusesDefaultAndMipViewsAcrossBindingGroups) {
  ObjPtr<ICore> core = CreateCore(BACKEND_API_VULKAN, {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  for (uint32_t levels : {1u, 4u}) {
    ObjPtr<IImage> image = device->CreateImage({8, 8, IMAGE_FORMAT_R8G8B8A8_UNORM, levels});
    ASSERT_TRUE(image);
    auto native = QueryInterfacePtr<backend::vulkan::IVulkanImage>(image.Get());
    ASSERT_TRUE(native);
    auto full = native->BindingView();
    ASSERT_NE(full, VK_NULL_HANDLE);
    EXPECT_EQ(full, native->BindingView(0, levels));
    EXPECT_EQ(native->AttachmentView(), native->BindingView(0, 1));
    EXPECT_EQ(native->BindingView(levels, 1), VK_NULL_HANDLE);
    EXPECT_EQ(native->BindingView(0, levels + 1), VK_NULL_HANDLE);
    BindingLayoutDesc layout{{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_COMPUTE}}};
    auto selected = native->BindingView(levels - 1, 1);
    ASSERT_NE(selected, VK_NULL_HANDLE);
    for (int iteration = 0; iteration < 3; ++iteration) {
      ObjPtr<IBindingGroup> group = device->CreateBindingGroup(layout);
      ASSERT_TRUE(group);
      ASSERT_TRUE(group->BindResource(0, ImageView{image.Get(), levels - 1, 1}));
      EXPECT_EQ(selected, native->BindingView(levels - 1, 1));
      ASSERT_TRUE(group->BindResource(0, image.Get()));
      EXPECT_EQ(full, native->BindingView());
    }
    // Releasing or rewriting groups must not discard the image's cached views.
    EXPECT_EQ(selected, native->BindingView(levels - 1, 0));
    if (levels > 1) {
      EXPECT_NE(full, selected);
      EXPECT_NE(native->BindingView(1, 1), native->BindingView(1, 2));
    }
  }
}
}  // namespace grassland::rhi
#endif
