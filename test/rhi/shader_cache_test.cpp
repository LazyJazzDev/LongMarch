#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <thread>

#include "grassland/rhi/backend/backend_mip_generator.h"
#include "grassland/rhi/backend/backend_shader_cache.h"
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/rhi.h"
#include "grassland/util/virtual_file_system.h"

namespace grassland::rhi::backend {
class ShaderCacheTest : public testing::Test {
 protected:
  void SetUp() override {
    previous_ = GetShaderCacheSettings();
    directory_ = std::filesystem::temp_directory_path() /
                 ("longmarch-cache-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ASSERT_TRUE(ConfigureShaderCache({directory_, false, true}));
  }

  void TearDown() override {
    ConfigureShaderCache(previous_);
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }

  ShaderCacheSettings previous_;
  std::filesystem::path directory_;
};

TEST(ShaderCacheKeyTest, RequestBoundaries) {
  EXPECT_NE(ShaderCacheKey({"ab", "c"}), ShaderCacheKey({"a", "bc"}));
}

TEST_F(ShaderCacheTest, ReplayValidatesPayloadAndReadOnlyDoesNotOverwrite) {
  const std::vector<uint8_t> expected{1, 2, 0, 4};
  ASSERT_TRUE(WriteShaderCache("payload", expected));
  ASSERT_TRUE(ConfigureShaderCache({directory_, true, true}));
  ASSERT_TRUE(WriteShaderCache("payload", {9}));
  std::vector<uint8_t> actual;
  ASSERT_TRUE(ReadShaderCache("payload", actual));
  EXPECT_EQ(actual, expected);
  {
    std::ofstream file(directory_ / "payload", std::ios::binary | std::ios::app);
    file.put('x');
  }
  EXPECT_FALSE(ReadShaderCache("payload", actual));
  EXPECT_TRUE(actual.empty());
  EXPECT_FALSE(ReadShaderCache("missing", actual));
}

TEST_F(ShaderCacheTest, InvalidPathAndConfigurationAreRejected) {
  std::vector<uint8_t> actual;
  EXPECT_FALSE(WriteShaderCache("../outside", {1}));
  EXPECT_FALSE(ReadShaderCache("../outside", actual));
  EXPECT_FALSE(ConfigureShaderCache({directory_ / "missing", true, false}));
  EXPECT_EQ(GetShaderCacheSettings().directory, directory_);
  {
    std::ofstream file(directory_ / "file");
    file << 'x';
  }
  EXPECT_FALSE(ConfigureShaderCache({directory_ / "file", false, false}));
}

TEST_F(ShaderCacheTest, ThreadGroupAndEmptyPayloadRoundTrip) {
  ASSERT_TRUE(WriteShaderCache("empty", {}));
  std::vector<uint8_t> actual{1};
  ASSERT_TRUE(ReadShaderCache("empty", actual));
  EXPECT_TRUE(actual.empty());
  const std::array<uint32_t, 3> expected{256, 2, 1};
  ASSERT_TRUE(WriteShaderThreadGroup("kernel", expected));
  std::array<uint32_t, 3> group{};
  ASSERT_TRUE(ReadShaderThreadGroup("kernel", group));
  EXPECT_EQ(group, expected);
  ASSERT_TRUE(WriteShaderCache("threads-kernel", {0}));
  EXPECT_FALSE(ReadShaderThreadGroup("kernel", group));
}

TEST_F(ShaderCacheTest, ConfigurationIsThreadLocal) {
  bool independent = false;
  std::thread thread([&] { independent = GetShaderCacheSettings().directory.empty(); });
  thread.join();
  EXPECT_TRUE(independent);
  EXPECT_EQ(GetShaderCacheSettings().directory, directory_);
}

TEST_F(ShaderCacheTest, PreprocessingReplaysTheTargetPlatformAndRejectsChangedRequests) {
  VirtualFileSystem files;
  ASSERT_EQ(files.WriteFile("test.slang", R"(
#if !RHI_PLATFORM_IOS || RHI_PLATFORM_MACOS || RHI_PLATFORM_WINDOWS || RHI_PLATFORM_LINUX
#error Expected the iOS target even during host preparation
#endif
float4 VSMain() : SV_Position { return float4(0, 0, 0, 1); }
)"),
            0);
  ShaderDesc desc;
  desc.filename = "test.slang";
  desc.entry_point = "VSMain";
  desc.stage = SHADER_STAGE_VERTEX;
  desc.file_system = &files;
  std::string prepared, replay;
  ASSERT_TRUE(backend::PreprocessShader(desc, BACKEND_API_METAL, prepared));
  ASSERT_TRUE(ConfigureShaderCache({directory_, true, true}));
  ASSERT_TRUE(backend::PreprocessShader(desc, BACKEND_API_METAL, replay));
  EXPECT_EQ(prepared, replay);
  EXPECT_FALSE(backend::PreprocessShader(desc, BACKEND_API_VULKAN, replay));
  ASSERT_EQ(files.WriteFile("test.slang", "float4 VSMain():SV_Position { return 1; }"), 0);
  EXPECT_FALSE(backend::PreprocessShader(desc, BACKEND_API_METAL, replay));
}

TEST(ShaderCacheKeyTest, LayoutIdentityIncludesEveryFieldAndGroupPosition) {
  PipelineLayoutDesc layout{{{{{3, BINDING_RW_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_COMPUTE, 4},
                               {7, BINDING_TEXTURE2D, 2, SHADER_VISIBILITY_FRAGMENT, 0}}},
                             {}}};
  const auto key = ShaderBindingLayoutKey(layout);
  auto reordered = layout;
  std::reverse(reordered.groups[0].bindings.begin(), reordered.groups[0].bindings.end());
  EXPECT_EQ(key, ShaderBindingLayoutKey(reordered));
  for (int field = 0; field < 5; ++field) {
    auto changed = layout;
    auto &binding = changed.groups[0].bindings[0];
    switch (field) {
      case 0:
        ++binding.binding;
        break;
      case 1:
        binding.type = BINDING_STRUCTURED_BUFFER;
        break;
      case 2:
        ++binding.count;
        break;
      case 3:
        binding.visibility = SHADER_VISIBILITY_ALL;
        break;
      case 4:
        binding.stride = 8;
        break;
    }
    EXPECT_NE(key, ShaderBindingLayoutKey(changed)) << "field " << field;
  }
  auto relocated = layout;
  std::swap(relocated.groups[0], relocated.groups[1]);
  EXPECT_NE(key, ShaderBindingLayoutKey(relocated));
  layout.groups.pop_back();
  EXPECT_NE(key, ShaderBindingLayoutKey(layout));
}

TEST_F(ShaderCacheTest, ProgramReplayRejectsChangedStrideAndVisibility) {
  const std::vector<BackendAPI> apis{
#ifdef LONGMARCH_METAL_ENABLED
      BACKEND_API_METAL,
#endif
#ifdef LONGMARCH_VULKAN_ENABLED
      BACKEND_API_VULKAN,
#endif
#ifdef LONGMARCH_D3D12_ENABLED
      BACKEND_API_D3D12,
#endif
  };
  for (const auto api : apis) {
    SCOPED_TRACE(std::to_string(api));
    const auto directory = directory_ / std::to_string(api);
    ASSERT_TRUE(ConfigureShaderCache({directory, false, false}));
    ObjPtr<ICore> core = CreateCore(api);
    ASSERT_TRUE(core);
    ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
    ASSERT_TRUE(device);
    ObjPtr<IShader> shader = device->CreateShaderFromSource(R"(
RHI_RW_STRUCTURED_BUFFER(0, 3, uint, output);
[numthreads(1, 1, 1)] void CSMain(uint3 id : SV_DispatchThreadID) { output[id.x] = 42; }
)",
                                                            "CSMain", SHADER_STAGE_COMPUTE);
    ASSERT_TRUE(shader);
    ComputeProgramDesc desc;
    desc.compute_shader = shader.Get();
    desc.pipeline_layout.groups = {{{{3, BINDING_RW_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_COMPUTE, 4},
                                     {7, BINDING_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE, 0}}}};
    ObjPtr<IComputeProgram> prepared = device->CreateComputeProgram(desc);
    ASSERT_TRUE(prepared);
    for (bool read_only : {false, true}) {
      SCOPED_TRACE(read_only ? "replay" : "online");
      ASSERT_TRUE(ConfigureShaderCache({directory, read_only, false}));
      auto wrong_stride = desc;
      wrong_stride.pipeline_layout.groups[0].bindings[0].stride = 8;
      ObjPtr<IComputeProgram> stride_program = device->CreateComputeProgram(wrong_stride);
      EXPECT_FALSE(stride_program);
      auto wrong_visibility = desc;
      wrong_visibility.pipeline_layout.groups[0].bindings[0].visibility = SHADER_VISIBILITY_FRAGMENT;
      ObjPtr<IComputeProgram> visibility_program = device->CreateComputeProgram(wrong_visibility);
      EXPECT_FALSE(visibility_program);
    }
    // Equivalent declaration order must still find the prepared bundle.
    std::reverse(desc.pipeline_layout.groups[0].bindings.begin(), desc.pipeline_layout.groups[0].bindings.end());
    ObjPtr<IComputeProgram> replay = device->CreateComputeProgram(desc);
    EXPECT_TRUE(replay);
  }
}
}  // namespace grassland::rhi::backend

#if defined(LONGMARCH_METAL_ENABLED)
namespace grassland::rhi::backend {
TEST_F(ShaderCacheTest, OfflineMipProgramReplaysAndGeneratesOnMetal) {
  ObjPtr<ICore> core = CreateCore(BACKEND_API_METAL);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  // Prepare only the internal pipeline, as an offline bundle tool would.
  ASSERT_TRUE(CreateMipGenerationProgram(device.Get()));
  ASSERT_TRUE(ConfigureShaderCache({directory_, true, true}));
  ObjPtr<IImage> image = device->CreateImage({8, 8, IMAGE_FORMAT_R8G8B8A8_UNORM, 4});
  ASSERT_TRUE(image);
  auto commands = ObjPtr<ICommandContext>(device->GetQueue(QUEUE_TYPE_MAIN)->CreateCommandContext());
  ASSERT_TRUE(commands);
  std::vector<uint8_t> input(8 * 8 * 4, 96);
  ASSERT_TRUE(commands->CmdUploadImage(image.Get(), input.data(), 32));
  ASSERT_TRUE(commands->CmdGenerateMipmaps(image.Get()));
  ObjPtr<IDownloadToken> download = commands->CmdDownloadImage(image.Get(), {0, 0, 0, 0, 3});
  ASSERT_TRUE(download);
  ObjPtr<ISubmissionToken> done = commands->Submit();
  ASSERT_TRUE(done);
  ASSERT_TRUE(done->WaitHost());
  std::array<uint8_t, 4> result{};
  ASSERT_TRUE(download->CopyTo(result.data(), result.size()));
  for (auto channel : result)
    EXPECT_EQ(channel, 96);
}
}  // namespace grassland::rhi::backend
#endif
