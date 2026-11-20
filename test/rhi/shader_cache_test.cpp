#include "grassland/rhi/shader_cache.h"

#include <gtest/gtest.h>

#include <chrono>
#include <fstream>
#include <thread>

#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/sha256.h"
#include "grassland/util/virtual_file_system.h"

namespace grassland::rhi {
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

TEST_F(ShaderCacheTest, DigestAndRequestBoundaries) {
  detail::SHA256 digest;
  digest.Update("abc");
  EXPECT_EQ(digest.Finish(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
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
}  // namespace grassland::rhi
