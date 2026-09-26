#include <gtest/gtest.h>

#include <fstream>
#include <regex>

#include "grassland/graphics/graphics.h"
#include "grassland/graphics/slang_version.h"

using namespace grassland;

TEST(SlangCompiler, StandaloneShaderCorpus) {
  const std::filesystem::path root = LONGMARCH_SOURCE_ROOT;
  const std::regex entry_pattern(
      R"(\b(VSMain|PSMain|GSMain|HSMain|DSMain|CSMain|RayGenMain|MissMain|ClosestHitMain|SphereClosestHitMain|SphereIntersectionMain|CallableMain)\s*\()");
  size_t compiled = 0;
  // Renderer-generated materials/RT entries are covered by sparkium_fallback_test.
  for (const auto &folder :
       {"demo", "python/shaders", "code/snowberg", "code/grassland/graphics/shaders", "assets/shaders/raytracing"}) {
    auto vfs = VirtualFileSystem::LoadDirectory(root / folder);
    for (const auto &file : std::filesystem::recursive_directory_iterator(root / folder)) {
      if (file.path().extension() != ".slang")
        continue;
      std::ifstream stream(file.path());
      std::string source((std::istreambuf_iterator<char>(stream)), {});
      std::set<std::string> entries;
      for (auto i = std::sregex_iterator(source.begin(), source.end(), entry_pattern); i != std::sregex_iterator(); ++i)
        entries.insert((*i)[1].str());
      if (std::string(folder) == "assets/shaders/raytracing" && source.find("void Main(") != std::string::npos)
        entries.insert("Main");
      auto relative = std::filesystem::relative(file.path(), root / folder).generic_string();
      for (const auto &entry : entries) {
        std::string profile = "lib_6_5";
        for (auto prefix : {"VS", "PS", "GS", "HS", "DS", "CS"}) {
          if (entry.substr(0, 2) == prefix) {
            profile = std::string(prefix) + "_6_5";
            std::transform(profile.begin(), profile.end(), profile.begin(), ::tolower);
          }
        }
        SCOPED_TRACE(relative + " / " + entry);
        auto code = graphics::CompileShader(
            vfs, relative, entry, profile,
            {"-target", "spirv", "-profile", "spirv_1_5", "-fvk-use-dx-layout", "-warnings-as-errors", "all"});
        EXPECT_FALSE(code.data.empty());
        ++compiled;
      }
    }
  }
  EXPECT_GT(compiled, 30u);
}

TEST(SlangCompiler, InvalidSourceReturnsNoBytecode) {
  EXPECT_TRUE(graphics::CompileShader("this is invalid source", "Main", "cs_6_0", {"-target", "spirv"}).data.empty());
}

TEST(SlangCompiler, NativeBackendsCompileAndRejectInvalidSource) {
  for (auto api : {graphics::BACKEND_API_D3D12, graphics::BACKEND_API_VULKAN}) {
    if (!graphics::SupportBackendAPI(api))
      continue;
    SCOPED_TRACE(graphics::BackendAPIString(api));
    std::unique_ptr<graphics::Core> core;
    ASSERT_EQ(graphics::CreateCore(api, {}, &core), 0);
    ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
    std::unique_ptr<graphics::Shader> shader;
    ASSERT_EQ(core->CreateShader("[numthreads(1,1,1)] void Main() {}", "Main", "cs_6_0", &shader), 0);
    ASSERT_NE(shader, nullptr);
    shader.reset();
    EXPECT_NE(core->CreateShader("this is invalid source", "Main", "cs_6_0", &shader), 0);
    EXPECT_EQ(shader, nullptr);
  }
}

TEST(SlangCompiler, LoadedVersionRequirement) {
  for (const char *tag :
       {"2026.18.1", "v2026.18.1", "2026.18.1-12-gabc123", "2026.18.2", "2026.18.3", "2026.18.1.1", "2027.1"}) {
    SCOPED_TRACE(tag);
    EXPECT_TRUE(graphics::detail::SlangVersionSupported(tag));
  }
  for (const char *tag : {"2026.1-52-gc8ddf20bb", "2026.11", "2026.18", "2026.18.0", "2026.18.1-rc1", "unknown", "",
                          "2026.999999999999999999999999999999.1", "2026.18.1garbage"}) {
    SCOPED_TRACE(tag);
    EXPECT_FALSE(graphics::detail::SlangVersionSupported(tag));
  }
}
