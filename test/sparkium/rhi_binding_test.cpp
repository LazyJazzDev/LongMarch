#include <gtest/gtest.h>

#include "grassland/rhi/rhi.h"
#include "grassland/util/virtual_file_system.h"
#include "sparkium/core/render_layout.h"

namespace {
using namespace grassland;
using namespace grassland::rhi;

class SparkiumBindingTest : public testing::TestWithParam<BackendAPI> {
 protected:
  void SetUp() override {
    core_ = CreateCore(GetParam());
    ASSERT_TRUE(core_);
    device_ = core_->CreateDeviceByAutoSelection();
    ASSERT_TRUE(device_);
    shaders_ = VirtualFileSystem::LoadDirectory(LONGMARCH_SPARKIUM_SHADERS);
    std::vector<uint8_t> bindings;
    ASSERT_EQ(shaders_.ReadFile("rhi_bindings.slang", bindings), 0);
    ASSERT_EQ(shaders_.WriteFile("bindings.slang", bindings), 0);
  }

  ObjPtr<IShader> Shader(const std::string &file,
                         const std::string &entry,
                         ShaderStage stage,
                         const std::vector<std::pair<std::string, std::string>> &defines = {}) {
    return device_->CreateShader(file, entry, stage, &shaders_, {"."}, defines);
  }

  ObjPtr<ICore> core_;
  ObjPtr<IDevice> device_;
  VirtualFileSystem shaders_;
};

TEST_P(SparkiumBindingTest, SharedPathTracerCompilesWithFourGroups) {
  shaders_.WriteFile("software_camera.slang", "#include \"camera/pinhole.slang\"\n");
  shaders_.WriteFile("software_materials.slang", R"(
#include "material/lambertian/sampler.slang"
void SoftwareSampleMaterial(uint material, inout RenderContext context, HitRecord hit) {
  SampleMaterial(context, hit);
}
float SoftwareShadowTransmission(uint material, HitRecord hit, float3 direction) {
  return 0.0f;
}
)");
  for (bool query : {false, true}) {
    if (query && !device_->SupportRayQuery())
      continue;
    SCOPED_TRACE(query);
    std::vector<std::pair<std::string, std::string>> defines{{"SPARKIUM_CAMERA_ENTRY", "CameraPinhole"}};
    if (query)
      defines.emplace_back("SPARKIUM_RAY_QUERY", "1");
    auto shader = Shader("software/render.slang", "Main", SHADER_STAGE_COMPUTE, defines);
    ASSERT_TRUE(shader);
    // More than one element exercises array placement within the Scene group,
    // alongside scalar descriptors of different types.
    ObjPtr<IComputeProgram> program =
        device_->CreateComputeProgram({shader.Get(), sparkium::RayTracingLayout(!query, 7, 3, 2)});
    ASSERT_TRUE(program);
  }
}

TEST_P(SparkiumBindingTest, NativeHitAndCallableShadersShareFourGroups) {
  if (!device_->SupportRayTracingPipeline())
    GTEST_SKIP();
  auto raygen = Shader("raygen.slang", "Main", SHADER_STAGE_RAY_GENERATION);
  auto miss = Shader("raygen.slang", "MissMain", SHADER_STAGE_MISS);
  auto shadow_miss = Shader("raygen.slang", "ShadowMiss", SHADER_STAGE_MISS);
  auto camera = Shader("camera/pinhole.slang", "CameraPinhole", SHADER_STAGE_CALLABLE);
  ASSERT_TRUE(raygen && miss && shadow_miss && camera);
  for (const auto *material : {"lambertian", "light", "principled", "specular"}) {
    SCOPED_TRACE(material);
    shaders_.WriteFile("material_sampler.slang", "#include \"material/" + std::string(material) + "/sampler.slang\"\n");
    auto hit = Shader("geometry/mesh/hit_group.slang", "RenderClosestHit", SHADER_STAGE_CLOSEST_HIT);
    auto shadow = Shader("geometry/mesh/hit_group.slang", "ShadowClosestHit", SHADER_STAGE_CLOSEST_HIT);
    ASSERT_TRUE(hit && shadow);
    RayTracingProgramDesc desc;
    desc.ray_generation = raygen.Get();
    desc.miss_shaders = {miss.Get(), shadow_miss.Get()};
    desc.hit_groups = {{hit.Get()}, {shadow.Get()}};
    desc.callable_shaders = {camera.Get()};
    desc.pipeline_layout = sparkium::RayTracingLayout(false, 7, 3, 2);
    desc.max_recursion_depth = 2;
    // RenderContext includes subsurface state and currently occupies 156 bytes.
    desc.max_payload_size = 160;
    ObjPtr<IRayTracingProgram> program = device_->CreateRayTracingProgram(desc);
    ASSERT_TRUE(program);
  }
}

const std::vector<BackendAPI> backends{
#if defined(LONGMARCH_VULKAN_ENABLED)
    BACKEND_API_VULKAN,
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
    BACKEND_API_D3D12,
#endif
#if defined(LONGMARCH_METAL_ENABLED)
    BACKEND_API_METAL,
#endif
};
INSTANTIATE_TEST_SUITE_P(Backends, SparkiumBindingTest, testing::ValuesIn(backends));
}  // namespace
