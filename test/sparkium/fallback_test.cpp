#include <gtest/gtest.h>
#include <long_march.h>

#include <glm/gtc/matrix_transform.hpp>
#include <numeric>
#include <random>

#include "sparkium/pipelines/raytracing/core/core.h"
#include "sparkium/pipelines/raytracing/core/software_pipeline.h"
#include "sparkium/pipelines/raytracing/geometry/geometry_mesh.h"
#include "sparkium/pipelines/raytracing/material/material_lambertian.h"

using namespace grassland;
namespace {
struct Ray {
  glm::vec3 origin;
  float t_min;
  glm::vec3 direction;
  float t_max;
};
struct Hit {
  float distance;
  uint32_t instance, primitive, found;
  float u, v, padding[2];
};
static_assert(sizeof(Ray) == 32 && sizeof(Hit) == 32);

class SoftwareBVHTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(graphics::CreateCore(graphics::BACKEND_API_DEFAULT, graphics::Core::Settings{2, false}, &graphics), 0);
    ASSERT_EQ(graphics->InitializeLogicalDeviceAutoSelect(false), 0);
    core = std::make_unique<sparkium::Core>(graphics.get());
  }
  std::unique_ptr<graphics::Core> graphics;
  std::unique_ptr<sparkium::Core> core;
};

Hit Oracle(const Ray &ray, const std::vector<Vector3<float>> &positions, const std::vector<glm::mat4x3> &transforms) {
  Hit hit{};
  hit.distance = ray.t_max;
  hit.instance = hit.primitive = ~0u;
  for (uint32_t instance = 0; instance < transforms.size(); ++instance) {
    for (uint32_t triangle = 0; triangle < positions.size() / 3; ++triangle) {
      glm::dvec3 p[3];
      for (int v = 0; v < 3; ++v) {
        auto vertex = positions[triangle * 3 + v];
        p[v] = glm::dmat4(transforms[instance]) * glm::dvec4(vertex.x(), vertex.y(), vertex.z(), 1.0);
      }
      const glm::dvec3 e1 = p[1] - p[0], e2 = p[2] - p[0], direction(ray.direction);
      const glm::dvec3 cross = glm::cross(direction, e2);
      double determinant = glm::dot(e1, cross);
      if (std::abs(determinant) < 1e-12)
        continue;
      glm::dvec3 delta = glm::dvec3(ray.origin) - p[0];
      double u = glm::dot(delta, cross) / determinant;
      glm::dvec3 q = glm::cross(delta, e1);
      double v = glm::dot(direction, q) / determinant;
      double t = glm::dot(e2, q) / determinant;
      if (u < 0 || v < 0 || u + v > 1 || t < ray.t_min || t >= hit.distance)
        continue;
      hit = {static_cast<float>(t), instance, triangle, 1, static_cast<float>(u), static_cast<float>(v), {0, 0}};
    }
  }
  return hit;
}

class SoftwareBVHSizeTest : public SoftwareBVHTest, public testing::WithParamInterface<int> {};

TEST_P(SoftwareBVHSizeTest, ComputeConstructionAndTraversalMatchDoublePrecisionOracle) {
  // Five triangles: a non-power-of-two tree, duplicate centroids and a degenerate leaf.
  std::vector<Vector3<float>> positions{{-1, -1, 0}, {1, -1, 0},  {0, 1, 0},  {-1, -1, -1}, {1, -1, -1},
                                        {0, 1, -1},  {-2, -1, 0}, {-2, 1, 0}, {-2, 0, 2},   {-1, -1, 0},
                                        {0, 1, 0},   {1, -1, 0},  {0, 0, 0},  {0, 0, 0},    {0, 0, 0}};
  std::mt19937 geometry_random(951);
  std::uniform_real_distribution<float> coordinate(-3.0f, 3.0f);
  while (positions.size() < GetParam() * 3) {
    Vector3<float> center(coordinate(geometry_random), coordinate(geometry_random), coordinate(geometry_random));
    positions.push_back(center + Vector3<float>(-0.3f, -0.2f, 0));
    positions.push_back(center + Vector3<float>(0.3f, -0.2f, 0.1f));
    positions.push_back(center + Vector3<float>(0, 0.3f, -0.1f));
  }
  positions.resize(GetParam() * 3);
  std::vector<uint32_t> indices(positions.size());
  std::iota(indices.begin(), indices.end(), 0);
  Mesh<> mesh(positions.size(), indices.size(), indices.data(), positions.data());
  sparkium::GeometryMesh geometry(core.get(), mesh);
  sparkium::MaterialLambertian material(core.get());
  sparkium::raytracing::GeometryMesh rt_geometry(geometry);
  sparkium::raytracing::MaterialLambertian rt_material(material);
  sparkium::raytracing::SoftwarePipeline pipeline(sparkium::raytracing::DedicatedCast(core.get()));
  std::vector<graphics::Buffer *> buffers{rt_geometry.Buffer()};

  std::mt19937 random(7411);
  std::uniform_real_distribution<float> position(-5.0f, 5.0f);
  std::vector<Ray> rays{{{0, 0, 4}, 0, {0, 0, -1}, 100}, {{0, 0, 4}, 0, {0, 0, 1}, 100},
                        {{0, 0, 0}, 0, {0, 0, -1}, 100}, {{0, 0, 4}, 0, {0, 0, -1}, 1},
                        {{10, 0, 0}, 0, {0, 1, 0}, 100}, {{0, 0, 4}, 4.1f, {0, 0, -1}, 100}};
  for (int i = 0; i < 1024; ++i) {
    glm::vec3 origin(position(random), position(random), 4.0f);
    glm::vec3 target(position(random), position(random), -1.0f);
    rays.push_back({origin, 0.001f, glm::normalize(target - origin), 100.0f});
  }
  std::unique_ptr<graphics::Buffer> ray_buffer, output;
  graphics->CreateBuffer(16 + rays.size() * sizeof(Ray), graphics::BUFFER_TYPE_STATIC, &ray_buffer);
  uint32_t count = rays.size();
  ray_buffer->UploadData(&count, sizeof(count));
  ray_buffer->UploadData(rays.data(), rays.size() * sizeof(Ray), 16);
  graphics->CreateBuffer(rays.size() * sizeof(Hit), graphics::BUFFER_TYPE_STATIC, &output);

  auto vfs = core->GetShadersVFS();
  vfs.WriteFile("bvh_test.hlsl", R"(
#define SOFTWARE_EXTERNAL_BINDINGS
ByteAddressBuffer software_nodes : register(t0, space0);
ByteAddressBuffer software_instances : register(t0, space1);
ByteAddressBuffer data_buffers[] : register(t0, space2);
ByteAddressBuffer rays : register(t0, space3);
RWByteAddressBuffer results : register(u0, space4);
#include "software/traversal.hlsli"
[numthreads(64, 1, 1)] void Main(uint3 id : SV_DispatchThreadID) {
  if (id.x >= rays.Load(0)) return;
  uint offset = 16 + id.x * 32;
  RayDesc ray;
  ray.Origin = asfloat(rays.Load3(offset)); ray.TMin = asfloat(rays.Load(offset + 12));
  ray.Direction = asfloat(rays.Load3(offset + 16)); ray.TMax = asfloat(rays.Load(offset + 28));
  SoftwareHit hit;
  bool found = SoftwareIntersect(ray, false, hit);
  results.Store4(id.x * 32, uint4(asuint(hit.distance), hit.instance, hit.primitive, uint(found)));
  results.Store4(id.x * 32 + 16, uint4(asuint(hit.barycentric), 0, 0));
}
)");
  std::unique_ptr<graphics::Shader> shader;
  ASSERT_EQ(graphics->CreateShader(vfs, "bvh_test.hlsl", "Main", "cs_6_0", {"-I."}, &shader), 0);
  std::unique_ptr<graphics::ComputeProgram> program;
  graphics->CreateComputeProgram(shader.get(), &program);
  for (int i = 0; i < 4; ++i)
    program->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  program->Finalize();

  glm::mat4 mirrored =
      glm::translate(glm::mat4(1), glm::vec3(3, 0, -1)) * glm::scale(glm::mat4(1), glm::vec3(-0.7f, 1.8f, 0.4f));
  std::vector<glm::mat4x3> transforms{glm::mat4x3(1), glm::mat4x3(mirrored)};
  for (int phase = 0; phase < 4; ++phase) {
    // Refit instances, shrink the instance tree, then exercise a completely empty TLAS.
    if (phase == 1)
      transforms[0] = glm::mat4x3(glm::translate(glm::mat4(1), glm::vec3(-1, 0.5f, -2)));
    if (phase == 2)
      transforms.resize(1);
    if (phase == 3)
      transforms.clear();
    pipeline.ClearInstances();
    for (auto transform : transforms)
      pipeline.AddInstance(&rt_geometry, &rt_material, transform, 0);
    std::unique_ptr<graphics::CommandContext> commands;
    graphics->CreateCommandContext(&commands);
    pipeline.Update(commands.get(), buffers, 1, 1);
    commands->CmdBindComputeProgram(program.get());
    commands->CmdBindResources(0, {pipeline.Nodes()}, graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(1, {pipeline.Instances()}, graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(2, buffers, graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(3, {ray_buffer.get()}, graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(4, {output.get()}, graphics::BIND_POINT_COMPUTE);
    commands->CmdDispatch((rays.size() + 63) / 64, 1, 1);
    graphics->SubmitCommandContext(commands.get());
    graphics->WaitGPU();
    std::vector<Hit> actual(rays.size());
    output->DownloadData(actual.data(), actual.size() * sizeof(Hit));
    for (size_t i = 0; i < rays.size(); ++i) {
      SCOPED_TRACE(testing::Message() << "phase " << phase << ", ray " << i);
      const Hit expected = Oracle(rays[i], positions, transforms);
      ASSERT_EQ(actual[i].found, expected.found);
      if (expected.found) {
        EXPECT_NEAR(actual[i].distance, expected.distance, 2.0e-4f * std::max(1.0f, expected.distance));
        EXPECT_EQ(actual[i].instance, expected.instance);
        if (actual[i].primitive == expected.primitive) {
          EXPECT_NEAR(actual[i].u, expected.u, 1.0e-4f);
          EXPECT_NEAR(actual[i].v, expected.v, 1.0e-4f);
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(TreeSizes, SoftwareBVHSizeTest, testing::Values(1, 5, 257));

TEST_F(SoftwareBVHTest, EmptySceneBackgroundAccumulationAndReset) {
  sparkium::Scene scene(core.get());
  scene.settings.samples_per_dispatch = 3;
  scene.settings.background_color = glm::vec3(0.2f, 0.4f, 0.7f);
  sparkium::Camera camera(core.get(), glm::mat4(1), glm::radians(45.0f), 17.0f / 13.0f);
  sparkium::Film film(core.get(), 17, 13);
  auto check = [&](glm::vec3 expected) {
    std::vector<glm::vec4> pixels(17 * 13);
    film.GetRawImage()->DownloadData(pixels.data());
    for (auto pixel : pixels)
      for (int c = 0; c < 3; ++c) {
        ASSERT_TRUE(std::isfinite(pixel[c]));
        EXPECT_NEAR(pixel[c], expected[c], 1e-5f);
      }
  };
  for (int frame = 1; frame <= 2; ++frame) {
    core->Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_RT_FALLBACK);
    EXPECT_EQ(film.info.accumulated_samples, frame * 3);
    check(scene.settings.background_color);
  }
  film.Reset();
  EXPECT_EQ(film.info.accumulated_samples, 0);
  scene.settings.background_color = glm::vec3(0.1f, 0.3f, 0.9f);
  core->Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_RT_FALLBACK);
  EXPECT_EQ(film.info.accumulated_samples, 3);
  check(scene.settings.background_color);
}

TEST_F(SoftwareBVHTest, TransparentShadowLayers) {
  std::vector<Vector3<float>> positions{{-2, -2, 0}, {2, -2, 0}, {0, 2, 0}, {-2, -2, -1}, {2, -2, -1}, {0, 2, -1}};
  uint32_t indices[]{0, 1, 2, 3, 4, 5};
  Mesh<> mesh(6, 6, indices, positions.data());
  sparkium::GeometryMesh geometry(core.get(), mesh);
  sparkium::MaterialLambertian material(core.get());
  sparkium::raytracing::GeometryMesh rt_geometry(geometry);
  sparkium::raytracing::MaterialLambertian rt_material(material);
  sparkium::raytracing::SoftwarePipeline pipeline(sparkium::raytracing::DedicatedCast(core.get()));
  pipeline.AddInstance(&rt_geometry, &rt_material, glm::mat4x3(1), 0);
  auto vfs = core->GetShadersVFS();
  vfs.WriteFile("shadow_test.hlsl", R"(
#define SOFTWARE_EXTERNAL_BINDINGS
#include "common.hlsli"
ByteAddressBuffer software_nodes : register(t0, space0);
ByteAddressBuffer software_instances : register(t0, space1);
ByteAddressBuffer data_buffers[] : register(t0, space2);
RWByteAddressBuffer results : register(u0, space3);
#include "software/traversal.hlsli"
HitRecord SoftwareHitRecord(SoftwareHit hit, float3 direction) { return (HitRecord)0; }
float SoftwareShadowTransmission(uint material, HitRecord hit, float3 direction) { return 0.5f; }
#include "software/shadow.hlsli"
[numthreads(1, 1, 1)] void Main() {
  float3 o = float3(0, 0, 4), d = float3(0, 0, -1);
  results.Store4(0, asuint(float4(ShadowRay(o, d, 10), ShadowRayNoAlpha(o, d, 10),
                                  ShadowRay(o, d, 4.5f), ShadowRay(o, d, 3.0f))));
}
)");
  std::unique_ptr<graphics::Shader> shader;
  ASSERT_EQ(graphics->CreateShader(vfs, "shadow_test.hlsl", "Main", "cs_6_0", {"-I."}, &shader), 0);
  std::unique_ptr<graphics::ComputeProgram> program;
  graphics->CreateComputeProgram(shader.get(), &program);
  for (int i = 0; i < 3; ++i)
    program->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  program->Finalize();
  std::unique_ptr<graphics::Buffer> output;
  graphics->CreateBuffer(16, graphics::BUFFER_TYPE_STATIC, &output);
  std::unique_ptr<graphics::CommandContext> commands;
  graphics->CreateCommandContext(&commands);
  pipeline.Update(commands.get(), {rt_geometry.Buffer()}, 1, 1);
  commands->CmdBindComputeProgram(program.get());
  commands->CmdBindResources(0, {pipeline.Nodes()}, graphics::BIND_POINT_COMPUTE);
  commands->CmdBindResources(1, {pipeline.Instances()}, graphics::BIND_POINT_COMPUTE);
  commands->CmdBindResources(2, {rt_geometry.Buffer()}, graphics::BIND_POINT_COMPUTE);
  commands->CmdBindResources(3, {output.get()}, graphics::BIND_POINT_COMPUTE);
  commands->CmdDispatch(1, 1, 1);
  graphics->SubmitCommandContext(commands.get());
  graphics->WaitGPU();
  float actual[4];
  output->DownloadData(actual, sizeof(actual));
  EXPECT_FLOAT_EQ(actual[0], 0.25f);
  EXPECT_FLOAT_EQ(actual[1], 0.0f);
  EXPECT_FLOAT_EQ(actual[2], 0.5f);
  EXPECT_FLOAT_EQ(actual[3], 1.0f);
}

TEST_F(SoftwareBVHTest, SharedShadersCompileForNativeRayTracingAndCompute) {
  auto vfs = core->GetShadersVFS();
  for (bool spirv : {false, true}) {
    std::vector<std::string> args{"-I."};
    if (spirv)
      args.insert(args.end(), {"-spirv", "-fspv-target-env=vulkan1.2", "-fvk-use-dx-layout"});
    auto compile = [&](const char *file, const char *entry, const char *target) {
      SCOPED_TRACE(testing::Message() << file << " / " << entry << ", SPIR-V=" << spirv);
      EXPECT_FALSE(graphics::CompileShader(vfs, file, entry, target, args).data.empty());
    };
    for (auto entry : {"Main", "MissMain", "ShadowMiss"})
      compile("raygen.hlsl", entry, "lib_6_5");
    compile("camera.hlsl", "CameraPinhole", "lib_6_5");
    for (auto material : {"lambertian", "light", "principled", "specular"}) {
      vfs.WriteFile("material_sampler.hlsli",
                    sparkium::CodeLines(vfs, std::string("material/") + material + "/sampler.hlsl"));
      compile("geometry/mesh/hit_group.hlsl", "RenderClosestHit", "lib_6_5");
      compile("geometry/mesh/hit_group.hlsl", "ShadowClosestHit", "lib_6_5");
    }
    for (auto entry : {"InitLeaves", "ReduceNodes", "MortonKeys", "BitonicSort", "SortLeaves"})
      compile("software/build.hlsl", entry, "cs_6_0");
  }
}

TEST_F(SoftwareBVHTest, HardwareImageParity) {
  if (!graphics->DeviceRayTracingSupport())
    GTEST_SKIP() << "requires a hardware ray tracing device";
  std::vector<Vector3<float>> positions{{-1, -1, 0}, {1, -1, 0}, {0, 1, 0}};
  uint32_t indices[]{0, 1, 2};
  Mesh<> mesh(3, 3, indices, positions.data());
  sparkium::GeometryMesh geometry(core.get(), mesh);
  sparkium::MaterialLambertian material(core.get(), glm::vec3(0.4f), glm::vec3(0.3f, 0.5f, 0.7f));
  sparkium::EntityGeometryMaterial entity(core.get(), &geometry, &material);
  sparkium::Scene scene(core.get());
  scene.AddEntity(&entity);
  scene.settings.samples_per_dispatch = 16;
  scene.settings.max_bounces = 4;
  scene.settings.background_color = glm::vec3(0.1f);
  sparkium::Camera camera(core.get(), glm::lookAt(glm::vec3(0, 0, 4), glm::vec3(0), glm::vec3(0, 1, 0)),
                          glm::radians(45.0f), 1.0f);
  sparkium::Film software(core.get(), 32, 32), hardware(core.get(), 32, 32);
  core->Render(&scene, &camera, &software, sparkium::RENDER_PIPELINE_RT_FALLBACK);
  core->Render(&scene, &camera, &hardware, sparkium::RENDER_PIPELINE_RAY_TRACING);
  std::vector<glm::vec4> a(1024), b(1024);
  software.GetRawImage()->DownloadData(a.data());
  hardware.GetRawImage()->DownloadData(b.data());
  double error = 0;
  for (size_t i = 0; i < a.size(); ++i)
    for (int c = 0; c < 3; ++c) {
      ASSERT_TRUE(std::isfinite(a[i][c]) && std::isfinite(b[i][c]));
      error += std::abs(a[i][c] - b[i][c]);
    }
  EXPECT_LT(error / (a.size() * 3), 0.01);
}
}  // namespace
