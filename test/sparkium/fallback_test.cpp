#include <gtest/gtest.h>
#include <long_march.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <numeric>
#include <random>
#include <tuple>

#include "../../demo/sparkium_backend.h"
#include "grassland/graphics/backend/backend.h"
#include "grassland/graphics/frame_profile.h"
#include "sparkium/pipelines/raytracing/core/core.h"
#include "sparkium/pipelines/raytracing/core/software_pipeline.h"
#include "sparkium/pipelines/raytracing/entity/entities.h"
#include "sparkium/pipelines/raytracing/geometry/geometry_mesh.h"
#include "sparkium/pipelines/raytracing/material/material_lambertian.h"
#include "sparkium/pipelines/realtime/core/core.h"
#include "sparkium/pipelines/realtime/geometry/geometry_mesh.h"
#include "sparkium/pipelines/realtime/material/material_lambertian.h"

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
    const char *backend = std::getenv("SPARKIUM_TEST_BACKEND");
    const bool debug = std::getenv("SPARKIUM_TEST_DEBUG") != nullptr;
    ASSERT_EQ(graphics::CreateCore(backend ? ParseSparkiumBackend(backend) : graphics::BACKEND_API_DEFAULT,
                                   graphics::Core::Settings{2, debug}, &graphics),
              0);
    ASSERT_EQ(graphics->InitializeLogicalDeviceAutoSelect(false), 0);
    core = std::make_unique<sparkium::Core>(graphics.get());
  }

  std::unique_ptr<graphics::Core> graphics;
  std::unique_ptr<sparkium::Core> core;
};

TEST_F(SoftwareBVHTest, HDRFilmDevelopmentPreservesHighlightsAndAccumulation) {
  sparkium::Film film(core.get(), 9, 3);
  std::vector<glm::vec4> source(27, glm::vec4(4.0f, 0.25f, -1.0f, 1.0f));
  source.back() = glm::vec4(100000.0f, 1.0f, 0.0f, 1.0f);
  film.GetRawImage()->UploadData(source.data());
  film.info.accumulated_samples = 17;
  film.info.exposure = 1.0f;
  film.info.gamma = 0.4f;
  film.info.contrast = 2.0f;
  int resets = 0;
  film.RegisterResetCallback([&]() { ++resets; });
  std::unique_ptr<graphics::Image> sdr, hdr;
  graphics->CreateImage(9, 3, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &sdr);
  graphics->CreateImage(9, 3, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &hdr);
  EXPECT_THROW(film.Develop(sdr.get(), true), std::invalid_argument);
  for (int transform : {0, 1, 2}) {
    film.info.view_transform = transform;
    film.Develop(sdr.get());
    std::vector<uint8_t> before(27 * 4), after(27 * 4);
    sdr->DownloadData(before.data());
    film.Develop(hdr.get(), true);
    std::vector<glm::vec4> actual(27);
    hdr->DownloadData(actual.data());
    for (size_t i = 0; i + 1 < actual.size(); ++i) {
      EXPECT_NEAR(actual[i].x, 8.0f, 1e-5f);
      EXPECT_NEAR(actual[i].y, 0.5f, 1e-5f);
      EXPECT_EQ(actual[i].z, 0.0f);
      EXPECT_EQ(actual[i].w, 1.0f);
    }
    EXPECT_EQ(actual.back().x, 65504.0f);
    film.Develop(sdr.get());
    sdr->DownloadData(after.data());
    EXPECT_EQ(before, after);
    EXPECT_EQ(film.info.accumulated_samples, 17);
    EXPECT_EQ(film.info.view_transform, transform);
  }
  std::vector<glm::vec4> unchanged(27);
  film.GetRawImage()->DownloadData(unchanged.data());
  EXPECT_EQ(unchanged, source);
  EXPECT_EQ(resets, 0);
}

TEST_F(SoftwareBVHTest, RealtimeRasterVisibilityPreservesHDRAndRejectsStaleHistory) {
  uint32_t indices[]{0, 1, 2, 0, 2, 3};
  std::vector<Vector3<float>> positions{{-20, -20, 0}, {20, -20, 0}, {20, 20, 0}, {-20, 20, 0}};
  Mesh<> mesh(4, 6, indices, positions.data());
  sparkium::GeometryMesh geometry(core.get(), mesh);
  sparkium::MaterialLambertian material(core.get(), glm::vec3(0), glm::vec3(4, 0.5f, 0.25f));
  sparkium::EntityGeometryMaterial entity(core.get(), &geometry, &material);
  sparkium::Scene scene(core.get());
  scene.AddEntity(&entity);
  scene.settings.background_color = glm::vec3(0.125f);
  scene.settings.realtime.bounces = 2;
  scene.settings.realtime.updates = 1;
  sparkium::Camera camera(core.get(), glm::lookAt(glm::vec3(0, 0, 4), glm::vec3(0), glm::vec3(0, 1, 0)),
                          glm::radians(45.0f), 37.0f / 19.0f);
  sparkium::Film film(core.get(), 37, 19);
  std::vector<glm::vec4> pixels(37 * 19);
  for (int frame = 0; frame < 3; ++frame) {
    graphics::FrameProfile profile(graphics.get(), false);
    profile.Begin();
    core->Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_REALTIME);
    profile.Finish();
    EXPECT_EQ(profile.counters["realtime_software_gi"], 1);
    EXPECT_EQ(profile.counters["native_ray_query"], 0);
    if (frame > 0)
      EXPECT_EQ(profile.counters["bvh_dispatches"], 0);
    film.GetRawImage()->DownloadData(pixels.data());
    for (auto pixel : pixels) {
      EXPECT_NEAR(pixel.r, 4, 0.002f);
      EXPECT_NEAR(pixel.g, 0.5f, 0.002f);
      EXPECT_NEAR(pixel.b, 0.25f, 0.002f);
      EXPECT_EQ(pixel.a, 1);
    }
  }
  // Realtime rendering must not construct any path-tracing implementation.
  EXPECT_EQ(core->GetComponent<sparkium::raytracing::Core>(), nullptr);
  EXPECT_EQ(geometry.GetComponent<sparkium::raytracing::GeometryMesh>(), nullptr);
  EXPECT_EQ(material.GetComponent<sparkium::raytracing::MaterialLambertian>(), nullptr);
  auto *realtime_core = core->GetComponent<sparkium::realtime::Core>();
  auto *realtime_material = material.GetComponent<sparkium::realtime::MaterialLambertian>();
  ASSERT_NE(realtime_core, nullptr);
  ASSERT_NE(realtime_material, nullptr);
  ASSERT_NE(geometry.GetComponent<sparkium::realtime::GeometryMesh>(), nullptr);
  auto *realtime_scan = realtime_core->GetComputeProgram("blelloch_scan_up");
  ASSERT_NE(realtime_scan, nullptr);
  std::vector<uint8_t> shader_source;
  EXPECT_NE(realtime_core->GetShadersVFS().ReadFile("geometry/mesh/hit_group.hlsl", shader_source), 0);
  // Independent pipelines must not inherit one another's film accumulation.
  scene.settings.samples_per_dispatch = 1;
  core->Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_RT_FALLBACK);
  auto *path_core = core->GetComponent<sparkium::raytracing::Core>();
  auto *path_material = material.GetComponent<sparkium::raytracing::MaterialLambertian>();
  ASSERT_NE(path_core, nullptr);
  ASSERT_NE(path_material, nullptr);
  std::vector<uint8_t> realtime_bsdf, path_bsdf;
  EXPECT_EQ(realtime_core->GetShadersVFS().ReadFile("bsdf/principled_bsdf.hlsli", realtime_bsdf), 0);
  EXPECT_EQ(path_core->GetShadersVFS().ReadFile("bsdf/principled_bsdf.hlsli", path_bsdf), 0);
  EXPECT_EQ(realtime_bsdf, path_bsdf);
  EXPECT_NE(path_material->Buffer(), realtime_material->Buffer());
  EXPECT_NE(path_core->GetComputeProgram("blelloch_scan_up"), realtime_scan);
  EXPECT_EQ(realtime_core->GetComputeProgram("blelloch_scan_up"), realtime_scan);
  EXPECT_EQ(path_core->GetShadersVFS().ReadFile("geometry/mesh/hit_group.hlsl", shader_source), 0);
  EXPECT_EQ(film.info.accumulated_samples, 1);
  core->Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_REALTIME);
  EXPECT_EQ(film.info.accumulated_samples, 1);
  film.GetRawImage()->DownloadData(pixels.data());
  for (auto pixel : pixels)
    EXPECT_NEAR(pixel.r, 4, 0.002f);
  // Interleaved updates must cover the entire grid after one complete period.
  scene.settings.realtime.updates = 4;
  for (int frame = 0; frame < 4; ++frame)
    core->Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_REALTIME);
  film.GetRawImage()->DownloadData(pixels.data());
  for (auto pixel : pixels)
    EXPECT_NEAR(pixel.r, 4, 0.002f);
  scene.settings.realtime.updates = 1;
  // A camera cut reveals the background. Old radiance must not bleed into it.
  camera.view = glm::lookAt(glm::vec3(100, 0, 4), glm::vec3(100, 0, 0), glm::vec3(0, 1, 0));
  core->Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_REALTIME);
  film.GetRawImage()->DownloadData(pixels.data());
  for (auto pixel : pixels)
    EXPECT_NEAR(pixel.r, 0.125f, 0.002f);
  // Reset invalidates history after an explicit material edit.
  camera.view = glm::lookAt(glm::vec3(0, 0, 4), glm::vec3(0), glm::vec3(0, 1, 0));
  material.emission = glm::vec3(0.2f);
  film.Reset();
  core->Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_REALTIME);
  film.GetRawImage()->DownloadData(pixels.data());
  for (auto pixel : pixels)
    EXPECT_NEAR(pixel.r, 0.2f, 0.002f);
  EXPECT_EQ(film.info.accumulated_samples, 1);
  // Transform edits update software visibility and reset per-view lighting history.
  entity.transform = glm::mat4x3(glm::translate(glm::mat4(1), glm::vec3(100, 0, 0)));
  core->Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_REALTIME);
  film.GetRawImage()->DownloadData(pixels.data());
  for (auto pixel : pixels)
    EXPECT_NEAR(pixel.r, 0.125f, 0.002f);
  EXPECT_EQ(film.info.accumulated_samples, 1);
}

TEST(GraphicsCoreCreation, UnsupportedAPIsDoNotFallBack) {
  for (auto api : {graphics::BACKEND_API_METAL, graphics::BACKEND_API_D3D12, graphics::BACKEND_API_VULKAN,
                   static_cast<graphics::BackendAPI>(99)}) {
    if (graphics::SupportBackendAPI(api))
      continue;
    std::unique_ptr<graphics::Core> core;
    EXPECT_EQ(graphics::CreateCore(api, graphics::Core::Settings{}, &core), -1);
    EXPECT_EQ(core, nullptr);
  }
}

TEST_F(SoftwareBVHTest, LightSamplingIsIndependentOfEntityAddresses) {
  std::vector<std::unique_ptr<sparkium::EntityPointLight>> owned;
  std::vector<std::pair<sparkium::raytracing::Entity *, sparkium::EntityPointLight *>> lights;
  for (int i = 0; i < 4; ++i) {
    owned.push_back(std::make_unique<sparkium::EntityPointLight>(core.get()));
    lights.emplace_back(sparkium::raytracing::DedicatedCast(owned.back().get()), owned.back().get());
  }

  std::sort(lights.begin(), lights.end(),
            [](const auto &a, const auto &b) { return std::less<sparkium::raytracing::Entity *>{}(a.first, b.first); });
  // Equivalent light lists, deliberately opposite address ordering in the
  // renderer's component cache. Sampling must follow insertion, not addresses.
  for (int i = 0; i < 4; ++i) {
    const bool red = i == 0 || i == 2;
    lights[i].second->position = red ? glm::vec3(-1, 1, 2) : glm::vec3(2, 0, 1);
    lights[i].second->color = red ? glm::vec3(1, 0.1f, 0.2f) : glm::vec3(0.1f, 0.2f, 1);
    lights[i].second->strength = red ? 10.0f : 40.0f;
  }

  std::vector<Vector3<float>> positions{{-2, -2, 0}, {2, -2, 0}, {0, 2, 0}};
  uint32_t indices[]{0, 1, 2};
  Mesh<> mesh(3, 3, indices, positions.data());
  sparkium::GeometryMesh geometry(core.get(), mesh);
  sparkium::MaterialLambertian material(core.get(), glm::vec3(0.6f));
  sparkium::EntityGeometryMaterial entity(core.get(), &geometry, &material);
  sparkium::Scene first(core.get()), second(core.get());
  for (auto *scene : {&first, &second}) {
    scene->AddEntity(&entity);
    scene->settings.samples_per_dispatch = 1;
    scene->settings.max_bounces = 2;
  }
  first.AddEntity(lights[0].second);
  first.AddEntity(lights[3].second);
  second.AddEntity(lights[2].second);
  second.AddEntity(lights[1].second);
  first.AddEntity(lights[0].second);  // Duplicate adds must not register twice.
  first.SetEntityActive(lights[0].second, false);
  first.SetEntityActive(lights[0].second, true);
  sparkium::Camera camera(core.get(), glm::lookAt(glm::vec3(0, 0, 4), glm::vec3(0), glm::vec3(0, 1, 0)),
                          glm::radians(45.0f), 1.0f);
  for (auto pipeline : {sparkium::RENDER_PIPELINE_RT_FALLBACK, sparkium::RENDER_PIPELINE_RAY_QUERY,
                        sparkium::RENDER_PIPELINE_RAY_TRACING, sparkium::RENDER_PIPELINE_REALTIME}) {
    if (pipeline == sparkium::RENDER_PIPELINE_RAY_QUERY && !graphics->DeviceRayQuerySupport())
      continue;
    if (pipeline == sparkium::RENDER_PIPELINE_RAY_TRACING && !graphics->DeviceRayTracingSupport())
      continue;
    SCOPED_TRACE(static_cast<int>(pipeline));
    sparkium::Film a(core.get(), 32, 32), b(core.get(), 32, 32);
    core->Render(&first, &camera, &a, pipeline);
    core->Render(&second, &camera, &b, pipeline);
    std::vector<glm::vec4> left(1024), right(1024);
    a.GetRawImage()->DownloadData(left.data());
    b.GetRawImage()->DownloadData(right.data());
    double radiance = 0;
    for (size_t i = 0; i < left.size(); ++i)
      for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(left[i][c], right[i][c], 1e-6f);
        radiance += left[i][c];
      }
    EXPECT_GT(radiance, 1.0);
  }
}

TEST_F(SoftwareBVHTest, NonblockingEmittersDoNotHideShadowOccluders) {
  uint32_t indices[]{0, 1, 2, 0, 2, 3};
  std::vector<Vector3<float>> receiver_positions{{-2, -2, 0}, {2, -2, 0}, {2, 2, 0}, {-2, 2, 0}};
  std::vector<Vector3<float>> wall_positions{{0, -10, -10}, {0, 10, -10}, {0, 10, 10}, {0, -10, 10}};
  Mesh<> receiver_mesh(4, 6, indices, receiver_positions.data());
  Mesh<> wall_mesh(4, 6, indices, wall_positions.data());
  sparkium::GeometryMesh receiver_geometry(core.get(), receiver_mesh), wall_geometry(core.get(), wall_mesh);
  sparkium::MaterialLambertian diffuse(core.get(), glm::vec3(0.7f));
  sparkium::MaterialLight invisible(core.get(), glm::vec3(0), false, false, false);
  sparkium::EntityGeometryMaterial receiver(core.get(), &receiver_geometry, &diffuse);
  sparkium::EntityGeometryMaterial emitter(core.get(), &wall_geometry, &invisible,
                                           glm::mat4x3(glm::translate(glm::mat4(1), glm::vec3(0.75f, 0, 0))));
  sparkium::EntityGeometryMaterial occluder(core.get(), &wall_geometry, &diffuse,
                                            glm::mat4x3(glm::translate(glm::mat4(1), glm::vec3(1.5f, 0, 0))));
  sparkium::EntityPointLight light(core.get(), glm::vec3(3, 0, 3), glm::vec3(1), 100.0f);
  sparkium::Scene scene(core.get());
  scene.AddEntity(&receiver);
  scene.AddEntity(&emitter);
  scene.AddEntity(&occluder);
  scene.AddEntity(&light);
  scene.settings.samples_per_dispatch = 16;
  scene.settings.max_bounces = 1;
  scene.settings.alpha_shadow = true;
  sparkium::Camera camera(core.get(), glm::lookAt(glm::vec3(0, 0, 4), glm::vec3(0), glm::vec3(0, 1, 0)),
                          glm::radians(20.0f), 1.0f);
  for (auto pipeline : {sparkium::RENDER_PIPELINE_RT_FALLBACK, sparkium::RENDER_PIPELINE_RAY_QUERY,
                        sparkium::RENDER_PIPELINE_RAY_TRACING}) {
    if (pipeline == sparkium::RENDER_PIPELINE_RAY_QUERY && !graphics->DeviceRayQuerySupport())
      continue;
    if (pipeline == sparkium::RENDER_PIPELINE_RAY_TRACING && !graphics->DeviceRayTracingSupport())
      continue;
    for (int mode = 0; mode < 3; ++mode) {
      SCOPED_TRACE(testing::Message() << "pipeline=" << pipeline << ", mode=" << mode);
      scene.SetEntityActive(&occluder, mode == 0);
      invisible.block_ray = mode == 2;
      sparkium::Film film(core.get(), 32, 32);
      core->Render(&scene, &camera, &film, pipeline);
      std::vector<glm::vec4> pixels(1024);
      film.GetRawImage()->DownloadData(pixels.data());
      double radiance = 0;
      for (const auto &pixel : pixels)
        for (int c = 0; c < 3; ++c) {
          ASSERT_TRUE(std::isfinite(pixel[c]));
          radiance += pixel[c];
          if (mode != 1)
            EXPECT_NEAR(pixel[c], 0.0f, 1e-6f);
        }
      if (mode == 1)
        EXPECT_GT(radiance, 1.0);
    }
  }
}

TEST_F(SoftwareBVHTest, RayQueryCapabilityMatchesDevice) {
#if defined(LONGMARCH_VULKAN_ENABLED)
  if (auto *vk = dynamic_cast<graphics::backend::VulkanCore *>(graphics.get())) {
    const auto &physical = vk->Device()->PhysicalDevice();
    EXPECT_EQ(graphics->DeviceRayQuerySupport(), physical.SupportRayQuery());
    if (physical.SupportRayQuery()) {
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
      std::unique_ptr<vulkan::Device> query_device;
      ASSERT_EQ(vk->Instance()->CreateDevice(physical, info, requirement.GetVmaAllocatorCreateFlags(), &query_device),
                VK_SUCCESS);
      EXPECT_NE(query_device->Procedures().vkCmdBuildAccelerationStructuresKHR, nullptr);
      EXPECT_EQ(query_device->Procedures().vkCmdTraceRaysKHR, nullptr);
    }
  }
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
  if (auto *dx = dynamic_cast<graphics::backend::D3D12Core *>(graphics.get())) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options{};
    const auto result =
        dx->Device()->Handle()->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options, sizeof(options));
    EXPECT_EQ(graphics->DeviceRayQuerySupport(),
              SUCCEEDED(result) && options.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1);
  }
#endif
}

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

class SoftwareBVHSizeTest : public SoftwareBVHTest, public testing::WithParamInterface<std::tuple<int, bool>> {};

TEST_P(SoftwareBVHSizeTest, ComputeConstructionAndTraversalMatchDoublePrecisionOracle) {
  auto [triangle_count, ray_query] = GetParam();
  if (ray_query && !graphics->DeviceRayQuerySupport())
    GTEST_SKIP() << "native ray query unavailable";
  // Five triangles: a non-power-of-two tree, duplicate centroids and a degenerate leaf.
  std::vector<Vector3<float>> positions{{-1, -1, 0}, {1, -1, 0},  {0, 1, 0},  {-1, -1, -1}, {1, -1, -1},
                                        {0, 1, -1},  {-2, -1, 0}, {-2, 1, 0}, {-2, 0, 2},   {-1, -1, 0},
                                        {0, 1, 0},   {1, -1, 0},  {0, 0, 0},  {0, 0, 0},    {0, 0, 0}};
  std::mt19937 geometry_random(951);
  std::uniform_real_distribution<float> coordinate(-3.0f, 3.0f);
  while (positions.size() < triangle_count * 3) {
    Vector3<float> center(coordinate(geometry_random), coordinate(geometry_random), coordinate(geometry_random));
    positions.push_back(center + Vector3<float>(-0.3f, -0.2f, 0));
    positions.push_back(center + Vector3<float>(0.3f, -0.2f, 0.1f));
    positions.push_back(center + Vector3<float>(0, 0.3f, -0.1f));
  }
  positions.resize(triangle_count * 3);
  std::vector<uint32_t> indices(positions.size());
  std::iota(indices.begin(), indices.end(), 0);
  Mesh<> mesh(positions.size(), indices.size(), indices.data(), positions.data());
  sparkium::GeometryMesh geometry(core.get(), mesh);
  sparkium::MaterialLambertian material(core.get());
  sparkium::raytracing::GeometryMesh rt_geometry(geometry);
  sparkium::raytracing::MaterialLambertian rt_material(material);
  sparkium::raytracing::SoftwarePipeline pipeline(sparkium::raytracing::DedicatedCast(core.get()), ray_query);
  std::vector<graphics::Buffer *> buffers{rt_geometry.Buffer()};

  std::mt19937 random(7411);
  std::uniform_real_distribution<float> position(-5.0f, 5.0f);
  // Match renderer self-intersection exclusion; exact t=0 acceptance differs across traversal APIs.
  std::vector<Ray> rays{{{0, 0, 4}, 0, {0, 0, -1}, 100},      {{0, 0, 4}, 0, {0, 0, 1}, 100},
                        {{0, 0, 0}, 0.001f, {0, 0, -1}, 100}, {{0, 0, 4}, 0, {0, 0, -1}, 1},
                        {{10, 0, 0}, 0, {0, 1, 0}, 100},      {{0, 0, 4}, 4.1f, {0, 0, -1}, 100}};
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

  auto vfs = sparkium::raytracing::DedicatedCast(core.get())->GetShadersVFS();
  vfs.WriteFile("bvh_test.hlsl", R"(
#define SOFTWARE_EXTERNAL_BINDINGS
#ifdef NATIVE_QUERY
RaytracingAccelerationStructure query_scene : register(t0, space0);
#else
ByteAddressBuffer software_nodes : register(t0, space0);
#endif
ByteAddressBuffer software_instances : register(t0, space1);
ByteAddressBuffer data_buffers[] : register(t0, space2);
ByteAddressBuffer rays : register(t0, space3);
RWByteAddressBuffer results : register(u0, space4);
#ifdef NATIVE_QUERY
#include "ray_query/traversal.hlsli"
#else
#include "software/traversal.hlsli"
#endif
[numthreads(64, 1, 1)] void Main(uint3 id : SV_DispatchThreadID) {
  if (id.x >= rays.Load(0)) return;
  uint offset = 16 + id.x * 32;
  RayDesc ray;
  ray.Origin = asfloat(rays.Load3(offset)); ray.TMin = asfloat(rays.Load(offset + 12));
  ray.Direction = asfloat(rays.Load3(offset + 16)); ray.TMax = asfloat(rays.Load(offset + 28));
  SoftwareHit hit;
  bool found = InlineIntersect(ray, false, hit);
  results.Store4(id.x * 32, uint4(asuint(hit.distance), hit.instance, hit.primitive, uint(found)));
  results.Store4(id.x * 32 + 16, uint4(asuint(hit.barycentric), 0, 0));
}
)");
  std::unique_ptr<graphics::Shader> shader;
  std::vector<std::string> args{"-I."};
  if (ray_query)
    args.push_back("-DNATIVE_QUERY");
  ASSERT_EQ(graphics->CreateShader(vfs, "bvh_test.hlsl", "Main", ray_query ? "cs_6_5" : "cs_6_0", args, &shader), 0);
  std::unique_ptr<graphics::ComputeProgram> program;
  graphics->CreateComputeProgram(shader.get(), &program);
  for (int i = 0; i < 4; ++i)
    program->AddResourceBinding(
        ray_query && i == 0 ? graphics::RESOURCE_TYPE_ACCELERATION_STRUCTURE : graphics::RESOURCE_TYPE_STORAGE_BUFFER,
        1);
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
    if (ray_query)
      commands->CmdBindResources(0, pipeline.AccelerationStructure(), graphics::BIND_POINT_COMPUTE);
    else
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

INSTANTIATE_TEST_SUITE_P(TreeSizes, SoftwareBVHSizeTest, testing::Combine(testing::Values(1, 5, 257), testing::Bool()));

class ComputeTraversalTest : public SoftwareBVHTest, public testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(TraversalModes, ComputeTraversalTest, testing::Bool());

TEST_P(ComputeTraversalTest, EmptySceneBackgroundAccumulationAndReset) {
  bool ray_query = GetParam();
  if (ray_query && !graphics->DeviceRayQuerySupport())
    GTEST_SKIP() << "native ray query unavailable";
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
    core->Render(&scene, &camera, &film,
                 ray_query ? sparkium::RENDER_PIPELINE_RAY_QUERY : sparkium::RENDER_PIPELINE_RT_FALLBACK);
    EXPECT_EQ(film.info.accumulated_samples, frame * 3);
    check(scene.settings.background_color);
  }
  scene.settings.background_color = glm::vec3(0.1f, 0.3f, 0.9f);
  film.Reset();
  EXPECT_EQ(film.info.accumulated_samples, 0);
  core->Render(&scene, &camera, &film,
               ray_query ? sparkium::RENDER_PIPELINE_RAY_QUERY : sparkium::RENDER_PIPELINE_RT_FALLBACK);
  EXPECT_EQ(film.info.accumulated_samples, 3);
  check(scene.settings.background_color);
  if (ray_query && (!graphics->DeviceRayTracingSupport() || graphics->API() == graphics::BACKEND_API_D3D12 ||
                    graphics->API() == graphics::BACKEND_API_VULKAN)) {
    EXPECT_EQ(core->ResolveRenderPipeline(sparkium::RENDER_PIPELINE_AUTO), sparkium::RENDER_PIPELINE_RAY_QUERY);
    graphics::FrameProfile profile(graphics.get(), false);
    profile.Begin();
    core->Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_AUTO);
    profile.Finish();
    EXPECT_EQ(profile.counters["native_ray_query"], 1u);
    // Auto must reuse the selected native pipeline and preserve its accumulation.
    EXPECT_EQ(film.info.accumulated_samples, 6);
    check(scene.settings.background_color);
    if (!graphics->DeviceRayTracingSupport()) {
      // Legacy JSON requests must select the same native query pipeline as Auto.
      EXPECT_EQ(core->ResolveRenderPipeline(sparkium::RENDER_PIPELINE_RAY_TRACING),
                sparkium::RENDER_PIPELINE_RAY_QUERY);
      profile.Begin();
      core->Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_RAY_TRACING);
      profile.Finish();
      EXPECT_EQ(profile.counters["native_ray_query"], 1u);
      EXPECT_EQ(film.info.accumulated_samples, 9);
      check(scene.settings.background_color);
    } else {
      EXPECT_EQ(core->ResolveRenderPipeline(sparkium::RENDER_PIPELINE_RAY_TRACING),
                sparkium::RENDER_PIPELINE_RAY_TRACING);
    }
  }
  if (graphics->DeviceRayQuerySupport()) {
    // Switching traversal implementations must discard accumulation in both directions.
    for (bool query : {!ray_query, ray_query}) {
      core->Render(&scene, &camera, &film,
                   query ? sparkium::RENDER_PIPELINE_RAY_QUERY : sparkium::RENDER_PIPELINE_RT_FALLBACK);
      EXPECT_EQ(film.info.accumulated_samples, 3);
      check(scene.settings.background_color);
    }
  }
}

TEST_P(ComputeTraversalTest, TransparentShadowLayers) {
  bool ray_query = GetParam();
  if (ray_query && !graphics->DeviceRayQuerySupport())
    GTEST_SKIP() << "native ray query unavailable";
  std::vector<Vector3<float>> positions{{-2, -2, 0}, {2, -2, 0}, {0, 2, 0}, {-2, -2, -1}, {2, -2, -1}, {0, 2, -1}};
  uint32_t indices[]{0, 1, 2, 3, 4, 5};
  Mesh<> mesh(6, 6, indices, positions.data());
  sparkium::GeometryMesh geometry(core.get(), mesh);
  sparkium::MaterialLambertian material(core.get());
  sparkium::raytracing::GeometryMesh rt_geometry(geometry);
  sparkium::raytracing::MaterialLambertian rt_material(material);
  sparkium::raytracing::SoftwarePipeline pipeline(sparkium::raytracing::DedicatedCast(core.get()), ray_query);
  pipeline.AddInstance(&rt_geometry, &rt_material, glm::mat4x3(1), 0);
  auto vfs = sparkium::raytracing::DedicatedCast(core.get())->GetShadersVFS();
  vfs.WriteFile("shadow_test.hlsl", R"(
#define SOFTWARE_EXTERNAL_BINDINGS
#include "common.hlsli"
#ifdef NATIVE_QUERY
RaytracingAccelerationStructure query_scene : register(t0, space0);
#else
ByteAddressBuffer software_nodes : register(t0, space0);
#endif
ByteAddressBuffer software_instances : register(t0, space1);
ByteAddressBuffer data_buffers[] : register(t0, space2);
RWByteAddressBuffer results : register(u0, space3);
#ifdef NATIVE_QUERY
#include "ray_query/traversal.hlsli"
#else
#include "software/traversal.hlsli"
#endif
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
  std::vector<std::string> args{"-I."};
  if (ray_query)
    args.push_back("-DNATIVE_QUERY");
  ASSERT_EQ(graphics->CreateShader(vfs, "shadow_test.hlsl", "Main", ray_query ? "cs_6_5" : "cs_6_0", args, &shader), 0);
  std::unique_ptr<graphics::ComputeProgram> program;
  graphics->CreateComputeProgram(shader.get(), &program);
  for (int i = 0; i < 3; ++i)
    program->AddResourceBinding(
        ray_query && i == 0 ? graphics::RESOURCE_TYPE_ACCELERATION_STRUCTURE : graphics::RESOURCE_TYPE_STORAGE_BUFFER,
        1);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  program->Finalize();
  std::unique_ptr<graphics::Buffer> output;
  graphics->CreateBuffer(16, graphics::BUFFER_TYPE_STATIC, &output);
  std::unique_ptr<graphics::CommandContext> commands;
  graphics->CreateCommandContext(&commands);
  pipeline.Update(commands.get(), {rt_geometry.Buffer()}, 1, 1);
  commands->CmdBindComputeProgram(program.get());
  if (ray_query)
    commands->CmdBindResources(0, pipeline.AccelerationStructure(), graphics::BIND_POINT_COMPUTE);
  else
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
  auto vfs = sparkium::raytracing::DedicatedCast(core.get())->GetShadersVFS();
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
    sparkium::CodeLines graph(vfs, "material/shader_graph/sampler.hlsl");
    graph.InsertAfter(sparkium::CodeLines(R"(
GraphSurface EvaluateShaderGraph(HitRecord hit, float3 direction, int bounce, uint ray_type,
                                 bool shadow, ByteAddressBuffer material) {
  GraphSurface surface = (GraphSurface)0;
  surface.normal = hit.normal;
  surface.opacity = 0.5f; surface.shadow_opacity = -1.0f;
  surface.ior = 1.45f; surface.roughness = 0.5f;
  return surface;
}
)"),
                      "// SHADER_GRAPH_IMPLEMENTATION");
    vfs.WriteFile("material_sampler.hlsli", graph);
    for (auto entry : {"RenderClosestHit", "ShadowClosestHit", "ShadowAnyHit"})
      compile("geometry/mesh/hit_group.hlsl", entry, "lib_6_5");
    for (auto entry : {"InitLeaves", "ReduceNodes", "MortonKeys", "BitonicSort", "SortLeaves"})
      compile("software/build.hlsl", entry, "cs_6_0");
  }
}

TEST_F(SoftwareBVHTest, RealtimePointLightsLeaveEmptyBackgroundUnchanged) {
  sparkium::Scene scene(core.get());
  scene.settings.background_color = glm::vec3(0.2f);
  sparkium::EntityPointLight light(core.get(), glm::vec3(0, 0, 1), glm::vec3(1), 100.0f);
  scene.AddEntity(&light);
  sparkium::Camera camera(core.get(), glm::lookAt(glm::vec3(0, 0, 4), glm::vec3(0), glm::vec3(0, 1, 0)),
                          glm::radians(45.0f), 17.0f / 13.0f);
  sparkium::Film film(core.get(), 17, 13);
  core->Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_REALTIME);
  std::vector<glm::vec4> pixels(17 * 13);
  film.GetRawImage()->DownloadData(pixels.data());
  for (const auto &pixel : pixels)
    for (int c = 0; c < 3; ++c)
      EXPECT_NEAR(pixel[c], 0.2f, 1e-6f);
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
