// Tests for the CPU path tracing backend.
//
// The backend renders on the host, so most of this runs with the host graphics
// backend and needs no device at all. Where a comparison against the GPU is
// meaningful it is added, and skipped when the device cannot run the compute
// fallback.
#include <gtest/gtest.h>
#include <long_march.h>

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <numeric>
#include <vector>

#include "grassland/graphics/backend/backend.h"
#include "sparkium/pipelines/raytracing/core/core.h"
#include "sparkium/pipelines/raytracing/entity/entities.h"
#include "sparkium/pipelines/raytracing/geometry/geometries.h"
#include "sparkium/pipelines/raytracing/material/materials.h"

using namespace grassland;

namespace {

// A small analytic scene: an emissive quad above a diffuse quad, which gives a
// gradient that a broken traversal or a missing light would not produce.
struct TestScene {
  std::unique_ptr<sparkium::Core> core;
  std::unique_ptr<sparkium::Scene> scene;
  std::unique_ptr<sparkium::Camera> camera;
  std::unique_ptr<sparkium::Film> film;
  std::vector<std::unique_ptr<sparkium::Entity>> entities;
  std::vector<std::unique_ptr<sparkium::Geometry>> geometries;
  std::vector<std::unique_ptr<sparkium::Material>> materials;
};

std::unique_ptr<TestScene> BuildScene(graphics::Core *graphics, int size) {
  auto test = std::make_unique<TestScene>();
  test->core = std::make_unique<sparkium::Core>(graphics);

  const std::vector<Vector3<float>> receiver_positions{{-2, 0, -2}, {2, 0, -2}, {2, 0, 2}, {-2, 0, 2}};
  const uint32_t receiver_indices[]{0, 1, 2, 0, 2, 3};
  const std::vector<Vector3<float>> emitter_positions{{-0.5f, 2, -0.5f}, {0.5f, 2, -0.5f}, {0.5f, 2, 0.5f},
                                                      {-0.5f, 2, 0.5f}};

  test->geometries.push_back(std::make_unique<sparkium::GeometryMesh>(
      test->core.get(), Mesh<float>(4, 6, receiver_indices, receiver_positions.data())));
  test->geometries.push_back(std::make_unique<sparkium::GeometryMesh>(
      test->core.get(), Mesh<float>(4, 6, receiver_indices, emitter_positions.data())));
  test->materials.push_back(std::make_unique<sparkium::MaterialLambertian>(test->core.get(), glm::vec3(0.7f)));
  test->materials.push_back(std::make_unique<sparkium::MaterialLight>(test->core.get(), glm::vec3(20.0f)));

  test->entities.push_back(std::make_unique<sparkium::EntityGeometryMaterial>(
      test->core.get(), test->geometries[0].get(), test->materials[0].get()));
  test->entities.push_back(std::make_unique<sparkium::EntityGeometryMaterial>(
      test->core.get(), test->geometries[1].get(), test->materials[1].get()));

  test->scene = std::make_unique<sparkium::Scene>(test->core.get());
  for (auto &entity : test->entities)
    test->scene->AddEntity(entity.get());
  test->scene->settings.samples_per_dispatch = 16;
  test->scene->settings.max_bounces = 4;
  test->camera = std::make_unique<sparkium::Camera>(
      test->core.get(), glm::lookAt(glm::vec3(0, 1.6f, 5), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0)),
      glm::radians(50.0f), 1.0f);
  test->film = std::make_unique<sparkium::Film>(test->core.get(), size, size);
  return test;
}

std::vector<glm::vec4> ReadFilm(sparkium::Film *film, int size) {
  std::vector<glm::vec4> pixels(static_cast<size_t>(size) * size);
  film->GetRawImage()->DownloadData(pixels.data());
  return pixels;
}

double MeanRadiance(const std::vector<glm::vec4> &pixels) {
  double total = 0.0;
  for (const glm::vec4 &pixel : pixels)
    total += std::max(0.0f, pixel.r) + std::max(0.0f, pixel.g) + std::max(0.0f, pixel.b);
  return total / (pixels.size() * 3.0);
}

// Normalized mean absolute error between two renders, the metric the
// verification scripts use on tone mapped output.
double MeanAbsoluteError(const std::vector<glm::vec4> &a, const std::vector<glm::vec4> &b) {
  double total = 0.0;
  double reference = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    for (int channel = 0; channel < 3; ++channel) {
      total += std::abs(a[i][channel] - b[i][channel]);
      reference += std::abs(a[i][channel]);
    }
  return reference > 0.0 ? total / reference : (total > 0.0 ? 1.0 : 0.0);
}

}  // namespace

// The point of the CPU backend: a full render with no graphics device behind it.
TEST(SparkiumCpuBackend, RendersWithoutAGraphicsDevice) {
  std::unique_ptr<graphics::Core> graphics;
  ASSERT_EQ(graphics::CreateCore(graphics::BACKEND_API_HOST, graphics::Core::Settings{2, false}, &graphics), 0);
  ASSERT_EQ(graphics->InitializeLogicalDeviceAutoSelect(false), 0);
  EXPECT_FALSE(graphics->DeviceRayTracingSupport());
  EXPECT_FALSE(graphics->DeviceRayQuerySupport());

  constexpr int kSize = 16;
  auto test = BuildScene(graphics.get(), kSize);
  test->core->Render(test->scene.get(), test->camera.get(), test->film.get(), sparkium::RENDER_PIPELINE_CPU);

  const std::vector<glm::vec4> pixels = ReadFilm(test->film.get(), kSize);
  EXPECT_GT(MeanRadiance(pixels), 0.0) << "the CPU backend rendered nothing";

  // A working render varies across the frame; a broken one tends to be flat.
  const auto [low, high] = std::minmax_element(pixels.begin(), pixels.end(),
                                               [](const glm::vec4 &a, const glm::vec4 &b) {
                                                 return a.r + a.g + a.b < b.r + b.g + b.b;
                                               });
  EXPECT_LT((low->r + low->g + low->b), (high->r + high->g + high->b));
}

// Accumulating over several frames must converge, not drift.
TEST(SparkiumCpuBackend, AccumulatesAcrossFrames) {
  std::unique_ptr<graphics::Core> graphics;
  ASSERT_EQ(graphics::CreateCore(graphics::BACKEND_API_HOST, graphics::Core::Settings{2, false}, &graphics), 0);
  ASSERT_EQ(graphics->InitializeLogicalDeviceAutoSelect(false), 0);

  constexpr int kSize = 8;
  auto test = BuildScene(graphics.get(), kSize);
  test->core->Render(test->scene.get(), test->camera.get(), test->film.get(), sparkium::RENDER_PIPELINE_CPU);
  const double once = MeanRadiance(ReadFilm(test->film.get(), kSize));

  for (int frame = 0; frame < 4; ++frame)
    test->core->Render(test->scene.get(), test->camera.get(), test->film.get(), sparkium::RENDER_PIPELINE_CPU);
  const double many = MeanRadiance(ReadFilm(test->film.get(), kSize));

  EXPECT_GT(once, 0.0);
  EXPECT_GT(many, 0.0);
  // More samples should not change the estimate by more than Monte Carlo noise.
  EXPECT_LT(std::abs(many - once) / std::max(once, 1e-6), 0.5);
}

// The host graphics backend has no compute shaders, so the film is tone mapped
// on the CPU. It has to produce an image rather than an untouched buffer.
TEST(SparkiumCpuBackend, DevelopsTheFilmWithoutAComputeDevice) {
  std::unique_ptr<graphics::Core> graphics;
  ASSERT_EQ(graphics::CreateCore(graphics::BACKEND_API_HOST, graphics::Core::Settings{2, false}, &graphics), 0);
  ASSERT_EQ(graphics->InitializeLogicalDeviceAutoSelect(false), 0);

  constexpr int kSize = 8;
  auto test = BuildScene(graphics.get(), kSize);
  test->core->Render(test->scene.get(), test->camera.get(), test->film.get(), sparkium::RENDER_PIPELINE_CPU);

  std::unique_ptr<graphics::Image> image;
  ASSERT_EQ(graphics->CreateImage(kSize, kSize, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image), 0);
  test->film->Develop(image.get());

  std::vector<uint8_t> pixels(static_cast<size_t>(kSize) * kSize * 4);
  image->DownloadData(pixels.data());
  EXPECT_GT(*std::max_element(pixels.begin(), pixels.end()), 0) << "the developed image is black";
}

// Where the device can run the compute fallback, the two backends must agree.
// They do not share an acceleration structure, so the comparison is on the
// converged image rather than on a single sample.
TEST(SparkiumCpuBackend, MatchesTheComputeFallback) {
  const char *backend = std::getenv("SPARKIUM_TEST_BACKEND");
  graphics::BackendAPI api = backend ? graphics::BACKEND_API_DEFAULT : graphics::BACKEND_API_DEFAULT;
  if (backend) {
    const std::string name = backend;
    api = name == "metal"    ? graphics::BACKEND_API_METAL
          : name == "vulkan" ? graphics::BACKEND_API_VULKAN
          : name == "d3d12"  ? graphics::BACKEND_API_D3D12
          : name == "host"   ? graphics::BACKEND_API_HOST
                             : graphics::BACKEND_API_DEFAULT;
  }
  if (api == graphics::BACKEND_API_HOST)
    GTEST_SKIP() << "the host backend can only run the CPU pipeline";

  std::unique_ptr<graphics::Core> graphics;
  ASSERT_EQ(graphics::CreateCore(api, graphics::Core::Settings{2, false}, &graphics), 0);
  ASSERT_EQ(graphics->InitializeLogicalDeviceAutoSelect(false), 0);

  constexpr int kSize = 32;
  constexpr int kFrames = 8;
  auto gpu = BuildScene(graphics.get(), kSize);
  auto cpu = BuildScene(graphics.get(), kSize);
  for (int frame = 0; frame < kFrames; ++frame) {
    gpu->core->Render(gpu->scene.get(), gpu->camera.get(), gpu->film.get(), sparkium::RENDER_PIPELINE_RT_FALLBACK);
    cpu->core->Render(cpu->scene.get(), cpu->camera.get(), cpu->film.get(), sparkium::RENDER_PIPELINE_CPU);
  }

  const std::vector<glm::vec4> reference = ReadFilm(gpu->film.get(), kSize);
  const std::vector<glm::vec4> actual = ReadFilm(cpu->film.get(), kSize);
  EXPECT_GT(MeanRadiance(reference), 0.0);
  EXPECT_LT(MeanAbsoluteError(reference, actual), 0.1)
      << "the CPU backend and the compute fallback disagree beyond sampling noise";
}
