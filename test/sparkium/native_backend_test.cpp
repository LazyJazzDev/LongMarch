// Regression checks for the native CPU and CUDA backends.
//
// The native backends share one `LM_DEVICE_FUNC` shading core with each other
// and reproduce the GPU pipelines' data layouts, so the checks below pin down
// the parts that must stay identical rather than the Monte Carlo estimate
// itself: analytic backgrounds, sample accounting and reset semantics, film
// persistence, tone mapping, and image-level agreement with the compute
// reference on scenes exercising every material type.
#include <gtest/gtest.h>
#include <long_march.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <glm/gtc/matrix_transform.hpp>
#include <numeric>
#include <vector>

#include "sparkium/pipelines/native/native.h"

using namespace grassland;

namespace {

// A native render is compared against the compute software pipeline, which is
// the implementation the native backends were ported from and which runs on
// every device here.
constexpr sparkium::RenderPipeline kReferencePipeline = sparkium::RENDER_PIPELINE_RT_FALLBACK;

std::vector<sparkium::RenderPipeline> NativePipelines() {
  std::vector<sparkium::RenderPipeline> pipelines{sparkium::RENDER_PIPELINE_NATIVE_CPU};
  if (sparkium::native::CudaAvailable())
    pipelines.push_back(sparkium::RENDER_PIPELINE_NATIVE_CUDA);
  return pipelines;
}

const char *PipelineName(sparkium::RenderPipeline pipeline) {
  return pipeline == sparkium::RENDER_PIPELINE_NATIVE_CUDA ? "native_cuda" : "native_cpu";
}

class NativeBackendTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(graphics::CreateCore(graphics::BACKEND_API_DEFAULT, graphics::Core::Settings{2, false}, &graphics), 0);
    ASSERT_EQ(graphics->InitializeLogicalDeviceAutoSelect(false), 0);
    core = std::make_unique<sparkium::Core>(graphics.get());
  }

  std::vector<glm::vec4> Read(sparkium::Film &film) {
    std::vector<glm::vec4> pixels(static_cast<size_t>(film.GetWidth()) * film.GetHeight());
    film.GetRawImage()->DownloadData(pixels.data());
    return pixels;
  }

  std::unique_ptr<graphics::Core> graphics;
  std::unique_ptr<sparkium::Core> core;
};

TEST_F(NativeBackendTest, PipelineSelectionIsExplicit) {
  // `auto` must never pick a native backend; the native pipelines must never be
  // rewritten into a GPU one.
  EXPECT_NE(core->ResolveRenderPipeline(sparkium::RENDER_PIPELINE_AUTO), sparkium::RENDER_PIPELINE_NATIVE_CPU);
  EXPECT_NE(core->ResolveRenderPipeline(sparkium::RENDER_PIPELINE_AUTO), sparkium::RENDER_PIPELINE_NATIVE_CUDA);
  EXPECT_EQ(core->ResolveRenderPipeline(sparkium::RENDER_PIPELINE_NATIVE_CPU), sparkium::RENDER_PIPELINE_NATIVE_CPU);
  EXPECT_EQ(core->ResolveRenderPipeline(sparkium::RENDER_PIPELINE_NATIVE_CUDA), sparkium::RENDER_PIPELINE_NATIVE_CUDA);
}

TEST_F(NativeBackendTest, EmptySceneBackgroundAccumulationAndReset) {
  for (auto pipeline : NativePipelines()) {
    SCOPED_TRACE(PipelineName(pipeline));
    sparkium::Scene scene(core.get());
    scene.settings.samples_per_dispatch = 3;
    scene.settings.background_color = glm::vec3(0.2f, 0.4f, 0.7f);
    sparkium::Camera camera(core.get(), glm::mat4(1), glm::radians(45.0f), 17.0f / 13.0f);
    sparkium::Film film(core.get(), 17, 13);
    auto check = [&](glm::vec3 expected) {
      for (auto pixel : Read(film))
        for (int c = 0; c < 3; ++c) {
          ASSERT_TRUE(std::isfinite(pixel[c]));
          EXPECT_NEAR(pixel[c], expected[c], 1e-5f);
        }
    };
    for (int frame = 1; frame <= 2; ++frame) {
      core->Render(&scene, &camera, &film, pipeline);
      EXPECT_EQ(film.info.accumulated_samples, frame * 3);
      check(scene.settings.background_color);
    }
    scene.settings.background_color = glm::vec3(0.1f, 0.3f, 0.9f);
    film.Reset();
    EXPECT_EQ(film.info.accumulated_samples, 0);
    core->Render(&scene, &camera, &film, pipeline);
    EXPECT_EQ(film.info.accumulated_samples, 3);
    check(scene.settings.background_color);
  }
}

TEST_F(NativeBackendTest, FilmPersistenceAndExposureClampMatchTheReference) {
  // `persistence` and `max_exposure` are applied per sample inside the pixel
  // loop, and `film2img` truncates the sample weight through `int`, so the
  // developed value is not simply the background. Pin the whole chain against
  // the reference pipeline instead of against an analytic value.
  for (float persistence : {1.0f, 0.5f}) {
    for (float max_exposure : {1.0f, 0.4f}) {
      auto render = [&](sparkium::RenderPipeline pipeline) {
        sparkium::Scene scene(core.get());
        scene.settings.samples_per_dispatch = 4;
        scene.settings.background_color = glm::vec3(0.25f, 0.5f, 0.75f);
        sparkium::Camera camera(core.get(), glm::mat4(1), glm::radians(45.0f), 1.0f);
        sparkium::Film film(core.get(), 16, 16);
        film.info.persistence = persistence;
        film.info.max_exposure = max_exposure;
        for (int frame = 0; frame < 3; ++frame) core->Render(&scene, &camera, &film, pipeline);
        EXPECT_EQ(film.info.accumulated_samples, 12);
        return Read(film);
      };
      const auto reference = render(kReferencePipeline);
      for (auto pipeline : NativePipelines()) {
        SCOPED_TRACE(testing::Message() << PipelineName(pipeline) << ", persistence=" << persistence
                                        << ", max_exposure=" << max_exposure);
        const auto candidate = render(pipeline);
        ASSERT_EQ(candidate.size(), reference.size());
        // An empty scene has no Monte Carlo variance, so this is exact up to
        // float rounding.
        for (size_t i = 0; i < candidate.size(); ++i)
          for (int c = 0; c < 3; ++c) EXPECT_NEAR(candidate[i][c], reference[i][c], 1e-5f);
      }
    }
  }
}

TEST_F(NativeBackendTest, DevelopToHostMatchesToneMappedGpuDevelop) {
  // The host develop path must agree with `film2img.hlsl` + `tone_mapping.hlsl`
  // for every view transform.
  for (auto pipeline : NativePipelines()) {
    for (int view_transform = 0; view_transform < 3; ++view_transform) {
      SCOPED_TRACE(testing::Message() << PipelineName(pipeline) << ", view_transform=" << view_transform);
      sparkium::Scene scene(core.get());
      scene.settings.samples_per_dispatch = 2;
      // A background above 1.0 exercises the highlight branches of all three
      // view transforms.
      scene.settings.background_color = glm::vec3(0.1f, 0.8f, 2.5f);
      sparkium::Camera camera(core.get(), glm::mat4(1), glm::radians(45.0f), 1.0f);
      sparkium::Film film(core.get(), 24, 16);
      film.info.view_transform = view_transform;
      film.info.exposure = 0.5f;
      film.info.gamma = 1.2f;
      film.info.contrast = 1.1f;
      core->Render(&scene, &camera, &film, pipeline);

      std::vector<uint8_t> host;
      sparkium::native::DevelopToHost(&film, host);

      std::unique_ptr<graphics::Image> image;
      graphics->CreateImage(24, 16, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image);
      film.Develop(image.get());
      std::vector<uint8_t> device(host.size());
      image->DownloadData(device.data());

      ASSERT_EQ(host.size(), device.size());
      for (size_t i = 0; i < host.size(); ++i) {
        // Both paths quantize to 8 bits; one level of rounding slack.
        EXPECT_LE(std::abs(static_cast<int>(host[i]) - static_cast<int>(device[i])), 1)
            << "byte " << i << ": host " << int(host[i]) << " vs device " << int(device[i]);
      }
    }
  }
}

// Mirrors the JSON scenes: a diffuse box with an emissive quad, a point light,
// plus one entity per material type.
struct TestScene {
  explicit TestScene(sparkium::Core *core) {
    uint32_t quad[]{0, 1, 3, 1, 2, 3};
    std::vector<Vector3<float>> floor{{-3, 0, -3}, {3, 0, -3}, {3, 0, 3}, {-3, 0, 3}};
    std::vector<Vector3<float>> emitter{{-1, 3, -1}, {-1, 3, 1}, {1, 3, 1}, {1, 3, -1}};
    std::vector<Vector3<float>> wall{{-3, 0, -3}, {-3, 3, -3}, {3, 3, -3}, {3, 0, -3}};
    std::vector<Vector2<float>> uv{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    floor_mesh = Mesh<>(4, 6, quad, floor.data(), nullptr, uv.data());
    emitter_mesh = Mesh<>(4, 6, quad, emitter.data(), nullptr, uv.data());
    wall_mesh = Mesh<>(4, 6, quad, wall.data(), nullptr, uv.data());
    floor_geometry = std::make_unique<sparkium::GeometryMesh>(core, floor_mesh);
    emitter_geometry = std::make_unique<sparkium::GeometryMesh>(core, emitter_mesh);
    wall_geometry = std::make_unique<sparkium::GeometryMesh>(core, wall_mesh);

    lambertian = std::make_unique<sparkium::MaterialLambertian>(core, glm::vec3(0.7f, 0.6f, 0.5f));
    light = std::make_unique<sparkium::MaterialLight>(core, glm::vec3(4.0f));
    specular = std::make_unique<sparkium::MaterialSpecular>(core, glm::vec3(0.9f));
    principled = std::make_unique<sparkium::MaterialPrincipled>(core);
    principled->info.base_color = glm::vec3(0.4f, 0.5f, 0.8f);
    principled->info.roughness = 0.35f;
    principled->info.metallic = 0.25f;

    entities.push_back(std::make_unique<sparkium::EntityGeometryMaterial>(core, floor_geometry.get(),
                                                                         lambertian.get()));
    entities.push_back(
        std::make_unique<sparkium::EntityGeometryMaterial>(core, emitter_geometry.get(), light.get()));
    entities.push_back(std::make_unique<sparkium::EntityGeometryMaterial>(
        core, wall_geometry.get(), specular.get(),
        glm::mat4x3(glm::translate(glm::mat4(1), glm::vec3(0, 0, 0)))));
    entities.push_back(std::make_unique<sparkium::EntityGeometryMaterial>(
        core, wall_geometry.get(), principled.get(),
        glm::mat4x3(glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(0, 1, 0)))));
    point_light = std::make_unique<sparkium::EntityPointLight>(core, glm::vec3(2, 2, 2), glm::vec3(1), 30.0f);

    scene = std::make_unique<sparkium::Scene>(core);
    for (auto &entity : entities) scene->AddEntity(entity.get());
    scene->AddEntity(point_light.get());
    scene->settings.samples_per_dispatch = 16;
    scene->settings.max_bounces = 8;
    scene->settings.ambient_light = glm::vec3(0.05f);
    camera = std::make_unique<sparkium::Camera>(
        core, glm::lookAt(glm::vec3(0, 2, 6), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0)), glm::radians(50.0f), 1.0f);
  }

  Mesh<> floor_mesh, emitter_mesh, wall_mesh;
  std::unique_ptr<sparkium::GeometryMesh> floor_geometry, emitter_geometry, wall_geometry;
  std::unique_ptr<sparkium::MaterialLambertian> lambertian;
  std::unique_ptr<sparkium::MaterialLight> light;
  std::unique_ptr<sparkium::MaterialSpecular> specular;
  std::unique_ptr<sparkium::MaterialPrincipled> principled;
  std::vector<std::unique_ptr<sparkium::EntityGeometryMaterial>> entities;
  std::unique_ptr<sparkium::EntityPointLight> point_light;
  std::unique_ptr<sparkium::Scene> scene;
  std::unique_ptr<sparkium::Camera> camera;
};

// Averages an image over `block` x `block` tiles. Independent Monte Carlo
// noise shrinks like 1/block under this pooling while a systematic difference
// survives, which is what lets a tolerance separate the two.
std::vector<glm::vec3> Pool(const std::vector<glm::vec4> &pixels, int width, int height, int block) {
  std::vector<glm::vec3> pooled(static_cast<size_t>(width / block) * (height / block), glm::vec3(0));
  for (int y = 0; y < height - height % block; ++y)
    for (int x = 0; x < width - width % block; ++x)
      pooled[static_cast<size_t>(y / block) * (width / block) + x / block] +=
          glm::vec3(pixels[static_cast<size_t>(y) * width + x]);
  for (auto &value : pooled) value /= static_cast<float>(block * block);
  return pooled;
}

TEST_F(NativeBackendTest, AllMaterialTypesAgreeWithComputeReference) {
  constexpr int kResolution = 96;
  constexpr int kFrames = 8;
  constexpr int kBlock = 8;
  TestScene built(core.get());

  auto render = [&](sparkium::RenderPipeline pipeline) {
    sparkium::Film film(core.get(), kResolution, kResolution);
    for (int frame = 0; frame < kFrames; ++frame)
      core->Render(built.scene.get(), built.camera.get(), &film, pipeline);
    EXPECT_EQ(film.info.accumulated_samples, kFrames * built.scene->settings.samples_per_dispatch);
    return Read(film);
  };

  const auto reference = render(kReferencePipeline);
  const auto pooled_reference = Pool(reference, kResolution, kResolution, kBlock);
  double reference_mean = 0;
  for (const auto &value : pooled_reference) reference_mean += (value.x + value.y + value.z) / 3.0;
  reference_mean /= pooled_reference.size();
  ASSERT_GT(reference_mean, 0.01) << "reference render is black; the comparison would be vacuous";

  for (auto pipeline : NativePipelines()) {
    SCOPED_TRACE(PipelineName(pipeline));
    const auto candidate = render(pipeline);
    ASSERT_EQ(candidate.size(), reference.size());
    for (const auto &pixel : candidate)
      for (int c = 0; c < 3; ++c) ASSERT_TRUE(std::isfinite(pixel[c]));

    const auto pooled = Pool(candidate, kResolution, kResolution, kBlock);
    double error = 0, signed_error = 0;
    for (size_t i = 0; i < pooled.size(); ++i)
      for (int c = 0; c < 3; ++c) {
        error += std::abs(pooled[i][c] - pooled_reference[i][c]);
        signed_error += pooled[i][c] - pooled_reference[i][c];
      }
    error /= pooled.size() * 3;
    signed_error /= pooled.size() * 3;
    // The native backends run the same sample sequence but not bit-identical
    // transcendentals, so paths decorrelate and the residual is noise. It must
    // stay small relative to the image mean, and unbiased.
    EXPECT_LT(error, 0.12 * reference_mean) << "block-averaged error " << error << ", mean " << reference_mean;
    EXPECT_LT(std::abs(signed_error), 0.05 * reference_mean)
        << "signed error " << signed_error << ", mean " << reference_mean;
  }
}

TEST_F(NativeBackendTest, CpuAndCudaAgreeOnTheSameScene) {
  if (!sparkium::native::CudaAvailable())
    GTEST_SKIP() << "built without the CUDA backend";
  constexpr int kResolution = 96;
  constexpr int kFrames = 8;
  constexpr int kBlock = 8;
  TestScene built(core.get());

  auto render = [&](sparkium::RenderPipeline pipeline) {
    sparkium::Film film(core.get(), kResolution, kResolution);
    for (int frame = 0; frame < kFrames; ++frame)
      core->Render(built.scene.get(), built.camera.get(), &film, pipeline);
    return Read(film);
  };

  const auto cpu = render(sparkium::RENDER_PIPELINE_NATIVE_CPU);
  const auto cuda = render(sparkium::RENDER_PIPELINE_NATIVE_CUDA);
  const auto pooled_cpu = Pool(cpu, kResolution, kResolution, kBlock);
  const auto pooled_cuda = Pool(cuda, kResolution, kResolution, kBlock);
  double mean = 0, error = 0;
  for (size_t i = 0; i < pooled_cpu.size(); ++i)
    for (int c = 0; c < 3; ++c) {
      mean += pooled_cpu[i][c];
      error += std::abs(pooled_cpu[i][c] - pooled_cuda[i][c]);
    }
  mean /= pooled_cpu.size() * 3;
  error /= pooled_cpu.size() * 3;
  ASSERT_GT(mean, 0.01);
  EXPECT_LT(error, 0.12 * mean) << "block-averaged error " << error << ", mean " << mean;
}

TEST_F(NativeBackendTest, ShadowRaysAndEmitterVisibilityMatchTheReference) {
  // Same arrangement as the GPU pipelines' occluder test: a receiver facing the
  // camera, a point light off to one side, an area emitter, and a wall that can
  // be moved into the light path. Both light types must be shadowed by the same
  // amount on the native backends as on the compute reference.
  uint32_t quad[]{0, 1, 2, 0, 2, 3};
  std::vector<Vector3<float>> receiver{{-2, -2, 0}, {2, -2, 0}, {2, 2, 0}, {-2, 2, 0}};
  std::vector<Vector3<float>> wall{{0, -10, -10}, {0, 10, -10}, {0, 10, 10}, {0, -10, 10}};
  std::vector<Vector3<float>> emitter{{2.4f, -1, 1}, {2.4f, 1, 1}, {2.4f, 1, 3}, {2.4f, -1, 3}};
  Mesh<> receiver_mesh(4, 6, quad, receiver.data()), wall_mesh(4, 6, quad, wall.data()),
      emitter_mesh(4, 6, quad, emitter.data());
  sparkium::GeometryMesh receiver_geometry(core.get(), receiver_mesh), wall_geometry(core.get(), wall_mesh),
      emitter_geometry(core.get(), emitter_mesh);
  sparkium::MaterialLambertian diffuse(core.get(), glm::vec3(0.8f));
  // Two sided and not camera visible, so the frame measures the receiver's
  // illumination rather than the emitter's own radiance.
  sparkium::MaterialLight light(core.get(), glm::vec3(20.0f), true, false, false);
  sparkium::EntityGeometryMaterial receiver_entity(core.get(), &receiver_geometry, &diffuse);
  sparkium::EntityGeometryMaterial emitter_entity(core.get(), &emitter_geometry, &light);
  sparkium::EntityGeometryMaterial occluder_entity(core.get(), &wall_geometry, &diffuse,
                                                   glm::mat4x3(glm::translate(glm::mat4(1), glm::vec3(2.2f, 0, 0))));
  sparkium::EntityPointLight point_light(core.get(), glm::vec3(3, 0, 3), glm::vec3(1), 100.0f);
  sparkium::Scene scene(core.get());
  scene.AddEntity(&receiver_entity);
  scene.AddEntity(&emitter_entity);
  scene.AddEntity(&occluder_entity);
  scene.AddEntity(&point_light);
  scene.settings.samples_per_dispatch = 32;
  scene.settings.max_bounces = 3;
  sparkium::Camera camera(core.get(), glm::lookAt(glm::vec3(0, 0, 4), glm::vec3(0), glm::vec3(0, 1, 0)),
                          glm::radians(20.0f), 1.0f);

  auto average = [&](sparkium::RenderPipeline pipeline) {
    sparkium::Film film(core.get(), 64, 64);
    for (int frame = 0; frame < 4; ++frame) core->Render(&scene, &camera, &film, pipeline);
    double total = 0;
    for (const auto &pixel : Read(film)) {
      for (int c = 0; c < 3; ++c) EXPECT_TRUE(std::isfinite(pixel[c]));
      total += (pixel.x + pixel.y + pixel.z) / 3.0;
    }
    return total / (64.0 * 64.0);
  };

  scene.SetEntityActive(&occluder_entity, false);
  const double unshadowed = average(kReferencePipeline);
  scene.SetEntityActive(&occluder_entity, true);
  const double shadowed = average(kReferencePipeline);
  ASSERT_GT(unshadowed, 1e-3) << "the reference render is black; the check would be vacuous";
  ASSERT_LT(shadowed, unshadowed * 0.5) << "the reference render shows no shadow; the check would be vacuous";

  for (auto pipeline : NativePipelines()) {
    SCOPED_TRACE(PipelineName(pipeline));
    scene.SetEntityActive(&occluder_entity, false);
    const double native_unshadowed = average(pipeline);
    scene.SetEntityActive(&occluder_entity, true);
    const double native_shadowed = average(pipeline);
    EXPECT_NEAR(native_unshadowed, unshadowed, 0.05 * unshadowed);
    EXPECT_NEAR(native_shadowed, shadowed, 0.05 * unshadowed);
  }
}

TEST_F(NativeBackendTest, JsonScenesRenderOnEveryNativeBackend) {
  // Every shipped scene must load and produce a finite, non-black frame; this
  // is the coverage claim the report makes about material and geometry support.
  const auto scenes = sparkium::FindJsonScenes(FindAssetPath("scenes"));
  ASSERT_FALSE(scenes.empty());
  for (const auto &path : scenes) {
    std::string error;
    auto loaded = sparkium::JsonScene::Load(core.get(), path, &error);
    ASSERT_TRUE(loaded) << path.string() << ": " << error;
    // The shipped resolutions are too slow for a unit test; the scene's own
    // sampling settings are kept.
    sparkium::Film film(core.get(), 64, 64);
    auto *loaded_camera = loaded->GetCamera();
    sparkium::Camera camera(core.get(), loaded_camera->view, loaded_camera->fovy, 1.0f);
    camera.aperture_radius = loaded_camera->aperture_radius;
    camera.focus_distance = loaded_camera->focus_distance;
    camera.aperture_blades = loaded_camera->aperture_blades;
    camera.aperture_rotation = loaded_camera->aperture_rotation;
    camera.aperture_ratio = loaded_camera->aperture_ratio;
    for (auto pipeline : NativePipelines()) {
      SCOPED_TRACE(testing::Message() << path.string() << " on " << PipelineName(pipeline));
      film.Reset();
      core->Render(loaded->GetScene(), &camera, &film, pipeline);
      double total = 0;
      for (const auto &pixel : Read(film)) {
        for (int c = 0; c < 3; ++c) ASSERT_TRUE(std::isfinite(pixel[c]));
        total += (pixel.x + pixel.y + pixel.z) / 3.0;
      }
      EXPECT_GT(total / (64.0 * 64.0), 1e-4);
    }
  }
}

}  // namespace
