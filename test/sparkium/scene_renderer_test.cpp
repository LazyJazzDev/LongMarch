#include <gtest/gtest.h>

#include <future>

#include "scene_fixture.h"
#include "sparkium/renderer/renderer.h"
#include "sparkium/scene_io/json_scene.h"

using namespace sparkium;

TEST(SceneRenderer, OneLoadedSceneRendersAfterSourcesAreDeletedOnEveryBackend) {
  sparkium_test::SceneFiles files;
  auto scene = LoadScene(files.directory / "scene.json");
  files.RemoveSources();
  const std::vector<RendererSettings> settings{{RenderBackend::Graphics, GraphicsAPI::D3D12},
                                               {RenderBackend::Graphics, GraphicsAPI::Vulkan},
                                               {RenderBackend::Graphics, GraphicsAPI::Metal},
                                               {RenderBackend::CPU},
                                               {RenderBackend::CUDA}};
  for (const auto &config : settings) {
    if (!SupportRenderer(config))
      continue;
    SCOPED_TRACE(int(config.backend));
    auto renderer = CreateRenderer(config);
    renderer->SetScene(scene);
    EXPECT_EQ(renderer->GetScene().get(), scene.get());
    std::vector<RenderPipeline> pipelines{RENDER_PIPELINE_AUTO};
    if (config.backend == RenderBackend::CUDA)
      pipelines.push_back(RENDER_PIPELINE_RT_FALLBACK);
    for (auto pipeline : pipelines) {
      renderer->Configure({pipeline, 2});
      renderer->Render();
      auto image = renderer->ReadImage();
      EXPECT_EQ(image.width, 17);
      EXPECT_EQ(image.height, 13);
      EXPECT_EQ(image.accumulated_samples, 2);
      ASSERT_EQ(image.rgba.size(), 17 * 13 * 4);
      EXPECT_GT(image.rgba[(6 * 17 + 8) * 4], 180);  // emissive textured triangle
      auto linear = renderer->ReadLinearImage();
      ASSERT_EQ(linear.size(), 17 * 13);
      for (const auto &pixel : linear)
        for (int c = 0; c < 3; ++c)
          EXPECT_TRUE(std::isfinite(pixel[c]));
      renderer->Reset();
      renderer->Render();
      EXPECT_EQ(renderer->ReadImage().rgba, image.rgba);
    }
  }
}

TEST(SceneRenderer, ProgrammaticSceneCanBeSharedByConcurrentRenderers) {
  if (!SupportRenderer({RenderBackend::CPU}))
    GTEST_SKIP() << "CPU backend not built";
  auto scene = std::make_shared<SceneDefinition>();
  scene->film.width = 7;
  scene->film.height = 5;
  scene->film.view_transform = 1;
  scene->integrator.background_color = {0.1f, 0.2f, 0.3f};
  std::shared_ptr<const SceneDefinition> snapshot = scene;
  auto render = [snapshot](int samples) {
    auto renderer = CreateRenderer({RenderBackend::CPU});
    renderer->SetScene(snapshot);
    renderer->Configure({RENDER_PIPELINE_RT_FALLBACK, samples});
    renderer->Render();
    return renderer->ReadImage();
  };
  auto a = std::async(std::launch::async, render, 1);
  auto b = std::async(std::launch::async, render, 3);
  const auto first = a.get(), second = b.get();
  EXPECT_EQ(first.rgba, second.rgba);
  EXPECT_EQ(first.accumulated_samples, 1);
  EXPECT_EQ(second.accumulated_samples, 3);
  EXPECT_EQ(snapshot->integrator.samples_per_dispatch, 32);
}

TEST(SceneRenderer, OwnsSnapshotAndRejectsInvalidCalls) {
  if (!SupportRenderer({RenderBackend::CPU}))
    GTEST_SKIP() << "CPU backend not built";
  auto renderer = CreateRenderer({RenderBackend::CPU});
  EXPECT_THROW(renderer->Render(), std::logic_error);
  EXPECT_THROW(renderer->SetScene(nullptr), std::invalid_argument);
  auto scene = std::make_shared<SceneDefinition>();
  scene->film.width = scene->film.height = 4;
  std::weak_ptr<const SceneDefinition> weak = scene;
  renderer->SetScene(scene);
  scene.reset();
  EXPECT_FALSE(weak.expired());
  EXPECT_THROW(renderer->Configure({RENDER_PIPELINE_RT_FALLBACK, 0}), std::invalid_argument);
  renderer->BeginProfile(false);
  EXPECT_THROW(renderer->BeginProfile(false), std::logic_error);
  renderer->Render();
  EXPECT_EQ(renderer->ReadImage().rgba.size(), 4 * 4 * 4);
  const auto profile = renderer->EndProfile();
  EXPECT_GT(profile.cpu_ms.at("frame_wall"), 0);
  EXPECT_GE(profile.cpu_ms.at("frame_wall"), profile.cpu_ms.at("render_wall"));
  EXPECT_THROW(renderer->EndProfile(), std::logic_error);
  renderer.reset();
  EXPECT_TRUE(weak.expired());
}
