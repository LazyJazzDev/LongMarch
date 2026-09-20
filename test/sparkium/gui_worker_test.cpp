#include <gtest/gtest.h>

#include <chrono>
#include <fstream>
#include <random>

#include "../../demo/sparkium_gui/render_worker.h"

using namespace grassland;
using namespace sparkium_gui;

namespace {
class GuiSceneTest : public testing::Test {
 protected:
  void SetUp() override {
    path = std::filesystem::temp_directory_path() /
           ("sparkium-gui-test-" + std::to_string(std::random_device{}()) + ".json");
    WriteScene(17, 13);
  }

  void TearDown() override {
    if (!path.empty())
      std::filesystem::remove(path);
  }

  void WriteScene(int width, int height) {
    std::ofstream file(path);
    file << R"({"format":"sparkium-scene","version":1,"name":"GUI test",
      "film":{"width":)"
         << width << R"(,"height":)" << height << R"(,"view_transform":"standard"},
      "renderer":{"pipeline":"rt_fallback","samples_per_dispatch":2,"max_bounces":2,
                  "background_color":[0.1,0.2,0.3]},
      "camera":{"eye":[0,0,4],"target":[0,0,0]},"materials":{},"geometries":{},"entities":[]})";
  }

  RenderStatus Wait(RenderWorker &worker, uint64_t revision) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (std::chrono::steady_clock::now() < deadline) {
      auto status = worker.Snapshot();
      if (status.revision == revision && status.finished)
        return status;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ADD_FAILURE() << "renderer did not finish";
    return worker.Snapshot();
  }

  std::filesystem::path path;
};

class GuiWorkerTest : public GuiSceneTest, public testing::WithParamInterface<sparkium::RenderBackend> {
 protected:
  void SetUp() override {
    if (!sparkium::SupportBackend(GetParam()))
      GTEST_SKIP() << "render backend not compiled";
    GuiSceneTest::SetUp();
  }
};

TEST_F(GuiSceneTest, BackendSwitchRecreatesResourcesAndRecovers) {
  RenderWorker worker(1);
  RenderRequest request;
  request.scene = path;
  request.pipeline = sparkium::RENDER_PIPELINE_AUTO;
  request.samples = 1;
  std::shared_ptr<const RenderFrame> previous;
  // Switching away from CUDA must release resources in its original context;
  // switching back must acquire a valid context on the same worker thread.
  for (auto backend : {sparkium::RenderBackend::Graphics, sparkium::RenderBackend::CUDA, sparkium::RenderBackend::CPU,
                       sparkium::RenderBackend::CUDA, sparkium::RenderBackend::Graphics}) {
    if (!sparkium::SupportBackend(backend))
      continue;
    request.backend = backend;
    const auto revision = worker.Submit(request);
    auto pending = worker.Snapshot();
    if (pending.frame)
      EXPECT_EQ(pending.frame->revision, revision);
    const auto status = Wait(worker, revision);
    ASSERT_TRUE(status.error.empty()) << status.error;
    ASSERT_TRUE(status.frame);
    EXPECT_EQ(status.backend, backend);
    EXPECT_EQ(status.frame->revision, revision);
    EXPECT_EQ(status.frame->accumulated_samples, 1);
    EXPECT_EQ(status.frame->rgba.size(), 17 * 13 * 4);
    EXPECT_GT(status.frame->rgba[0], 0);
    if (previous)
      EXPECT_EQ(previous->rgba.size(), 17 * 13 * 4);
    previous = status.frame;
  }
  request.backend = static_cast<sparkium::RenderBackend>(99);
  auto status = Wait(worker, worker.Submit(request));
  EXPECT_FALSE(status.error.empty());
  EXPECT_FALSE(status.frame);
  request.backend = sparkium::RenderBackend::Graphics;
  status = Wait(worker, worker.Submit(request));
  ASSERT_TRUE(status.error.empty()) << status.error;
  ASSERT_TRUE(status.frame);
  EXPECT_EQ(status.frame->accumulated_samples, 1);
}

TEST_F(GuiSceneTest, GraphicsApiSwitchRecreatesRenderDevice) {
  RenderWorker worker(1);
  RenderRequest request;
  request.backend = sparkium::RenderBackend::Graphics;
  request.scene = path;
  request.pipeline = sparkium::RENDER_PIPELINE_RT_FALLBACK;
  for (auto api : {graphics::BACKEND_API_D3D12, graphics::BACKEND_API_VULKAN, graphics::BACKEND_API_METAL}) {
    if (!graphics::SupportBackendAPI(api))
      continue;
    request.graphics_api = api;
    auto revision = worker.Submit(request);
    auto status = Wait(worker, revision);
    ASSERT_TRUE(status.error.empty()) << status.error;
    ASSERT_NE(status.frame, nullptr);
    EXPECT_EQ(status.backend, sparkium::RenderBackend::Graphics);
    EXPECT_EQ(status.graphics_api, api);
    EXPECT_EQ(status.frame->revision, revision);
    EXPECT_EQ(status.frame->accumulated_samples, 2);
  }
}

TEST_F(GuiSceneTest, OptixPipelineSwitchResetsAccumulation) {
#ifndef SPARKIUM_OPTIX_ENABLED
  GTEST_SKIP() << "OptiX not built";
#endif
  RenderWorker worker(1);
  RenderRequest request;
  request.scene = path;
  request.backend = sparkium::RenderBackend::CUDA;
  request.samples = 3;
  std::vector<uint8_t> reference;
  for (auto pipeline :
       {sparkium::RENDER_PIPELINE_AUTO, sparkium::RENDER_PIPELINE_RT_FALLBACK, sparkium::RENDER_PIPELINE_RAY_TRACING}) {
    request.pipeline = pipeline;
    const auto status = Wait(worker, worker.Submit(request));
    ASSERT_TRUE(status.error.empty()) << status.error;
    ASSERT_TRUE(status.frame);
    EXPECT_TRUE(status.ray_tracing);
    EXPECT_EQ(status.resolved_pipeline,
              pipeline == sparkium::RENDER_PIPELINE_AUTO ? sparkium::RENDER_PIPELINE_RAY_TRACING : pipeline);
    EXPECT_EQ(status.frame->accumulated_samples, 3);
    if (reference.empty())
      reference = status.frame->rgba;
    else
      EXPECT_EQ(status.frame->rgba, reference);
  }
}

TEST_P(GuiWorkerTest, IndependentDisplayResetCoalescingAndRecovery) {
  RenderWorker worker(3);
  RenderRequest request;
  request.backend = GetParam();
  request.scene = path;
  auto revision = worker.Submit(request);
  auto status = Wait(worker, revision);
  ASSERT_TRUE(status.error.empty()) << status.error;
  ASSERT_TRUE(status.frame);
  EXPECT_EQ(status.frame->number, 3);
  EXPECT_EQ(status.frame->accumulated_samples, 6);
  EXPECT_EQ(status.frame->rgba.size(), 17 * 13 * 4);
  EXPECT_GT(status.frame->render_seconds, 0);
  const auto old_frame = status.frame;
  const auto old_pixels = old_frame->rgba;

  // Presentable pixels survive transfer into a completely separate device.
  std::unique_ptr<graphics::Core> display;
  ASSERT_EQ(graphics::CreateCore(graphics::BACKEND_API_DEFAULT, {}, &display), 0);
  ASSERT_EQ(display->InitializeLogicalDeviceAutoSelect(false), 0);
  std::unique_ptr<graphics::Image> image;
  ASSERT_EQ(display->CreateImage(17, 13, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image), 0);
  image->UploadData(old_pixels.data());
  std::vector<uint8_t> readback(old_pixels.size());
  image->DownloadData(readback.data());
  EXPECT_EQ(readback, old_pixels);
  EXPECT_GT(readback[0], 0);
  EXPECT_EQ(readback[3], 255);

  // Rapid pending UI edits must not expose an older revision's frame.
  for (int i = 0; i < 20; ++i) {
    request.samples = i + 1;
    revision = worker.Submit(request);
    const auto pending = worker.Snapshot();
    EXPECT_EQ(pending.revision, revision);
    if (pending.frame)
      EXPECT_EQ(pending.frame->revision, revision);
  }
  status = Wait(worker, revision);
  ASSERT_TRUE(status.error.empty()) << status.error;
  ASSERT_TRUE(status.frame);
  EXPECT_EQ(status.frame->revision, revision);
  EXPECT_EQ(status.frame->accumulated_samples, 20);
  EXPECT_EQ(old_frame->rgba, old_pixels);

  // A failed load reaches the frontend and is recoverable without restarting.
  request.scene = path.string() + ".missing";
  status = Wait(worker, worker.Submit(request));
  EXPECT_FALSE(status.error.empty());
  WriteScene(19, 11);
  request.scene = path;
  ++request.reload;
  request.samples = 1;
  status = Wait(worker, worker.Submit(request));
  ASSERT_TRUE(status.error.empty()) << status.error;
  ASSERT_TRUE(status.frame);
  EXPECT_EQ(status.frame->width, 19);
  EXPECT_EQ(status.frame->height, 11);
  EXPECT_EQ(status.frame->rgba.size(), 19 * 11 * 4);
  EXPECT_EQ(status.frame->accumulated_samples, 1);
  display->WaitGPU();
  worker.Stop();
}

INSTANTIATE_TEST_SUITE_P(Backends,
                         GuiWorkerTest,
                         testing::Values(sparkium::RenderBackend::Graphics,
                                         sparkium::RenderBackend::CPU,
                                         sparkium::RenderBackend::CUDA));
}  // namespace
