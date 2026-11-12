#include "sparkium/core/render_worker.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "../../demo/franka_kinematics/franka_scene.h"
#include "../../demo/practium_hello/practium_scene.h"
#include "../../demo/thin_lens/thin_lens_scene.h"

namespace {
using namespace grassland::rhi;
using Worker = sparkium::RenderWorker<ThinLensScene>;

class RenderWorkerTest : public testing::TestWithParam<BackendAPI> {};

TEST_P(RenderWorkerTest, SnapshotsAndHeldOutputsMatchSynchronousRendering) {
  ObjPtr<ICore> native = CreateCore(GetParam(), {true});
  ASSERT_TRUE(native);
  ObjPtr<IDevice> device = native->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ASSERT_TRUE(queue);
  auto direct = ThinLensScene::Create(device.Get(), 32, 24);
  ASSERT_TRUE(direct);
  ThinLensScene::Settings settings;
  std::atomic<Worker *> callback_worker{};
  std::atomic<uint32_t> notifications{};
  Worker worker([&] { return ThinLensScene::Create(device.Get(), 32, 24); }, settings, 8, false, 4,
                [&] {
                  if (auto *observer = callback_worker.load()) {
                    // Frontends may query published state while processing a wakeup.
                    // Calling this under the worker mutex would deadlock.
                    (void)observer->GetState();
                    ++notifications;
                  }
                });
  callback_worker.store(&worker);
  auto wait = [&](uint64_t revision, int samples = 8) -> std::shared_ptr<const Worker::Frame> {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
      if (!worker.GetState().error.empty()) {
        ADD_FAILURE() << worker.GetState().error;
        return {};
      }
      auto frame = worker.Latest();
      if (frame && frame->revision == revision && frame->samples == samples)
        return frame;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ADD_FAILURE() << "Timed out waiting for worker output";
    return {};
  };
  auto read = [&](IImage *image, ISubmissionToken *rendered) {
    std::vector<uint8_t> result;
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    if (!commands)
      return result;
    ObjPtr<IDownloadToken> download = commands->CmdDownloadImage(image);
    SubmitInfo dependencies;
    dependencies.waits.push_back(rendered->GetEvent());
    ObjPtr<ISubmissionToken> done = commands->Submit(dependencies);
    if (!download || !done || !done->WaitHost())
      return result;
    result.resize(download->GetSize());
    if (!download->CopyTo(result.data(), result.size()))
      result.clear();
    return result;
  };
  auto reference = [&](bool hdr, int samples = 8) {
    EXPECT_TRUE(direct->Configure(settings));
    auto image = direct->GetFilm()->AcquireOutput(hdr);
    EXPECT_TRUE(image);
    if (!image)
      return std::vector<uint8_t>{};
    auto submitted = direct->Submit(samples, image->Image(), hdr, {});
    EXPECT_TRUE(submitted);
    return submitted ? read(image->Image(), submitted.Get()) : std::vector<uint8_t>{};
  };
  auto first = wait(1);
  ASSERT_TRUE(first);
  const auto original = read(first->Image(), first->rendered.Get());
  ASSERT_EQ(original.size(), 32u * 24u * 4u);
  EXPECT_EQ(original, reference(false));
  settings.focus_distance = 2.5f;
  settings.aperture_radius = 0.45f;
  worker.Configure(settings, 8);
  settings.pinhole_mode = true;
  settings.geometry_offset = 0;
  const auto revision = worker.Configure(settings, 8);
  auto second = wait(revision);
  ASSERT_TRUE(second);
  const auto changed = read(second->Image(), second->rendered.Get());
  EXPECT_EQ(changed, reference(false));
  EXPECT_NE(changed, original);
  EXPECT_EQ(read(first->Image(), first->rendered.Get()), original);
  second.reset();
  const auto hdr_revision = worker.Configure(settings, 8, true);
  auto hdr = wait(hdr_revision);
  ASSERT_TRUE(hdr);
  EXPECT_EQ(hdr->Image()->GetDesc().format, IMAGE_FORMAT_R32G32B32A32_SFLOAT);
  const auto hdr_pixels = read(hdr->Image(), hdr->rendered.Get());
  ASSERT_EQ(hdr_pixels.size(), 32u * 24u * 4u * sizeof(float));
  EXPECT_EQ(hdr_pixels, reference(true));
  auto third = wait(worker.Configure(settings, 4), 4);
  ASSERT_TRUE(third);
  const auto third_pixels = read(third->Image(), third->rendered.Get());
  auto fourth = wait(worker.Configure(settings, 4), 4);
  ASSERT_TRUE(fourth);
  const auto fourth_pixels = read(fourth->Image(), fourth->rendered.Get());
  const auto blocked_revision = worker.Configure(settings, 16);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline && worker.GetState().completed_samples != 12)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ASSERT_TRUE(worker.GetState().error.empty()) << worker.GetState().error;
  EXPECT_EQ(worker.GetState().revision, blocked_revision);
  EXPECT_EQ(worker.GetState().completed_samples, 12);
  // All output slots are held: accumulation progresses, but the final batch
  // waits for a reusable image instead of publishing an incomplete result.
  EXPECT_EQ(worker.Latest(), fourth);
  EXPECT_EQ(read(first->Image(), first->rendered.Get()), original);
  first.reset();
  auto final = wait(blocked_revision, 16);
  ASSERT_TRUE(final);
  EXPECT_EQ(read(final->Image(), final->rendered.Get()), reference(false, 16));
  worker.Stop();
  EXPECT_GT(notifications.load(), 0u);
  EXPECT_EQ(read(third->Image(), third->rendered.Get()), third_pixels);
  EXPECT_EQ(read(fourth->Image(), fourth->rendered.Get()), fourth_pixels);
  EXPECT_EQ(read(hdr->Image(), hdr->rendered.Get()), hdr_pixels);
}

// Temporal clients keep Film history when accepting a new scheduling budget.
struct ContinuingScene {
  struct Settings {};

  std::unique_ptr<ThinLensScene> scene;

  bool Configure(const Settings &) {
    return true;
  }

  sparkium::Film *GetFilm() {
    return scene->GetFilm();
  }

  ObjPtr<ISubmissionToken> Submit(int samples, IImage *image, bool hdr, const SubmitInfo &dependencies) {
    return scene->Submit(samples, image, hdr, dependencies);
  }
};

TEST_P(RenderWorkerTest, FiniteBudgetsPreserveExistingAccumulation) {
  ObjPtr<ICore> native = CreateCore(GetParam(), {true});
  ASSERT_TRUE(native);
  ObjPtr<IDevice> device = native->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  using ContinuingWorker = sparkium::RenderWorker<ContinuingScene>;
  ContinuingWorker worker(
      [&]() -> std::unique_ptr<ContinuingScene> {
        auto scene = ThinLensScene::Create(device.Get(), 32, 24);
        if (!scene || !scene->Configure({}))
          return {};
        scene->GetFilm()->info.persistence = 0.98f;
        return std::make_unique<ContinuingScene>(ContinuingScene{std::move(scene)});
      },
      {}, 8, true);
  auto wait = [&](uint64_t revision) {
    std::shared_ptr<const ContinuingWorker::Frame> frame;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline && worker.GetState().error.empty()) {
      frame = worker.Latest();
      if (frame && frame->revision == revision && frame->samples == 8)
        return frame;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return std::shared_ptr<const ContinuingWorker::Frame>{};
  };
  ASSERT_TRUE(wait(1));
  auto final = wait(worker.Configure({}, 8, true));
  ASSERT_TRUE(final) << worker.GetState().error;
  worker.Stop();
  auto direct = ThinLensScene::Create(device.Get(), 32, 24);
  ASSERT_TRUE(direct);
  ASSERT_TRUE(direct->Configure({}));
  direct->GetFilm()->info.persistence = 0.98f;
  auto image = direct->GetFilm()->AcquireOutput(true);
  ASSERT_TRUE(image);
  auto submitted = direct->Submit(16, image->Image(), true, {});
  ASSERT_TRUE(submitted);
  ASSERT_TRUE(submitted->WaitHost());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ASSERT_TRUE(queue);
  auto read = [&](IImage *source) {
    std::vector<uint8_t> pixels;
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    if (!commands)
      return pixels;
    ObjPtr<IDownloadToken> download = commands->CmdDownloadImage(source);
    ObjPtr<ISubmissionToken> completed = commands->Submit();
    if (!download || !completed || !completed->WaitHost())
      return pixels;
    pixels.resize(download->GetSize());
    if (!download->CopyTo(pixels.data(), pixels.size()))
      pixels.clear();
    return pixels;
  };
  const auto expected = read(image->Image());
  ASSERT_EQ(expected.size(), 32u * 24u * 4u * sizeof(float));
  EXPECT_EQ(read(final->Image()), expected);
}

TEST_P(RenderWorkerTest, FrankaSnapshotsPreserveJointHistoryAndPipelineSwitches) {
  ObjPtr<ICore> native = CreateCore(GetParam(), {true});
  ASSERT_TRUE(native);
  ObjPtr<IDevice> device = native->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ASSERT_TRUE(queue);
  auto direct = FrankaScene::Create(device.Get(), 48, 24);
  ASSERT_TRUE(direct);
  FrankaScene::Settings settings;
  using FrankaWorker = sparkium::RenderWorker<FrankaScene>;
  FrankaWorker worker([&] { return FrankaScene::Create(device.Get(), 48, 24); }, settings, 8, true);
  auto wait = [&](uint64_t revision, int samples) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline && worker.GetState().error.empty()) {
      auto frame = worker.Latest();
      if (frame && frame->revision == revision && frame->samples == samples)
        return frame;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return std::shared_ptr<const FrankaWorker::Frame>{};
  };
  auto read = [&](IImage *source, ISubmissionToken *rendered) {
    std::vector<uint8_t> pixels;
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    if (!commands)
      return pixels;
    ObjPtr<IDownloadToken> download = commands->CmdDownloadImage(source);
    SubmitInfo dependencies;
    dependencies.waits.push_back(rendered->GetEvent());
    ObjPtr<ISubmissionToken> done = commands->Submit(dependencies);
    if (!download || !done || !done->WaitHost())
      return pixels;
    pixels.resize(download->GetSize());
    if (!download->CopyTo(pixels.data(), pixels.size()))
      pixels.clear();
    return pixels;
  };
  auto reference = [&](int samples) {
    EXPECT_TRUE(direct->Configure(settings));
    auto output = direct->GetFilm()->AcquireOutput(true);
    if (!output)
      return std::vector<uint8_t>{};
    auto submitted = direct->Submit(samples, output->Image(), true, {});
    return submitted ? read(output->Image(), submitted.Get()) : std::vector<uint8_t>{};
  };
  auto first = wait(1, 8);
  ASSERT_TRUE(first) << worker.GetState().error;
  auto original = read(first->Image(), first->rendered.Get());
  ASSERT_EQ(original.size(), 48u * 24u * 4u * sizeof(float));
  EXPECT_EQ(original, reference(8));
  for (size_t i = 0; i < settings.joints.size(); ++i)
    settings.joints[i] = (FrankaScene::joint_lower[i] + FrankaScene::joint_upper[i]) * 0.5f;
  auto moved = wait(worker.Configure(settings, 8, true), 8);
  ASSERT_TRUE(moved) << worker.GetState().error;
  auto moved_pixels = read(moved->Image(), moved->rendered.Get());
  EXPECT_EQ(moved_pixels, reference(8));
  EXPECT_NE(moved_pixels, original);
  EXPECT_EQ(read(first->Image(), first->rendered.Get()), original);
  moved.reset();
  settings.ray_tracing = false;
  auto raster = wait(worker.Configure(settings, 1, true), 1);
  ASSERT_TRUE(raster) << worker.GetState().error;
  EXPECT_EQ(read(raster->Image(), raster->rendered.Get()), reference(1));
  raster.reset();
  settings.ray_tracing = true;
  auto resumed = wait(worker.Configure(settings, 8, true), 8);
  ASSERT_TRUE(resumed) << worker.GetState().error;
  EXPECT_EQ(read(resumed->Image(), resumed->rendered.Get()), reference(8));
  worker.Stop();
  EXPECT_EQ(read(first->Image(), first->rendered.Get()), original);
}

TEST_P(RenderWorkerTest, PractiumFirstTracingSubmissionPreservesRevisionProgress) {
  ObjPtr<ICore> native = CreateCore(GetParam(), {true});
  ASSERT_TRUE(native);
  ObjPtr<IDevice> device = native->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  PractiumDemoScene::Settings settings;
  settings.ray_tracing = false;
  using PhysicsWorker = sparkium::RenderWorker<PractiumDemoScene>;
  PhysicsWorker worker([&] { return PractiumDemoScene::Create(device.Get(), 48, 36); }, settings, 64, false, 1);
  auto raster = worker.WaitForFrame(1, 64);
  ASSERT_TRUE(raster) << worker.GetState().error;
  EXPECT_EQ(raster->samples, 64);
  settings.ray_tracing = true;
  const auto revision = worker.Configure(settings, 32, false, 8);
  auto traced = worker.WaitForFrame(revision, 32);
  ASSERT_TRUE(traced) << worker.GetState().error;
  EXPECT_EQ(traced->revision, revision);
  EXPECT_EQ(traced->samples, 32);
  EXPECT_TRUE(traced->rendered->WaitHost());
  worker.Stop();
}

TEST_P(RenderWorkerTest, PractiumBatchesPreservePhysicsStepsAndResetSequence) {
  ObjPtr<ICore> native = CreateCore(GetParam(), {true});
  ASSERT_TRUE(native);
  ObjPtr<IDevice> device = native->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ASSERT_TRUE(queue);
  auto direct = PractiumDemoScene::Create(device.Get(), 48, 36);
  ASSERT_TRUE(direct);
  PractiumDemoScene::Settings settings;
  settings.pause = false;
  settings.ray_tracing = true;
  using PhysicsWorker = sparkium::RenderWorker<PractiumDemoScene>;
  PhysicsWorker worker([&] { return PractiumDemoScene::Create(device.Get(), 48, 36); }, settings, 96, true, 8);
  auto wait = [&](uint64_t revision, int samples) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline && worker.GetState().error.empty()) {
      auto frame = worker.Latest();
      if (frame && frame->revision == revision && frame->samples == samples)
        return frame;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return std::shared_ptr<const PhysicsWorker::Frame>{};
  };
  auto read = [&](IImage *source, ISubmissionToken *rendered) {
    std::vector<uint8_t> pixels;
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    if (!commands)
      return pixels;
    ObjPtr<IDownloadToken> download = commands->CmdDownloadImage(source);
    SubmitInfo dependencies;
    dependencies.waits.push_back(rendered->GetEvent());
    ObjPtr<ISubmissionToken> done = commands->Submit(dependencies);
    if (!download || !done || !done->WaitHost())
      return pixels;
    pixels.resize(download->GetSize());
    if (!download->CopyTo(pixels.data(), pixels.size()))
      pixels.clear();
    return pixels;
  };
  auto reference = [&](int cycles) {
    EXPECT_TRUE(direct->Configure(settings));
    auto output = direct->GetFilm()->AcquireOutput(true);
    if (!output)
      return std::vector<uint8_t>{};
    ObjPtr<ISubmissionToken> submitted;
    for (int i = 0; i < cycles; ++i) {
      submitted = direct->Submit(settings.ray_tracing ? 32 : 1, output->Image(), true, {});
      if (!submitted)
        return std::vector<uint8_t>{};
    }
    return read(output->Image(), submitted.Get());
  };
  auto moving = wait(1, 96);
  ASSERT_TRUE(moving) << worker.GetState().error;
  const auto moving_pixels = read(moving->Image(), moving->rendered.Get());
  ASSERT_EQ(moving_pixels.size(), 48u * 36u * 4u * sizeof(float));
  EXPECT_EQ(moving_pixels, reference(3));
  moving.reset();
  settings.pause = true;
  auto paused = wait(worker.Configure(settings, 64, true, 32), 64);
  ASSERT_TRUE(paused) << worker.GetState().error;
  EXPECT_EQ(read(paused->Image(), paused->rendered.Get()), reference(2));
  paused.reset();
  settings.day_light = false;
  settings.ray_tracing = false;
  settings.reset_sequence = 1;
  ASSERT_TRUE(direct->Configure(settings));
  settings.reset_sequence = 2;
  auto reset = wait(worker.Configure(settings, 3, true, 1), 3);
  ASSERT_TRUE(reset) << worker.GetState().error;
  const auto reset_pixels = read(reset->Image(), reset->rendered.Get());
  EXPECT_EQ(reset_pixels, reference(3));
  EXPECT_NE(reset_pixels, moving_pixels);
  reset.reset();
  settings.pause = false;
  settings.day_light = true;
  settings.ray_tracing = true;
  auto resumed = wait(worker.Configure(settings, 32, true, 8), 32);
  ASSERT_TRUE(resumed) << worker.GetState().error;
  EXPECT_EQ(read(resumed->Image(), resumed->rendered.Get()), reference(1));
  worker.Stop();
}

std::vector<BackendAPI> Backends() {
  std::vector<BackendAPI> result;
#if defined(LONGMARCH_VULKAN_ENABLED)
  result.push_back(BACKEND_API_VULKAN);
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
  result.push_back(BACKEND_API_D3D12);
#endif
#if defined(LONGMARCH_METAL_ENABLED)
  result.push_back(BACKEND_API_METAL);
#endif
  return result;
}

INSTANTIATE_TEST_SUITE_P(Backends, RenderWorkerTest, testing::ValuesIn(Backends()));
}  // namespace
