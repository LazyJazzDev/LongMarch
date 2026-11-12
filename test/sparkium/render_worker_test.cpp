#include "sparkium/core/render_worker.h"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

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
  Worker worker([&] { return ThinLensScene::Create(device.Get(), 32, 24); }, settings, 8);
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
  const auto blocked_revision = worker.Configure(settings, 16);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline && worker.GetState().completed_samples != 12)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ASSERT_TRUE(worker.GetState().error.empty()) << worker.GetState().error;
  EXPECT_EQ(worker.GetState().revision, blocked_revision);
  EXPECT_EQ(worker.GetState().completed_samples, 12);
  // All output slots are held: accumulation progresses, but the final batch
  // waits for a reusable image instead of publishing an incomplete result.
  EXPECT_EQ(worker.Latest(), third);
  EXPECT_EQ(read(first->Image(), first->rendered.Get()), original);
  first.reset();
  auto final = wait(blocked_revision, 16);
  ASSERT_TRUE(final);
  EXPECT_EQ(read(final->Image(), final->rendered.Get()), reference(false, 16));
  worker.Stop();
  EXPECT_EQ(read(third->Image(), third->rendered.Get()), third_pixels);
  EXPECT_EQ(read(hdr->Image(), hdr->rendered.Get()), hdr_pixels);
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
