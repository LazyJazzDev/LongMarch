#include "../../demo/sparkium_gui/browser_renderer.h"

#include <gtest/gtest.h>

#include <chrono>
#include <fstream>
#include <thread>

namespace {
using namespace grassland::rhi;

class BrowserRendererTest : public testing::TestWithParam<BackendAPI> {
 protected:
  void SetUp() override {
    core_ = CreateCore(GetParam());
    ASSERT_TRUE(core_);
    device_ = core_->CreateDeviceByAutoSelection();
    ASSERT_TRUE(device_);
    path_ = std::filesystem::temp_directory_path() /
            ("sparkium-worker-" + std::to_string(reinterpret_cast<uintptr_t>(this)) + ".json");
    std::ofstream file(path_);
    file << R"({"format":"sparkium-scene","version":1,"name":"Worker test",
      "renderer":{"pipeline":"rt_fallback","samples_per_dispatch":2,"max_bounces":2,"background_color":[0.1,0.2,0.3]},
      "film":{"width":16,"height":16,"view_transform":"standard"},
      "camera":{"eye":[0,0,4],"target":[0,0,0]},"materials":{},"geometries":{},"entities":[]})";
    ASSERT_TRUE(file.good());
  }

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  template <typename Predicate>
  std::shared_ptr<const BrowserRenderer::Frame> WaitFrame(BrowserRenderer &renderer, Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
      auto state = renderer.GetState();
      if (!state.error.empty()) {
        ADD_FAILURE() << state.error;
        return {};
      }
      auto frame = renderer.Latest();
      if (frame && predicate(*frame))
        return frame;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ADD_FAILURE() << "Timed out waiting for worker output";
    return {};
  }

  std::vector<uint8_t> Read(BrowserRenderer &renderer, const std::shared_ptr<const BrowserRenderer::Frame> &frame) {
    ObjPtr<IQueue> queue = device_->GetQueue(QUEUE_TYPE_MAIN);
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    ObjPtr<IDownloadToken> download = commands->CmdDownloadImage(frame->image.Get());
    if (!download)
      return {};
    SubmitInfo waits;
    waits.waits.push_back(frame->rendered->GetEvent());
    ObjPtr<ISubmissionToken> submitted = commands->Submit(waits);
    if (!submitted)
      return {};
    renderer.Presented(frame, submitted);
    if (!submitted->WaitHost())
      return {};
    std::vector<uint8_t> bytes(download->GetSize());
    if (!download->CopyTo(bytes.data(), bytes.size()))
      return {};
    return bytes;
  }

  ObjPtr<ICore> core_;
  ObjPtr<IDevice> device_;
  std::filesystem::path path_;
};

TEST_P(BrowserRendererTest, HeldImagesStayImmutableAcrossConfigurationAndReload) {
  BrowserRenderer renderer(device_.Get());
  renderer.Load(path_, false);
  auto first = WaitFrame(renderer, [](const auto &) { return true; });
  ASSERT_TRUE(first);
  const auto original = Read(renderer, first);
  ASSERT_EQ(original.size(), 16u * 16u * 4u);
  auto settings = renderer.GetState().settings;
  settings.ray.background_color = {0.9f, 0.1f, 0.05f};
  renderer.Configure(settings, true);
  settings.ray.background_color = {0.8f, 0.15f, 0.02f};
  const auto configured = renderer.Configure(settings, true);
  auto second = WaitFrame(renderer, [&](const auto &frame) { return frame.revision == configured; });
  ASSERT_TRUE(second);
  EXPECT_NE(Read(renderer, second), original);
  EXPECT_EQ(Read(renderer, first), original);
  EXPECT_NE(first->image.Get(), second->image.Get());
  const auto revision = second->revision;
  second.reset();
  settings.hdr = true;
  renderer.Configure(settings, false);
  auto hdr = WaitFrame(renderer, [&](const auto &frame) { return frame.revision > revision && frame.hdr; });
  ASSERT_TRUE(hdr);
  EXPECT_EQ(hdr->image->GetDesc().format, IMAGE_FORMAT_R32G32B32A32_SFLOAT);
  EXPECT_EQ(Read(renderer, first), original);
  const auto hdr_revision = hdr->revision;
  hdr.reset();
  {
    std::ifstream input(path_);
    std::string source((std::istreambuf_iterator<char>(input)), {});
    input.close();
    source.replace(source.find("\"width\":16"), 10, "\"width\":24");
    source.replace(source.find("\"height\":16"), 11, "\"height\":20");
    std::ofstream output(path_);
    output << source;
  }
  renderer.Load(path_, false);
  auto reloaded = WaitFrame(renderer, [&](const auto &frame) { return frame.revision > hdr_revision; });
  ASSERT_TRUE(reloaded);
  EXPECT_FALSE(reloaded->hdr);
  EXPECT_EQ(reloaded->image->GetDesc().width, 24u);
  EXPECT_EQ(reloaded->image->GetDesc().height, 20u);
  const auto resized = Read(renderer, reloaded);
  ASSERT_EQ(resized.size(), 24u * 20u * 4u);
  for (size_t i = 0; i < resized.size(); ++i)
    EXPECT_EQ(resized[i], original[i % 4]);
  EXPECT_EQ(Read(renderer, first), original);
}

TEST_P(BrowserRendererTest, SplittingBatchesPreservesAccumulation) {
  {
    std::ofstream file(path_);
    file << R"({"format":"sparkium-scene","version":1,
      "renderer":{"pipeline":"rt_fallback","samples_per_dispatch":8,"max_bounces":3,"background_color":[0.2,0.4,0.8]},
      "film":{"width":32,"height":32,"persistence":0.97,"clamping":2,"max_exposure":2},
      "camera":{"eye":[0,0,4],"target":[0,0,0]},
      "materials":{"surface":{"type":"lambertian","base_color":[0.7,0.3,0.2]}},
      "geometries":{"sphere":{"type":"sphere","longitude_segments":12}},
      "entities":[{"type":"mesh","geometry":"sphere","material":"surface"}]})";
  }
  auto core = sparkium::Core::Create(device_.Get());
  ASSERT_TRUE(core);
  std::string error;
  auto scene = sparkium::JsonScene::Load(core.get(), path_, &error);
  ASSERT_TRUE(scene) << error;
  sparkium::Renderer renderer(core.get());
  auto render = [&](int samples, int batches) {
    scene->GetFilm()->Reset();
    scene->GetScene()->settings.samples_per_dispatch = samples;
    for (int batch = 0; batch < batches; ++batch) {
      auto submitted = renderer.Submit(scene->GetScene(), scene->GetCamera(), scene->GetFilm(),
                                       sparkium::RENDER_PIPELINE_RT_FALLBACK);
      if (!submitted)
        return std::vector<uint8_t>{};
    }
    ObjPtr<IQueue> queue = device_->GetQueue(QUEUE_TYPE_ASSIST);
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    ObjPtr<IDownloadToken> download = commands->CmdDownloadImage(scene->GetFilm()->GetRawImage());
    ObjPtr<ISubmissionToken> submitted = commands->Submit();
    if (!download || !submitted || !submitted->WaitHost())
      return std::vector<uint8_t>{};
    std::vector<uint8_t> bytes(download->GetSize());
    if (!download->CopyTo(bytes.data(), bytes.size()))
      return std::vector<uint8_t>{};
    return bytes;
  };
  const auto single = render(8, 1);
  ASSERT_EQ(single.size(), 32u * 32u * 4u * sizeof(float));
  EXPECT_EQ(render(2, 4), single);
  EXPECT_EQ(scene->GetFilm()->info.accumulated_samples, 8);
}

std::vector<BackendAPI> Backends() {
  std::vector<BackendAPI> backends;
#if defined(LONGMARCH_VULKAN_ENABLED)
  backends.push_back(BACKEND_API_VULKAN);
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
  backends.push_back(BACKEND_API_D3D12);
#endif
#if defined(LONGMARCH_METAL_ENABLED)
  backends.push_back(BACKEND_API_METAL);
#endif
  return backends;
}

INSTANTIATE_TEST_SUITE_P(Backends, BrowserRendererTest, testing::ValuesIn(Backends()));
}  // namespace
