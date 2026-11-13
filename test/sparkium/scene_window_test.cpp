#include <gtest/gtest.h>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <glm/gtc/packing.hpp>
#include <stdexcept>

#include "sparkium/sparkium.h"

namespace {
grassland::rhi::BackendAPI ParseBackend(const std::string &name) {
  using namespace grassland::rhi;
  BackendAPI api;
  if (name == "auto")
    api = BACKEND_API_DEFAULT;
  else if (name == "metal")
    api = BACKEND_API_METAL;
  else if (name == "vulkan")
    api = BACKEND_API_VULKAN;
  else if (name == "d3d12")
    api = BACKEND_API_D3D12;
  else
    throw std::invalid_argument("unknown RHI backend: " + name);
  return api;
}
}  // namespace

using namespace grassland;
using namespace grassland::rhi;

// The scene/Film invariants belong to Sparkium, independently of the legacy
// graphics swapchain-recovery and color-encoding tests.
TEST(SparkiumWindowResizeTest, ProgrammaticSquareToWideUpdatesPresentation) {
  if (!std::getenv("LONGMARCH_TEST_HDR") && !std::getenv("LONGMARCH_TEST_HDR_WINDOWS"))
    GTEST_SKIP() << "Requires an interactive desktop session";
  const char *backend = std::getenv("SPARKIUM_TEST_BACKEND");
  const auto api = backend ? ParseBackend(backend) : BACKEND_API_DEFAULT;
  RecordProperty("backend", BackendAPIName(api));
  ObjPtr<ICore> native = CreateCore(api, {true});
  ASSERT_TRUE(native);
  ObjPtr<IDevice> device = native->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ASSERT_TRUE(queue);
  auto core = sparkium::Core::Create(device.Get());
  ASSERT_TRUE(core);
  sparkium::Renderer renderer(core.get(), QUEUE_TYPE_MAIN);
  int sdr_frames = 0, hdr_frames = 0;
  for (bool hdr : {false, true}) {
    ObjPtr<IWindow> window = device->CreateGLFWWindow({1024, 1024, "Scene resize regression"});
    ASSERT_TRUE(window);
    auto output_space = COLOR_SPACE_SRGB_NONLINEAR;
    if (hdr) {
      auto options = window->GetSupportedPresentationOptions();
      auto supported = [&](ColorSpace space) {
        return std::any_of(options.begin(), options.end(),
                           [&](const auto &option) { return option.color_space == space; });
      };
      if (supported(COLOR_SPACE_EXTENDED_SRGB_LINEAR))
        output_space = COLOR_SPACE_EXTENDED_SRGB_LINEAR;
      else if (supported(COLOR_SPACE_HDR10_ST2084))
        output_space = COLOR_SPACE_HDR10_ST2084;
      else
        continue;
      ASSERT_TRUE(window->SetColorSpace(output_space));
    }
    ASSERT_TRUE(window->EnableImGui());
    for (const char *scene_name : {"cornell_box", "texture", "cornell_box"}) {
      std::string error;
      auto loaded = sparkium::JsonScene::Load(
          core.get(), FindAssetPath(std::string("scenes/") + scene_name + "/scene.json"), &error);
      ASSERT_TRUE(loaded) << error;
      auto *film = loaded->GetFilm();
      auto *camera = loaded->GetCamera();
      const auto view = camera->view;
      const auto aspect = camera->aspect;
      const glm::ivec2 size{film->GetWidth(), film->GetHeight()};
      loaded->GetScene()->settings.samples_per_dispatch = 1;
      loaded->GetScene()->settings.max_bounces = 1;
      auto source = film->AcquireOutput(hdr);
      ASSERT_TRUE(source);
      auto rendered =
          renderer.Submit(loaded->GetScene(), camera, film, sparkium::RENDER_PIPELINE_AUTO, source->Image(), hdr);
      ASSERT_TRUE(rendered);
      ASSERT_TRUE(rendered->WaitHost());
      const auto samples = film->info.accumulated_samples;
      ASSERT_GT(samples, 0);
      glfwSetWindowSize(window->GLFWWindow(), size.x, size.y);
      for (int frame = 0; frame < 6; ++frame) {
        if (frame == 2)
          glfwSetWindowSize(window->GLFWWindow(), size.x / 2, size.y / 2);
        if (frame == 4)
          glfwSetWindowSize(window->GLFWWindow(), size.x, size.y);
        glfwWaitEventsTimeout(0.02);
        if (window->IsSwapChainInvalid())
          ASSERT_TRUE(window->RebuildSwapChain());
        ASSERT_FALSE(window->IsSwapChainInvalid());
        int width{}, height{};
        glfwGetFramebufferSize(window->GLFWWindow(), &width, &height);
        SCOPED_TRACE(testing::Message() << "hdr=" << hdr << " framebuffer=" << width << "x" << height
                                        << " frame=" << frame);
        ObjPtr<IImage> target = window->GetNextImage();
        ASSERT_TRUE(target);
        ASSERT_EQ(target->GetDesc().width, width);
        ASSERT_EQ(target->GetDesc().height, height);
        ASSERT_TRUE(window->UpdateImGui([] {
          ImGui::GetIO().IniFilename = nullptr;
          ImGui::TextUnformatted("Scene resize regression");
        }));
        ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
        ASSERT_TRUE(commands);
        ASSERT_TRUE(commands->CmdClearImage(source->Image(), {0.5f, 0.5f, 0.5f, 1.0f}));
        ASSERT_TRUE(commands->CmdPresentImage(window.Get(), source->Image(),
                                              hdr ? COLOR_SPACE_EXTENDED_SRGB_LINEAR : COLOR_SPACE_SRGB_NONLINEAR));
        ObjPtr<IDownloadToken> readback;
        if (hdr) {
          readback = commands->CmdDownloadImage(target.Get());
          ASSERT_TRUE(readback);
        }
        SubmitInfo dependencies;
        dependencies.waits.push_back(rendered->GetEvent());
        ObjPtr<ISubmissionToken> done = commands->Submit(dependencies);
        ASSERT_TRUE(done);
        ASSERT_TRUE(done->WaitHost());
        EXPECT_EQ(loaded->GetFilm(), film);
        EXPECT_EQ(film->GetWidth(), size.x);
        EXPECT_EQ(film->GetHeight(), size.y);
        EXPECT_EQ(film->info.accumulated_samples, samples);
        EXPECT_EQ(camera->view, view);
        EXPECT_EQ(camera->aspect, aspect);
        EXPECT_EQ(source->Image()->GetDesc().width, size.x);
        EXPECT_EQ(source->Image()->GetDesc().height, size.y);
        hdr ? ++hdr_frames : ++sdr_frames;
        if (hdr) {
          // Both corners catch a stale square target, cropping and unwritten borders.
          const bool pq = output_space == COLOR_SPACE_HDR10_ST2084;
          const size_t pixel_size = pq ? 4 : 8;
          std::vector<uint8_t> pixels(size_t(width) * height * pixel_size);
          ASSERT_TRUE(readback->CopyTo(pixels.data(), pixels.size()));
          const double p = std::pow(0.5 * 203.0 / 10000.0, 2610.0 / 16384.0);
          const float expected =
              pq ? std::pow((3424.0 / 4096.0 + 2413.0 / 128.0 * p) / (1.0 + 2392.0 / 128.0 * p), 2523.0 / 32.0)
                 : 0.5f * window->GetHDRReferenceWhiteScale();
          for (int x : {0, width - 1}) {
            const size_t offset = (size_t(height - 1) * width + x) * pixel_size;
            for (int c = 0; c < 3; ++c) {
              float value;
              if (pq) {
                uint32_t packed;
                std::memcpy(&packed, pixels.data() + offset, sizeof(packed));
                value = float((packed >> (10 * c)) & 1023) / 1023.0f;
              } else {
                uint16_t half;
                std::memcpy(&half, pixels.data() + offset + c * 2, sizeof(half));
                value = glm::unpackHalf1x16(half);
              }
              EXPECT_NEAR(value, expected, 0.01f);
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(sdr_frames, 18);
  RecordProperty("sdr_frames", sdr_frames);
  RecordProperty("hdr_frames", hdr_frames);
}
