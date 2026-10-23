#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <thread>

#include "grassland/rhi/backend/metal/metal_core.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_diagnostics.h"
#include "grassland/rhi/backend/metal/metal_image.h"
#include "grassland/rhi/rhix/rhix.h"

namespace grassland::rhix {
using namespace rhi;
namespace backend = rhi::backend;

namespace {

TEST(RhiMetal, NativeCompilationErrorsReachConsole) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto core = rhix::CreateCore(BACKEND_API_METAL, {true});
  ASSERT_TRUE(core);
  auto device = core.CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto native = static_cast<backend::metal::MetalDevice *>(device.Get());
  NS::Error *error = nullptr;
  auto library = NS::TransferPtr(native->NativeDevice()->newLibrary(
      NS::String::string("#error RHI_Metal_console_probe", NS::UTF8StringEncoding), nullptr, &error));
  ASSERT_FALSE(library);
  ASSERT_NE(error, nullptr);
  testing::internal::CaptureStdout();
  backend::metal::ReportMetalError(error, "Metal library compilation failed", true);
  backend::metal::ReportMetalError(nullptr, "Metal idle marker failed");
  auto output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("[ERROR]"), std::string::npos);
  EXPECT_NE(output.find("RHI_Metal_console_probe"), std::string::npos);
  EXPECT_NE(output.find(error->domain()->utf8String()), std::string::npos);
  EXPECT_NE(output.find("Metal diagnostic information:"), std::string::npos);
  EXPECT_NE(output.find("Metal idle marker failed"), std::string::npos);
}

TEST(RhiMetal, CoreDebugOptionsAcrossSequentialCores) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  for (bool enabled : {CoreDesc{}.enable_debug, false, true, false}) {
    auto core = rhix::CreateCore(BACKEND_API_METAL, {enabled});
    ASSERT_TRUE(core);
    auto device = core.CreateDeviceByAutoSelection();
    ASSERT_TRUE(device);
    auto native = static_cast<backend::metal::MetalDevice *>(device.Get());
    auto commands = native->CreateCommandBuffer();
    ASSERT_TRUE(commands);
    EXPECT_EQ(commands->errorOptions(),
              enabled ? MTL::CommandBufferErrorOptionEncoderExecutionStatus : MTL::CommandBufferErrorOptionNone);
    EXPECT_TRUE(commands->retainedReferences());
    auto buffer = NS::TransferPtr(native->NativeDevice()->newBuffer(256, MTL::ResourceStorageModeShared));
    ASSERT_TRUE(buffer);
    auto blit = commands->blitCommandEncoder();
    ASSERT_NE(blit, nullptr);
    blit->fillBuffer(buffer.get(), NS::Range::Make(0, 256), 0x5a);
    blit->endEncoding();
    commands->commit();
    commands->waitUntilCompleted();
    ASSERT_EQ(commands->status(), MTL::CommandBufferStatusCompleted);
    for (size_t i = 0; i < 256; ++i)
      EXPECT_EQ(static_cast<const uint8_t *>(buffer->contents())[i], 0x5a);
    auto queue = device.GetQueue(QUEUE_TYPE_GRAPHICS);
    auto context = device.CreateCommandContext();
    ASSERT_TRUE(queue);
    ASSERT_TRUE(context);
    auto done = queue.Submit(std::move(context));
    ASSERT_TRUE(done);
    EXPECT_TRUE(done.WaitHost());
  }
}

TEST(RhiMetal, CapabilitiesMatchNativeDeviceAndEnumerationRemainsStable) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto core = rhix::CreateCore(BACKEND_API_METAL);
  ASSERT_TRUE(core);
  const auto devices = core.GetPhysicalDevices();
  ASSERT_FALSE(devices.empty());
  for (size_t id = 0; id < devices.size(); ++id) {
    auto device = core.CreateDevice(static_cast<int32_t>(id));
    ASSERT_TRUE(device);
    auto native = (*static_cast<backend::metal::MetalDevice *>((device).Get())).NativeDevice();
    ASSERT_NE(native, nullptr);
    EXPECT_EQ(devices[id].name, native->name()->utf8String());
    EXPECT_EQ(devices[id].ray_query_feature, native->supportsRaytracing());
    EXPECT_FALSE(devices[id].ray_tracing_pipeline_feature);
    EXPECT_FALSE(devices[id].geometry_shader_feature);
    EXPECT_EQ(devices[id].type,
              native->hasUnifiedMemory() ? PHYSICAL_DEVICE_TYPE_INTEGRATED : PHYSICAL_DEVICE_TYPE_DEDICATED);
    EXPECT_EQ(devices[id].dedicated_memory, 0u);
  }
  const auto again = core.GetPhysicalDevices();
  ASSERT_EQ(again.size(), devices.size());
  for (size_t id = 0; id < devices.size(); ++id) {
    EXPECT_EQ(again[id].name, devices[id].name);
    EXPECT_EQ(again[id].unified_memory, devices[id].unified_memory);
    EXPECT_EQ(again[id].ray_query_feature, devices[id].ray_query_feature);
  }
  EXPECT_FALSE(core.CreateDeviceByAutoSelection({true, false, false}));
  EXPECT_FALSE(core.CreateDeviceByAutoSelection({false, false, true}));
}

TEST(RhiMetal, NativeDeviceExecutesWorkBeforeParentRelease) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto core = rhix::CreateCore(BACKEND_API_METAL);
  ASSERT_TRUE(core);
  auto first = core.CreateDeviceByAutoSelection();
  auto device = core.CreateDeviceByAutoSelection();
  ASSERT_TRUE(first);
  ASSERT_TRUE(device);
  first = {};
  auto native = (*static_cast<backend::metal::MetalDevice *>((device).Get())).NativeDevice();
  auto queue = NS::TransferPtr(native->newCommandQueue());
  ASSERT_TRUE(queue);
  auto buffer = NS::TransferPtr(native->newBuffer(256, MTL::ResourceStorageModeShared));
  ASSERT_TRUE(buffer);
  auto commands = NS::RetainPtr(queue->commandBuffer());
  ASSERT_TRUE(commands);
  auto blit = NS::RetainPtr(commands->blitCommandEncoder());
  ASSERT_TRUE(blit);
  blit->fillBuffer(buffer.get(), NS::Range::Make(0, 256), 0x5a);
  blit->endEncoding();
  commands->commit();
  commands->waitUntilCompleted();
  ASSERT_EQ(commands->status(), MTL::CommandBufferStatusCompleted);
  const auto bytes = static_cast<const uint8_t *>(buffer->contents());
  for (size_t i = 0; i < 256; ++i)
    EXPECT_EQ(bytes[i], 0x5a);
}

TEST(RhiMetal, CompletionTokensSurviveBatchesAndParentRelease) {
  std::vector<rhi::ObjPtr<rhi::ISubmissionToken>> tokens;
  {
    rhi::ObjPtr<rhi::ICore> core = rhi::CreateCore(BACKEND_API_METAL);
    ASSERT_TRUE(core);
    rhi::ObjPtr<rhi::IDevice> device = core->CreateDeviceByAutoSelection();
    ASSERT_TRUE(device);
    rhi::ObjPtr<rhi::IQueue> graphics = device->GetQueue(QUEUE_TYPE_GRAPHICS);
    rhi::ObjPtr<rhi::IQueue> transfer = device->GetQueue(QUEUE_TYPE_TRANSFER);
    ASSERT_TRUE(graphics);
    ASSERT_TRUE(transfer);
    for (int batch = 0; batch < 8; ++batch) {
      std::vector<rhi::ObjPtr<rhi::ICommandContext>> held;
      for (int i = 0; i < 16; ++i) {
        // Abandoned recordings must not consume queue capacity permanently.
        {
          rhi::ObjPtr<rhi::ICommandContext> abandoned = device->CreateCommandContext();
          ASSERT_TRUE(abandoned);
        }
        rhi::ObjPtr<rhi::ICommandContext> context = device->CreateCommandContext();
        ASSERT_TRUE(context);
        auto queue = i % 2 ? graphics.Get() : transfer.Get();
        rhi::SubmitInfo desc;
        if (!tokens.empty())
          desc.waits.push_back(tokens.back().Get());
        rhi::ObjPtr<rhi::ISubmissionToken> token = queue->Submit(context.Get(), desc);
        ASSERT_TRUE(token);
        tokens.push_back(std::move(token));
        held.push_back(std::move(context));
      }
      ASSERT_TRUE(tokens.back()->WaitHost());
      for (const auto &token : tokens)
        EXPECT_EQ(token->GetStatus(), SUBMISSION_STATUS_COMPLETED);
    }
  }  // Completed tokens retain no contexts, windows or device.
  for (const auto &token : tokens) {
    EXPECT_TRUE(token->WaitHost());
    EXPECT_EQ(token->GetStatus(), SUBMISSION_STATUS_COMPLETED);
  }
}

TEST(RhiMetal, OrderedClearsReadbackAndRetainedPresentedImages) {
  if (!std::getenv("LONGMARCH_RHI_WINDOW_TESTS"))
    GTEST_SKIP() << "Requires a desktop.";
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto core = rhix::CreateCore(BACKEND_API_METAL);
  auto device = core.CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto queue = device.GetQueue(QUEUE_TYPE_GRAPHICS);
  auto first = device.CreateGLFWWindow({160, 120, "Metal readback A"});
  auto second = device.CreateGLFWWindow({160, 120, "Metal readback B"});
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  auto a = first.GetNextImage();
  auto b = second.GetNextImage();
  auto clear = device.CreateCommandContext();
  ASSERT_TRUE(clear.CmdClearColor(a, {1, 0, 0, 1}));
  ASSERT_TRUE(clear.CmdClearColor(b, {0, 1, 0, 1}));
  ASSERT_TRUE(clear.CmdClearColor(a, {0.6f, 0.7f, 0.8f, 1}));
  auto done = queue.Submit(std::move(clear));
  ASSERT_TRUE(done.WaitHost());
  {
    auto abandoned = device.CreateCommandContext();
    ASSERT_TRUE(abandoned.CmdClearColor(a, {1, 0, 0, 1}));
  }
  auto native = static_cast<backend::metal::MetalDevice *>(device.Get());
  auto readback = NS::TransferPtr(native->NativeDevice()->newBuffer(256, MTL::ResourceStorageModeShared));
  ASSERT_TRUE(readback);
  auto check = [&](const Image &image, std::array<int, 4> expected) {
    auto commands = NS::RetainPtr(native->NativeQueue()->commandBuffer());
    auto blit = commands->blitCommandEncoder();
    auto texture = static_cast<backend::metal::MetalImage *>(image.Get())->NativeImage();
    blit->copyFromTexture(texture, 0, 0, MTL::Origin::Make(0, 0, 0), MTL::Size::Make(1, 1, 1), readback.get(), 0, 256,
                          256);
    blit->endEncoding();
    commands->commit();
    commands->waitUntilCompleted();
    ASSERT_EQ(commands->status(), MTL::CommandBufferStatusCompleted);
    auto bytes = static_cast<const uint8_t *>(readback->contents());
    for (size_t i = 0; i < 4; ++i)
      EXPECT_NEAR(bytes[i], expected[i], 1);
  };
  check(a, {204, 179, 153, 255});
  check(b, {0, 255, 0, 255});
  a.Release();
  b.Release();
  std::vector<Image> old_images;
  for (int i = 0; i < 12; ++i) {
    old_images.push_back(first.GetNextImage());
    auto present = device.CreateCommandContext();
    ASSERT_TRUE(present.CmdPresent(first));
    auto presented = queue.Submit(std::move(present));
    ASSERT_TRUE(presented);
    ASSERT_TRUE(presented.WaitHost());
  }
}

TEST(RhiMetal, ResizeAndRestoreAfterModeChangeWhileMinimized) {
  if (!std::getenv("LONGMARCH_RHI_WINDOW_TESTS"))
    GTEST_SKIP() << "Requires a desktop.";
  auto core = rhix::CreateCore(BACKEND_API_METAL);
  auto device = core.CreateDeviceByAutoSelection();
  auto window = device.CreateGLFWWindow({160, 120, "Metal resize test", PRESENT_MODE_CONTINUOUS});
  ASSERT_TRUE(window);
  EXPECT_EQ(window.GetPresentMode(), PRESENT_MODE_CONTINUOUS);
  int user_data = 42;
  glfwSetWindowUserPointer(window.GLFWWindow(), &user_data);
  glfwSetWindowSize(window.GLFWWindow(), 300, 200);
  auto poll_until = [&](auto predicate) {
    for (int i = 0; i < 200; ++i) {
      if (!Window::PollEvents())
        return false;
      if (predicate())
        return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
  };
  ASSERT_TRUE(poll_until([&] {
    int width, height;
    glfwGetFramebufferSize(window.GLFWWindow(), &width, &height);
    auto image = window.GetNextImage();
    return image && image.GetDesc().width == width && image.GetDesc().height == height;
  }));
  EXPECT_EQ(glfwGetWindowUserPointer(window.GLFWWindow()), &user_data);
  EXPECT_EQ(window.GetPresentMode(), PRESENT_MODE_CONTINUOUS);
  ASSERT_TRUE(window.SetPresentMode(PRESENT_MODE_VSYNC));
  glfwIconifyWindow(window.GLFWWindow());
  ASSERT_TRUE(poll_until([&] { return glfwGetWindowAttrib(window.GLFWWindow(), GLFW_ICONIFIED); }));
  EXPECT_FALSE(window.GetNextImage());
  ASSERT_TRUE(window.SetPresentMode(PRESENT_MODE_CONTINUOUS));
  glfwRestoreWindow(window.GLFWWindow());
  ASSERT_TRUE(poll_until([&] { return !glfwGetWindowAttrib(window.GLFWWindow(), GLFW_ICONIFIED); }));
  EXPECT_TRUE(window.GetNextImage());
  EXPECT_EQ(window.GetPresentMode(), PRESENT_MODE_CONTINUOUS);
}

}  // namespace
}  // namespace grassland::rhix
