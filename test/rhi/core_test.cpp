#include <gtest/gtest.h>

#include "grassland/rhi/rhix/rhix.h"
#if defined(LONGMARCH_D3D12_ENABLED)
#include <d3d12sdklayers.h>

#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/rhi/backend/vulkan/vulkan_core.h"
#endif

namespace grassland::rhi {
TEST(RhiCoreSettings, ReportsBackendThroughRawAndWrapperInterfaces) {
  std::vector<BackendAPI> apis{BACKEND_API_DEFAULT};
#if defined(LONGMARCH_METAL_ENABLED)
  apis.push_back(BACKEND_API_METAL);
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
  apis.push_back(BACKEND_API_VULKAN);
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
  apis.push_back(BACKEND_API_D3D12);
#endif
  for (auto api : apis) {
    auto core = rhix::CreateCore(api);
    ASSERT_TRUE(core);
    EXPECT_EQ(core.GetBackendAPI(), api);
    EXPECT_EQ(core.Get()->GetBackendAPI(), api);
  }
}

TEST(RhiCoreSettings, DefaultFollowsBuildConfiguration) {
#if !defined(NDEBUG)
  EXPECT_TRUE(CoreDesc{}.enable_debug);
#else
  EXPECT_FALSE(CoreDesc{}.enable_debug);
#endif
}

#if defined(LONGMARCH_D3D12_ENABLED)
TEST(RhiCoreSettings, D3D12OverridesDefaultAcrossSequentialCores) {
  for (bool enabled : {false, true, false}) {
    auto core = rhix::CreateCore(BACKEND_API_D3D12, {enabled});
    ASSERT_TRUE(core);
    ObjPtr<IDevice> device = core.Get()->CreateDeviceByAutoSelection();
    ASSERT_TRUE(device);
    auto native = static_cast<backend::d3d12::D3D12Device *>(device.Get())->NativeDevice();
    backend::d3d12::ComPtr<ID3D12InfoQueue> messages;
    EXPECT_EQ(SUCCEEDED(native->QueryInterface(IID_PPV_ARGS(&messages))), enabled);
    ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
    ObjPtr<ICommandContext> commands = device->CreateCommandContext();
    ASSERT_TRUE(queue);
    ASSERT_TRUE(commands);
    ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
    ASSERT_TRUE(done);
    ASSERT_TRUE(done->WaitHost());
  }
}

TEST(RhiCoreSettings, D3D12ValidationMessagesReachConsole) {
  ObjPtr<ICore> core = CreateCore(BACKEND_API_D3D12, {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  backend::d3d12::ComPtr<ID3D12InfoQueue1> messages;
  ASSERT_EQ(
      static_cast<backend::d3d12::D3D12Device *>(device.Get())->NativeDevice()->QueryInterface(IID_PPV_ARGS(&messages)),
      S_OK);
  testing::internal::CaptureStdout();
  auto warning = messages->AddApplicationMessage(D3D12_MESSAGE_SEVERITY_WARNING, "RHI console warning probe");
  auto error = messages->AddApplicationMessage(D3D12_MESSAGE_SEVERITY_ERROR, "RHI console error probe");
  auto output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(warning, S_OK);
  EXPECT_EQ(error, S_OK);
  EXPECT_NE(output.find("[WARNING]"), std::string::npos);
  EXPECT_NE(output.find("[ERROR]"), std::string::npos);
  EXPECT_NE(output.find("D3D12 ["), std::string::npos);
  EXPECT_NE(output.find("RHI console warning probe"), std::string::npos);
  EXPECT_NE(output.find("RHI console error probe"), std::string::npos);
}

#endif

#if defined(LONGMARCH_VULKAN_ENABLED)
TEST(RhiCoreSettings, VulkanValidationOptionsAcrossSequentialCores) {
  for (bool enabled : {false, true, false}) {
    ObjPtr<ICore> core = CreateCore(BACKEND_API_VULKAN, {enabled});
    ASSERT_TRUE(core);
    if (enabled) {
      auto instance = static_cast<backend::vulkan::VulkanCore *>(core.Get())->Instance();
      EXPECT_NE(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"), nullptr);
    }
    ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
    ASSERT_TRUE(device);
  }
}
#endif
}  // namespace grassland::rhi
