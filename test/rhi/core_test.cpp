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
