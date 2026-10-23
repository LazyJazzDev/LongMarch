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
TEST(RhiCoreSettings, D3D12OverridesDefaultAndSharesValidationUntilLastCore) {
  for (bool enabled : {false, true, false}) {
    auto first = rhix::CreateCore(BACKEND_API_D3D12, {enabled});
    ASSERT_TRUE(first);
    ObjPtr<ICore> second = CreateCore(BACKEND_API_D3D12, {enabled});
    ASSERT_TRUE(second);
    ObjPtr<IDevice> device = second->CreateDeviceByAutoSelection();
    ASSERT_TRUE(device);
    auto native = static_cast<backend::d3d12::D3D12Device *>(device.Get())->NativeDevice();
    backend::d3d12::ComPtr<ID3D12InfoQueue> messages;
    EXPECT_EQ(SUCCEEDED(native->QueryInterface(IID_PPV_ARGS(&messages))), enabled);
    EXPECT_EQ(CreateCore(BACKEND_API_D3D12, {!enabled}), nullptr);
    first.Release();
    // Releasing one Core must not change validation or remove the surviving device.
    EXPECT_EQ(native->GetDeviceRemovedReason(), S_OK);
    EXPECT_EQ(CreateCore(BACKEND_API_D3D12, {!enabled}), nullptr);
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
TEST(RhiCoreSettings, VulkanValidationOptionsArePerInstance) {
  ObjPtr<ICore> enabled = CreateCore(BACKEND_API_VULKAN, {true});
  ASSERT_TRUE(enabled);
  auto disabled = rhix::CreateCore(BACKEND_API_VULKAN, {false});
  ASSERT_TRUE(disabled);
  auto instance = static_cast<backend::vulkan::VulkanCore *>(enabled.Get())->Instance();
  EXPECT_NE(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"), nullptr);
  ObjPtr<IDevice> first_device = enabled->CreateDeviceByAutoSelection();
  ASSERT_TRUE(first_device);
  auto second_device = disabled.CreateDeviceByAutoSelection();
  ASSERT_TRUE(second_device);
}
#endif
}  // namespace grassland::rhi
