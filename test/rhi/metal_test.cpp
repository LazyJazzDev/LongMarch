#include <gtest/gtest.h>

#include "grassland/rhi/backend/metal/metal_core.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/rhi.h"

namespace grassland::rhi {
namespace {

TEST(RhiMetal, CapabilitiesMatchNativeDeviceAndEnumerationRemainsStable) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto core = CreateCore(BACKEND_API_METAL);
  ASSERT_TRUE(core);
  const auto devices = core.GetPhysicalDevices();
  ASSERT_FALSE(devices.empty());
  for (size_t id = 0; id < devices.size(); ++id) {
    auto device = core.CreateDevice(static_cast<int32_t>(id));
    ASSERT_TRUE(device);
    auto native = backend::Access::As<backend::metal::MetalDevice>(device).NativeDevice();
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

TEST(RhiMetal, NativeDeviceSurvivesCoreAndSiblingReleaseAndCanExecuteWork) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto core = CreateCore(BACKEND_API_METAL);
  ASSERT_TRUE(core);
  auto first = core.CreateDeviceByAutoSelection();
  auto device = core.CreateDeviceByAutoSelection();
  ASSERT_TRUE(first);
  ASSERT_TRUE(device);
  core = {};
  first = {};
  auto native = backend::Access::As<backend::metal::MetalDevice>(device).NativeDevice();
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

}  // namespace
}  // namespace grassland::rhi
