#include <gtest/gtest.h>
#include <long_march.h>

#include <type_traits>

namespace {
using namespace grassland;

static_assert(!std::is_base_of_v<graphics::Core, sparkium::backend::Device>);

TEST(SparkiumBackend, NativeDevicesAreNotGraphicsDevices) {
  EXPECT_THROW(sparkium::ToGraphicsBackend(sparkium::RenderBackend::CPU), std::invalid_argument);
  EXPECT_THROW(sparkium::ToGraphicsBackend(sparkium::RenderBackend::CUDA), std::invalid_argument);
  // The former graphics CPU/CUDA values must not silently select a graphics API.
  for (int value : {3, 4}) {
    auto api = static_cast<graphics::BackendAPI>(value);
    EXPECT_FALSE(graphics::SupportBackendAPI(api));
    std::unique_ptr<graphics::Core> graphics_device;
    EXPECT_NE(graphics::CreateCore(api, {}, &graphics_device), 0);
    EXPECT_EQ(graphics_device, nullptr);
  }
  if (!sparkium::SupportBackend(sparkium::RenderBackend::CPU))
    return;
  std::unique_ptr<sparkium::backend::Device> device;
  ASSERT_EQ(sparkium::CreateDevice(sparkium::RenderBackend::CPU, {}, &device), 0);
  EXPECT_EQ(device->API(), sparkium::RenderBackend::CPU);
  EXPECT_EQ(device->GraphicsCore(), nullptr);
  ASSERT_EQ(device->InitializeLogicalDeviceAutoSelect(false), 0);
  std::unique_ptr<graphics::Buffer> buffer;
  ASSERT_EQ(device->CreateBuffer(sizeof(uint32_t), graphics::BUFFER_TYPE_STATIC, &buffer), 0);
  uint32_t input = 0x12345678, output = 0;
  buffer->UploadData(&input, sizeof(input));
  buffer->DownloadData(&output, sizeof(output));
  EXPECT_EQ(input, output);
}

TEST(SparkiumBackend, GraphicsAdapterPreservesDeviceIdentity) {
  std::unique_ptr<sparkium::backend::Device> device;
  ASSERT_EQ(sparkium::CreateDevice(sparkium::RenderBackend::Graphics, {}, &device), 0);
  ASSERT_NE(device->GraphicsCore(), nullptr);
  EXPECT_EQ(device->API(), sparkium::RenderBackend::Graphics);
  EXPECT_EQ(device->GraphicsCore()->API(), graphics::BACKEND_API_DEFAULT);
  EXPECT_EQ(device->DeviceRayQuerySupport(), device->GraphicsCore()->DeviceRayQuerySupport());
}
}  // namespace
