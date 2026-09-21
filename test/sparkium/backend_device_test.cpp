#include <gtest/gtest.h>
#include <long_march.h>

#include <type_traits>
#ifdef SPARKIUM_NATIVE_ENABLED
#include "sparkium/backend/common/native_image.h"
#endif

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

#ifdef SPARKIUM_NATIVE_ENABLED
TEST(SparkiumBackend, CpuTextureBorrowsScenePixelsAndRetainsOwnership) {
  auto texture = std::make_shared<sparkium::TextureData>();
  texture->width = texture->height = 1;
  texture->rgba = {10, 20, 30, 255};
  auto image = std::make_unique<sparkium::backend::NativeImage>(texture);
  EXPECT_EQ(image->memory->Data(), texture->rgba.data());
  EXPECT_THROW(image->UploadData(texture->rgba.data()), std::logic_error);
  std::weak_ptr<const sparkium::TextureData> weak = texture;
  texture.reset();
  EXPECT_FALSE(weak.expired());
  std::array<uint8_t, 4> pixel{};
  image->DownloadData(pixel.data());
  EXPECT_EQ(pixel, (std::array<uint8_t, 4>{10, 20, 30, 255}));
  image.reset();
  EXPECT_TRUE(weak.expired());
}
#endif
}  // namespace
