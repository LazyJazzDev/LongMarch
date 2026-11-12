#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <type_traits>

#include "grassland/rhi/rhi_core.h"
#include "grassland/rhi/rhi_cuda_buffer.h"
#include "grassland/rhi/rhi_device.h"
#include "grassland/rhi/rhi_obj_ptr.h"

namespace grassland::rhi {
namespace {
static_assert(std::is_convertible_v<ICUDABuffer *, IBuffer *>);
// This downcast is only well-formed with ordinary interface inheritance.
static_assert(std::is_same_v<decltype(static_cast<ICUDABuffer *>(static_cast<IBuffer *>(nullptr))), ICUDABuffer *>);

struct RestoreDevice {
  int previous{-1};

  RestoreDevice() {
    cudaGetDevice(&previous);
  }

  ~RestoreDevice() {
    if (previous >= 0)
      cudaSetDevice(previous);
  }
};

class RhiCUDADevice : public testing::TestWithParam<BackendAPI> {};

TEST_P(RhiCUDADevice, CUDADevicesExposeOrdinalBeforeBufferCreation) {
  RestoreDevice restore;
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  const auto adapters = core->GetPhysicalDevices();
  int matches = 0;
  for (size_t i = 0; i < adapters.size(); ++i) {
    const auto &properties = adapters[i];
    if (properties.cuda_device < 0)
      continue;
    SCOPED_TRACE(properties.name);
    ++matches;
    cudaDeviceProp cuda_properties{};
    ASSERT_EQ(cudaGetDeviceProperties(&cuda_properties, properties.cuda_device), cudaSuccess);
    ObjPtr<IDevice> device = core->CreateDevice(static_cast<int32_t>(i));
    ASSERT_TRUE(device);
    EXPECT_EQ(device->Name(), properties.name);
    ASSERT_EQ(device->GetCUDADeviceIndex(), properties.cuda_device);
    int previous = -1;
    ASSERT_EQ(cudaGetDevice(&previous), cudaSuccess);
    ObjPtr<ICUDABuffer> buffer = device->CreateCUDABuffer({4096});
    ASSERT_TRUE(buffer);
    int current = -1;
    ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, previous);
    cudaPointerAttributes attributes{};
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, buffer->GetDevicePointer()), cudaSuccess);
    EXPECT_EQ(attributes.device, properties.cuda_device);
  }
  if (!matches)
    GTEST_SKIP() << "No CUDA graphics adapter is available";
  RecordProperty("cuda_adapters_tested", matches);
}

TEST_P(RhiCUDADevice, NonCUDADevicesRejectCUDABufferCreation) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  const auto adapters = core->GetPhysicalDevices();
  int matches = 0;
  for (size_t i = 0; i < adapters.size(); ++i) {
    if (adapters[i].cuda_device >= 0)
      continue;
    SCOPED_TRACE(adapters[i].name);
    ++matches;
    EXPECT_EQ(adapters[i].cuda_device, -1);
    ObjPtr<IDevice> device = core->CreateDevice(static_cast<int32_t>(i));
    ASSERT_TRUE(device);
    EXPECT_EQ(device->GetCUDADeviceIndex(), -1);
    ObjPtr<ICUDABuffer> buffer = device->CreateCUDABuffer({4096});
    EXPECT_FALSE(buffer);
  }
  if (!matches)
    GTEST_SKIP() << "No non-CUDA graphics adapter is available";
  RecordProperty("non_cuda_adapters_tested", matches);
}

class RhiCUDABuffer : public testing::TestWithParam<BackendAPI> {
 protected:
  void SetUp() override {
    int count{};
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count)
      GTEST_SKIP() << "CUDA device unavailable";
    core_ = CreateCore(GetParam());
    ASSERT_TRUE(core_);
    // Select from cached device properties before creating any CUDA buffer.
    const auto adapters = core_->GetPhysicalDevices();
    for (size_t i = 0; i < adapters.size(); ++i) {
      if (adapters[i].cuda_device < 0)
        continue;
      device_ = core_->CreateDevice(static_cast<int32_t>(i));
      ASSERT_TRUE(device_);
      ASSERT_EQ(device_->GetCUDADeviceIndex(), adapters[i].cuda_device);
      buffer_ = device_->CreateCUDABuffer({65561});
      ASSERT_TRUE(buffer_);
      return;
    }
    GTEST_SKIP() << "No CUDA graphics adapter is available";
  }

  RestoreDevice restore_;
  ObjPtr<ICore> core_;
  ObjPtr<IDevice> device_;
  ObjPtr<ICUDABuffer> buffer_;
};

TEST_P(RhiCUDABuffer, CUDAOnlyCopiesAndOrdinaryInterfaceLifetime) {
  ASSERT_EQ(buffer_->GetDesc().size, 65561u);
  EXPECT_EQ(device_->CreateCUDABuffer({0}), nullptr);
  void *pointer = buffer_->GetDevicePointer();
  ASSERT_NE(pointer, nullptr);
  cudaPointerAttributes attributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attributes, pointer), cudaSuccess);
  EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
  EXPECT_EQ(attributes.device, device_->GetCUDADeviceIndex());
  ASSERT_EQ(cudaSetDevice(attributes.device), cudaSuccess);
  std::vector<uint8_t> input(65561), output(input.size());
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<uint8_t>(i * 13 + 5);
  ASSERT_EQ(cudaMemcpy(pointer, input.data(), input.size(), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(output.data(), pointer, output.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(input, output);
  ObjPtr<IBuffer> base = buffer_;
  EXPECT_EQ(QueryInterfacePtr<ICUDABuffer>(base.Get()).Get(), buffer_.Get());
  buffer_.Release();
  auto retained = QueryInterfacePtr<ICUDABuffer>(base.Get());
  ASSERT_TRUE(retained);
  EXPECT_EQ(QueryInterfacePtr<IObject>(base.Get()).Get(), QueryInterfacePtr<IObject>(retained.Get()).Get());
  base.Release();
  EXPECT_EQ(retained->GetDevicePointer(), pointer);
  EXPECT_EQ(retained->GetDesc().size, input.size());
  ASSERT_EQ(cudaMemset(pointer, 0x5a, input.size()), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(output.data(), pointer, output.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](uint8_t v) { return v == 0x5a; }));
}

std::vector<BackendAPI> Backends() {
  std::vector<BackendAPI> result;
#if defined(LONGMARCH_D3D12_ENABLED)
  result.push_back(BACKEND_API_D3D12);
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
  result.push_back(BACKEND_API_VULKAN);
#endif
  return result;
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiCUDABuffer, testing::ValuesIn(Backends()));
INSTANTIATE_TEST_SUITE_P(Backends, RhiCUDADevice, testing::ValuesIn(Backends()));
}  // namespace
}  // namespace grassland::rhi
