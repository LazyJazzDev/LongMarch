#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <type_traits>

#include "grassland/rhi/rhi_core.h"
#include "grassland/rhi/rhi_cuda_buffer.h"
#include "grassland/rhi/rhi_device.h"
#include "grassland/rhi/rhi_obj_ptr.h"
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#endif

#if defined(LONGMARCH_CUDA_ENABLED)
cudaError_t TransformCUDABuffer(void *, uint32_t, cudaStream_t);
#endif

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

struct Stream {
  cudaStream_t value{};

  ~Stream() {
    if (value) {
      cudaStreamSynchronize(value);
      cudaStreamDestroy(value);
    }
  }
};

class RhiCUDADevice : public testing::TestWithParam<BackendAPI> {};

void ExpectCUDAProceduresLoaded(IDevice *device, BackendAPI api) {
#if defined(LONGMARCH_VULKAN_ENABLED)
  if (api == BACKEND_API_VULKAN) {
    const auto native = static_cast<backend::vulkan::VulkanDevice *>(device);
#if defined(_WIN32)
    EXPECT_NE(native->Procedures().GetMemoryWin32HandleKHR, nullptr);
#else
    EXPECT_NE(native->Procedures().GetMemoryFdKHR, nullptr);
#endif
  }
#endif
}

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
    ExpectCUDAProceduresLoaded(device.Get(), GetParam());
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
  EXPECT_EQ(static_cast<ICUDABuffer *>(base.Get()), buffer_.Get());
  buffer_.Release();
  auto retained = static_cast<ICUDABuffer *>(base.Get());
  EXPECT_EQ(retained->GetDevicePointer(), pointer);
  EXPECT_EQ(retained->GetDesc().size, input.size());
  ASSERT_EQ(cudaMemset(pointer, 0x5a, input.size()), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(output.data(), pointer, output.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](uint8_t v) { return v == 0x5a; }));
}

#if defined(LONGMARCH_CUDA_ENABLED)
TEST_P(RhiCUDABuffer, CUDAOnlyKernelReadModifyWriteKeepsPointerAndTail) {
  constexpr uint32_t count = 16387;
  constexpr size_t bytes = count * sizeof(uint32_t);
  constexpr size_t tail = 13;
  ASSERT_EQ(buffer_->GetDesc().size, bytes + tail);
  void *pointer = buffer_->GetDevicePointer();
  cudaPointerAttributes attributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attributes, pointer), cudaSuccess);
  ASSERT_EQ(cudaSetDevice(attributes.device), cudaSuccess);
  // Declared after buffer_ allocation; synchronizes before buffer destruction on all exits.
  Stream stream;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream.value, cudaStreamNonBlocking), cudaSuccess);
  std::vector<uint32_t> expected(count), output(count);
  std::iota(expected.begin(), expected.end(), 0);
  ASSERT_EQ(cudaMemcpyAsync(pointer, expected.data(), bytes, cudaMemcpyHostToDevice, stream.value), cudaSuccess);
  auto last = static_cast<uint8_t *>(pointer) + bytes;
  ASSERT_EQ(cudaMemsetAsync(last, 0x6d, tail, stream.value), cudaSuccess);
  for (int round = 0; round < 4; ++round) {
    ASSERT_EQ(TransformCUDABuffer(pointer, count, stream.value), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(output.data(), pointer, bytes, cudaMemcpyDeviceToHost, stream.value), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream.value), cudaSuccess);
    for (uint32_t i = 0; i < count; ++i) {
      expected[i] = expected[i] * 3u + 7u;
      ASSERT_EQ(output[i], expected[i]) << "round=" << round << " index=" << i;
    }
    EXPECT_EQ(buffer_->GetDevicePointer(), pointer);
  }
  std::vector<uint8_t> suffix(tail);
  ASSERT_EQ(cudaMemcpyAsync(suffix.data(), last, tail, cudaMemcpyDeviceToHost, stream.value), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.value), cudaSuccess);
  EXPECT_TRUE(std::all_of(suffix.begin(), suffix.end(), [](uint8_t v) { return v == 0x6d; }));
}
#endif

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
