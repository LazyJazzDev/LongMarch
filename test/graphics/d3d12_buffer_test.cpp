#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <vector>

#include "grassland/graphics/graphics.h"

namespace {
using namespace grassland::graphics;

class D3D12BufferTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(CreateCore(BACKEND_API_D3D12, Core::Settings{2, true}, &core_), 0);
    ASSERT_EQ(core_->InitializeLogicalDeviceAutoSelect(false), 0);
  }

  void TearDown() override {
    if (core_) {
      core_->WaitGPU();
    }
  }

  void CheckResize(Buffer *buffer) {
    ASSERT_EQ(buffer->Size(), 4u);
    std::vector<uint8_t> expected(4, 0x31);
    buffer->UploadData(expected.data(), expected.size());
    // Exercise growth/shrink within one physical allocation size and across
    // D3D12's 256-byte boundary, including non-word-sized logical buffers.
    for (size_t size : {12u, 252u, 256u, 260u, 7u, 4u}) {
      SCOPED_TRACE(size);
      buffer->Resize(size);
      ASSERT_EQ(buffer->Size(), size);
      EXPECT_EQ(buffer->Range().size, size);
      EXPECT_EQ(buffer->Range(1).size, size - 1);
      std::vector<uint8_t> preserved(std::min(size, expected.size()));
      buffer->DownloadData(preserved.data(), preserved.size());
      ASSERT_TRUE(std::equal(preserved.begin(), preserved.end(), expected.begin()));
      expected.resize(size);
      for (size_t i = 0; i < size; ++i) {
        expected[i] = static_cast<uint8_t>(size + i);
      }
      buffer->UploadData(expected.data(), expected.size());
      std::vector<uint8_t> actual(size);
      buffer->DownloadData(actual.data(), actual.size());
      EXPECT_EQ(actual, expected);
    }
  }

  std::unique_ptr<Core> core_;
};

TEST_F(D3D12BufferTest, StaticSizeAndResizePreserveLogicalContents) {
  std::unique_ptr<Buffer> buffer;
  ASSERT_EQ(core_->CreateBuffer(4, BUFFER_TYPE_STATIC, &buffer), 0);
  CheckResize(buffer.get());
}

TEST_F(D3D12BufferTest, DynamicSizeAndResizePreserveLogicalContents) {
  std::unique_ptr<Buffer> buffer;
  ASSERT_EQ(core_->CreateBuffer(4, BUFFER_TYPE_DYNAMIC, &buffer), 0);
  CheckResize(buffer.get());
}

TEST_F(D3D12BufferTest, DynamicTransferAndShaderViewsUseLogicalRanges) {
  std::unique_ptr<Buffer> input, parameters, output;
  ASSERT_EQ(core_->CreateBuffer(4, BUFFER_TYPE_DYNAMIC, &input), 0);
  ASSERT_EQ(core_->CreateBuffer(4, BUFFER_TYPE_DYNAMIC, &parameters), 0);
  ASSERT_EQ(core_->CreateBuffer(8, BUFFER_TYPE_STATIC, &output), 0);
  std::unique_ptr<Shader> shader;
  ASSERT_EQ(core_->CreateShader(R"(
ByteAddressBuffer input : register(t0, space0);
cbuffer Parameters : register(b0, space1) { uint value; };
RWByteAddressBuffer output : register(u0, space2);
[numthreads(1,1,1)] void Main() {
  uint bytes;
  input.GetDimensions(bytes);
  output.Store(0, bytes);
  output.Store(4, input.Load(bytes - 4) + value);
}
)",
                                "Main", "cs_6_0", &shader),
            0);
  std::unique_ptr<ComputeProgram> program;
  ASSERT_EQ(core_->CreateComputeProgram(shader.get(), &program), 0);
  program->AddResourceBinding(RESOURCE_TYPE_STORAGE_BUFFER, 1);
  program->AddResourceBinding(RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program->AddResourceBinding(RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  program->Finalize();
  const uint32_t value = 17;
  parameters->UploadData(&value, sizeof(value));
  for (size_t size : {4u, 12u, 260u, 8u}) {
    input->Resize(size);
    std::vector<uint32_t> values(size / 4, static_cast<uint32_t>(size));
    input->UploadData(values.data(), size);
    // Visit both frame slots after each resize.
    for (int frame = 0; frame < core_->FramesInFlight(); ++frame) {
      std::unique_ptr<CommandContext> commands;
      ASSERT_EQ(core_->CreateCommandContext(&commands), 0);
      commands->CmdBindComputeProgram(program.get());
      commands->CmdBindResources(0, std::vector<Buffer *>{input.get()}, BIND_POINT_COMPUTE);
      commands->CmdBindResources(1, std::vector<Buffer *>{parameters.get()}, BIND_POINT_COMPUTE);
      commands->CmdBindResources(2, std::vector<Buffer *>{output.get()}, BIND_POINT_COMPUTE);
      commands->CmdDispatch(1, 1, 1);
      ASSERT_EQ(core_->SubmitCommandContext(commands.get()), 0);
      core_->WaitGPU();
      std::array<uint32_t, 2> result{};
      output->DownloadData(result.data(), sizeof(result));
      EXPECT_EQ(result[0], size);
      EXPECT_EQ(result[1], size + value);
    }
  }
}

#if defined(LONGMARCH_CUDA_RUNTIME)
TEST_F(D3D12BufferTest, CudaSizeResizeAndMappingPreserveLogicalContents) {
  if (core_->CUDADeviceIndex() < 0) {
    GTEST_SKIP() << "Selected graphics device does not support CUDA interop";
  }
  ASSERT_EQ(cudaSetDevice(core_->CUDADeviceIndex()), cudaSuccess);
  std::unique_ptr<CUDABuffer> buffer;
  ASSERT_EQ(core_->CreateCUDABuffer(4, &buffer), 0);
  CheckResize(buffer.get());
  for (size_t size : {12u, 260u, 7u}) {
    buffer->Resize(size);
    void *mapped = nullptr;
    buffer->GetCUDAMemoryPointer(&mapped);
    ASSERT_NE(mapped, nullptr);
    ASSERT_EQ(cudaMemset(mapped, 0x5a, size), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaFree(mapped), cudaSuccess);
    std::vector<uint8_t> actual(size);
    buffer->DownloadData(actual.data(), actual.size());
    EXPECT_EQ(actual, std::vector<uint8_t>(size, 0x5a));
    EXPECT_EQ(buffer->Size(), size);
  }
}
#endif
}  // namespace
