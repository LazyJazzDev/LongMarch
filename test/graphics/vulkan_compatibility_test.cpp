#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "grassland/graphics/graphics.h"

namespace {
using namespace grassland::graphics;

// No scene assets or mobile host are needed. Enable synchronization validation
// with VK_LAYER_VALIDATE_SYNC=1 when running this explicit Vulkan GPU test.
void CheckTiledComputeWithExplicitPixelOrigins(BackendAPI backend) {
  std::unique_ptr<Core> core;
  ASSERT_EQ(CreateCore(backend, Core::Settings{1, true}, &core), 0);
  ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
  constexpr uint32_t width = 259, height = 130;
  std::unique_ptr<Image> image;
  ASSERT_EQ(core->CreateImage(width, height, IMAGE_FORMAT_R32G32B32A32_SFLOAT, &image), 0);
  std::unique_ptr<Buffer> parameters;
  ASSERT_EQ(core->CreateBuffer(16, BUFFER_TYPE_DYNAMIC, &parameters), 0);
  std::unique_ptr<Shader> shader;
  ASSERT_EQ(core->CreateShader(R"(
cbuffer Parameters : register(b0, space0) { uint width; uint height; uint value; uint unused; };
RWTexture2D<float4> output : register(u0, space1);
cbuffer Tile : register(b0, space2) { uint2 origin; uint2 padding; };
[numthreads(8,8,1)] void Main(uint3 id : SV_DispatchThreadID) {
  id.xy += origin;
  if (id.x < width && id.y < height)
    output[id.xy] += float4(id.x, id.y, value, 1);
}
)",
                               "Main", "cs_6_0", &shader),
            0);
  std::unique_ptr<ComputeProgram> program;
  ASSERT_EQ(core->CreateComputeProgram(shader.get(), &program), 0);
  program->AddResourceBinding(RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program->AddResourceBinding(RESOURCE_TYPE_WRITABLE_IMAGE, 1);
  program->AddResourceBinding(RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program->Finalize();
  std::vector<std::array<float, 4>> pixels(width * height, {1, 2, 3, 4});
  image->UploadData(pixels.data());
  for (uint32_t phase = 1; phase <= 2; ++phase) {
    const std::array<uint32_t, 4> values{width, height, phase * 7, 0};
    parameters->UploadData(values.data(), sizeof(values));
    std::unique_ptr<CommandContext> commands;
    ASSERT_EQ(core->CreateCommandContext(&commands), 0);
    commands->CmdBindComputeProgram(program.get());
    commands->CmdBindResources(0, std::vector<Buffer *>{parameters.get()}, BIND_POINT_COMPUTE);
    commands->CmdBindResources(1, std::vector<Image *>{image.get()}, BIND_POINT_COMPUTE);
    const uint32_t nx = (width + 7) / 8, ny = (height + 7) / 8;
    std::vector<std::unique_ptr<Buffer>> origins;
    for (uint32_t y = 0; y < ny; y += 16)
      for (uint32_t x = 0; x < nx; x += 16) {
        std::unique_ptr<Buffer> origin;
        ASSERT_EQ(core->CreateBuffer(16, BUFFER_TYPE_DYNAMIC, &origin), 0);
        const std::array<uint32_t, 4> tile{x * 8, y * 8, 0, 0};
        origin->UploadData(tile.data(), sizeof(tile));
        commands->CmdBindResources(2, std::vector<Buffer *>{origin.get()}, BIND_POINT_COMPUTE);
        commands->CmdDispatch(std::min(16u, nx - x), std::min(16u, ny - y), 1);
        origins.push_back(std::move(origin));
      }
    ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
    core->WaitGPU();
    image->DownloadData(pixels.data());
    for (uint32_t y = 0; y < height; ++y)
      for (uint32_t x = 0; x < width; ++x) {
        const auto &pixel = pixels[y * width + x];
        ASSERT_EQ(pixel[0], 1 + phase * x) << x << "," << y;
        ASSERT_EQ(pixel[1], 2 + phase * y) << x << "," << y;
        ASSERT_EQ(pixel[2], phase == 1 ? 10 : 24) << x << "," << y;
        ASSERT_EQ(pixel[3], 4 + phase) << x << "," << y;
      }
  }
}

TEST(VulkanCompatibility, TiledComputeUsesExplicitPixelOrigins) {
  CheckTiledComputeWithExplicitPixelOrigins(BACKEND_API_VULKAN);
}

#ifdef LONGMARCH_METAL_ENABLED
TEST(MetalCompatibility, TiledComputeUsesExplicitPixelOrigins) {
  CheckTiledComputeWithExplicitPixelOrigins(BACKEND_API_METAL);
}
#endif

TEST(VulkanCompatibility, DivergentStorageBuffersThroughHelperProduceCorrectValues) {
  std::unique_ptr<Core> core;
  ASSERT_EQ(CreateCore(BACKEND_API_VULKAN, Core::Settings{1, true}, &core), 0);
  ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
  std::array<std::unique_ptr<Buffer>, 2> inputs;
  for (uint32_t i = 0; i < inputs.size(); ++i) {
    ASSERT_EQ(core->CreateBuffer(16, BUFFER_TYPE_STATIC, &inputs[i]), 0);
    const std::array<uint32_t, 4> data{11 + 100 * i, 0, 0, 0};
    inputs[i]->UploadData(data.data(), sizeof(data));
  }
  std::unique_ptr<Buffer> output;
  ASSERT_EQ(core->CreateBuffer(64 * sizeof(uint32_t), BUFFER_TYPE_STATIC, &output), 0);
  std::unique_ptr<Shader> shader;
  ASSERT_EQ(core->CreateShader(R"(
ByteAddressBuffer inputs[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);
uint ReadValue(uint index) {
  ByteAddressBuffer selected = inputs[NonUniformResourceIndex(index)];
  return selected.Load(0);
}
[numthreads(64,1,1)] void Main(uint3 id : SV_DispatchThreadID) {
  output[id.x] = ReadValue(id.x % 2) + id.x;
}
)",
                               "Main", "cs_6_0", &shader),
            0);
  std::unique_ptr<ComputeProgram> program;
  ASSERT_EQ(core->CreateComputeProgram(shader.get(), &program), 0);
  program->AddResourceBinding(RESOURCE_TYPE_STORAGE_BUFFER, 2);
  program->AddResourceBinding(RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  program->Finalize();
  std::unique_ptr<CommandContext> commands;
  ASSERT_EQ(core->CreateCommandContext(&commands), 0);
  commands->CmdBindComputeProgram(program.get());
  commands->CmdBindResources(0, std::vector<Buffer *>{inputs[0].get(), inputs[1].get()}, BIND_POINT_COMPUTE);
  commands->CmdBindResources(1, std::vector<Buffer *>{output.get()}, BIND_POINT_COMPUTE);
  commands->CmdDispatch(1, 1, 1);
  ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
  core->WaitGPU();
  std::array<uint32_t, 64> result{};
  output->DownloadData(result.data(), sizeof(result));
  for (uint32_t i = 0; i < result.size(); ++i)
    EXPECT_EQ(result[i], 11 + (i % 2) * 100 + i) << i;
}
}  // namespace
