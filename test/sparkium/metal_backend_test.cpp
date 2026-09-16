#include <gtest/gtest.h>
#include <long_march.h>

#include <cstring>
#include <numeric>

#if defined(LONGMARCH_METAL_ENABLED)
using namespace grassland;
namespace {
class MetalBackendTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(graphics::CreateCore(graphics::BACKEND_API_METAL, {}, &core), 0);
    ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
  }
  std::unique_ptr<graphics::Core> core;
};
TEST_F(MetalBackendTest, LargeArgumentArraysAndBindingSnapshots) {
  constexpr int count = 80;  // Deliberately exceeds the 31 direct Metal buffer slots.
  std::vector<std::unique_ptr<graphics::Buffer>> inputs(count);
  std::vector<graphics::Buffer *> bindings;
  for (int i = 0; i < count; ++i) {
    ASSERT_EQ(core->CreateBuffer(4, graphics::BUFFER_TYPE_STATIC, &inputs[i]), 0);
    uint32_t value = i + 1;
    inputs[i]->UploadData(&value, 4);
    bindings.push_back(inputs[i].get());
  }
  std::unique_ptr<graphics::Buffer> output1, output2;
  core->CreateBuffer(count * 4, graphics::BUFFER_TYPE_STATIC, &output1);
  core->CreateBuffer(count * 4, graphics::BUFFER_TYPE_STATIC, &output2);
  std::unique_ptr<graphics::Shader> shader;
  ASSERT_EQ(core->CreateShader(R"(
ByteAddressBuffer inputs[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);
[numthreads(1, 1, 1)] void Main(uint3 tid : SV_DispatchThreadID) {
  output[tid.x] = inputs[NonUniformResourceIndex(tid.x)].Load(0);
})",
                               "Main", "cs_6_0", &shader),
            0);
  std::unique_ptr<graphics::ComputeProgram> program;
  core->CreateComputeProgram(shader.get(), &program);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, count);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  program->Finalize();
  std::unique_ptr<graphics::CommandContext> commands;
  core->CreateCommandContext(&commands);
  commands->CmdBindComputeProgram(program.get());
  commands->CmdBindResources(0, bindings, graphics::BIND_POINT_COMPUTE);
  commands->CmdBindResources(1, {output1.get()}, graphics::BIND_POINT_COMPUTE);
  commands->CmdDispatch(count, 1, 1);
  std::reverse(bindings.begin(), bindings.end());
  commands->CmdBindResources(0, bindings, graphics::BIND_POINT_COMPUTE);
  commands->CmdBindComputeProgram(program.get());  // Compatible bindings survive a program switch.
  commands->CmdBindResources(1, {output2.get()}, graphics::BIND_POINT_COMPUTE);
  commands->CmdDispatch(count, 1, 1);
  ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
  commands.reset();  // Native command retains argument snapshots until completion.
  std::vector<uint32_t> actual(count);
  output1->DownloadData(actual.data(), count * 4);
  for (int i = 0; i < count; ++i)
    EXPECT_EQ(actual[i], i + 1);
  output2->DownloadData(actual.data(), count * 4);
  for (int i = 0; i < count; ++i)
    EXPECT_EQ(actual[i], count - i);
}
TEST_F(MetalBackendTest, LightSelectionPartialWorkgroupDoesNotOverwriteTail) {
  sparkium::Core renderer(core.get());
  std::unique_ptr<graphics::Shader> shader;
  ASSERT_EQ(core->CreateShader(renderer.GetShadersVFS(), "gather_light_power.hlsl", "GatherLightPowerKernel", "cs_6_3",
                               {"-I."}, &shader),
            0);
  std::unique_ptr<graphics::ComputeProgram> program;
  core->CreateComputeProgram(shader.get(), &program);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
  program->Finalize();
  for (uint32_t count : {1u, 5u, 63u, 65u}) {
    SCOPED_TRACE(count);
    std::unique_ptr<graphics::Buffer> metadata, power, selector;
    core->CreateBuffer(count * 16, graphics::BUFFER_TYPE_STATIC, &metadata);
    core->CreateBuffer(4, graphics::BUFFER_TYPE_STATIC, &power);
    core->CreateBuffer((count + 65) * 4, graphics::BUFFER_TYPE_STATIC, &selector);
    std::vector<uint32_t> records(count * 4, 0), actual(count + 65, 0xdeadbeef);
    metadata->UploadData(records.data(), records.size() * 4);
    float one = 1;
    power->UploadData(&one, 4);
    actual[0] = count;
    selector->UploadData(actual.data(), actual.size() * 4);
    std::unique_ptr<graphics::CommandContext> commands;
    core->CreateCommandContext(&commands);
    commands->CmdBindComputeProgram(program.get());
    commands->CmdBindResources(0, {metadata.get()}, graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(1, {selector.get()}, graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(2, {power.get()}, graphics::BIND_POINT_COMPUTE);
    commands->CmdDispatch((count + 63) / 64, 1, 1);
    core->SubmitCommandContext(commands.get());
    selector->DownloadData(actual.data(), actual.size() * 4);
    EXPECT_EQ(actual[0], count);
    for (uint32_t i = 0; i < count; ++i) {
      float value;
      std::memcpy(&value, &actual[i + 1], 4);
      EXPECT_FLOAT_EQ(value, float(i % 64 + 1));
    }
    for (size_t i = count + 1; i < actual.size(); ++i)
      EXPECT_EQ(actual[i], 0xdeadbeef);
  }
}
TEST_F(MetalBackendTest, TextureRoundTripAndPartialRegion) {
  for (auto format : {graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                      graphics::IMAGE_FORMAT_R32G32B32_SFLOAT, graphics::IMAGE_FORMAT_D32_SFLOAT,
                      graphics::IMAGE_FORMAT_R32_SINT, graphics::IMAGE_FORMAT_R32_UINT}) {
    SCOPED_TRACE(int(format));
    std::unique_ptr<graphics::Image> image;
    core->CreateImage(17, 13, format, &image);
    size_t pixel = graphics::PixelSize(format);
    std::vector<uint8_t> source(17 * 13 * pixel, 0), actual(source.size());
    image->UploadData(source.data());
    std::vector<uint8_t> patch(3 * 2 * pixel, 0);
    for (size_t i = 0; i < patch.size(); i += 4) {
      float value = 0.375f;
      std::memcpy(patch.data() + i, &value, 4);
    }
    image->UploadData(patch.data(), {2, 3}, {3, 2});
    for (int y = 0; y < 2; ++y)
      std::memcpy(source.data() + ((y + 3) * 17 + 2) * pixel, patch.data() + y * 3 * pixel, 3 * pixel);
    image->DownloadData(actual.data());
    EXPECT_EQ(actual, source);
    std::vector<uint8_t> region(patch.size());
    image->DownloadData(region.data(), {2, 3}, {3, 2});
    EXPECT_EQ(region, patch);
  }
}
TEST_F(MetalBackendTest, AttachmentlessRasterPassPreservesViewportAndScissor) {
  std::unique_ptr<graphics::Image> output;
  core->CreateImage(5, 4, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &output);
  const std::string source = R"(
RWTexture2D<float4> output_image : register(u0, space0);
float4 VSMain(uint id : SV_VertexID) : SV_POSITION {
  float2 p = float2((id << 1) & 2, id & 2);
  return float4(p * 2.0 - 1.0, 0, 1);
}
void PSMain(float4 position : SV_POSITION) {
  output_image[uint2(position.xy)] = float4(2, 3, 4, 1);
})";
  std::unique_ptr<graphics::Shader> vertex, fragment;
  ASSERT_EQ(core->CreateShader(source, "VSMain", "vs_6_0", &vertex), 0);
  ASSERT_EQ(core->CreateShader(source, "PSMain", "ps_6_0", &fragment), 0);
  std::unique_ptr<graphics::Program> program;
  core->CreateProgram({}, graphics::IMAGE_FORMAT_UNDEFINED, &program);
  program->BindShader(vertex.get(), graphics::SHADER_TYPE_VERTEX);
  program->BindShader(fragment.get(), graphics::SHADER_TYPE_PIXEL);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);
  program->Finalize();
  std::unique_ptr<graphics::CommandContext> commands;
  core->CreateCommandContext(&commands);
  commands->CmdClearImage(output.get(), {});
  commands->CmdBeginRendering({output.get()}, nullptr);
  commands->CmdSetViewport({0, 0, 5, 4, 0, 1});
  commands->CmdSetScissor({{1, 1}, {3, 2}});
  commands->CmdEndRendering();
  commands->CmdBeginRendering({}, nullptr);
  commands->CmdBindProgram(program.get());
  commands->CmdBindResources(0, {output.get()});
  commands->CmdDraw(3, 1, 0, 0);
  commands->CmdEndRendering();
  core->SubmitCommandContext(commands.get());
  std::vector<glm::vec4> pixels(20);
  output->DownloadData(pixels.data());
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 5; ++x) {
      auto expected = (x >= 1 && x < 4 && y >= 1 && y < 3) ? glm::vec4(2, 3, 4, 1) : glm::vec4(0);
      EXPECT_EQ(pixels[y * 5 + x], expected);
    }
}

TEST_F(MetalBackendTest, BufferCopyResizeAndCompletionCallback) {
  std::unique_ptr<graphics::Buffer> source, target;
  core->CreateBuffer(64, graphics::BUFFER_TYPE_DYNAMIC, &source);
  core->CreateBuffer(64, graphics::BUFFER_TYPE_STATIC, &target);
  std::vector<uint32_t> values(16), actual(16);
  std::iota(values.begin(), values.end(), 1);
  source->UploadData(values.data(), 64);
  std::unique_ptr<graphics::CommandContext> commands;
  core->CreateCommandContext(&commands);
  commands->CmdCopyBuffer(target.get(), source.get(), 64);
  int completed = 0;
  commands->PushPostExecutionCallback([&] { ++completed; });
  ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
  EXPECT_NE(core->SubmitCommandContext(commands.get()), 0);
  target->Resize(128);
  target->DownloadData(actual.data(), 64);
  EXPECT_EQ(actual, values);
  EXPECT_EQ(completed, 1);
  core->WaitGPU();
  EXPECT_EQ(completed, 1);
}
}  // namespace
#endif
