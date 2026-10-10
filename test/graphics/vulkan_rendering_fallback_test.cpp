#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_program.h"
#include "grassland/graphics/graphics.h"

namespace {
using namespace grassland::graphics;
using namespace grassland::graphics::backend;

class VulkanRenderingFallback : public testing::TestWithParam<int> {};

TEST_P(VulkanRenderingFallback, FrozenProgramsRenderWithBothCapabilityPaths) {
  VulkanCore core({1, true}, {bool(GetParam() & 1), bool(GetParam() & 2)});
  ASSERT_EQ(core.InitializeLogicalDeviceAutoSelect(false), 0);
  RecordProperty("dynamic_rendering", core.DynamicRenderingEnabled());
  RecordProperty("dynamic_topology", core.DynamicTopologyEnabled());
  if (!(GetParam() & 1)) {
    EXPECT_FALSE(core.DynamicRenderingEnabled());
    EXPECT_EQ(core.InstanceProcedures().vkCmdBeginRenderingKHR, nullptr);
  }
  if (!(GetParam() & 2)) {
    EXPECT_FALSE(core.DynamicTopologyEnabled());
    EXPECT_EQ(core.InstanceProcedures().vkCmdSetPrimitiveTopologyEXT, nullptr);
  }
  std::unique_ptr<Shader> vs, ps;
  ASSERT_EQ(core.CreateShader(R"(
struct Output { float4 position : SV_Position; float pointSize : SV_PointSize; };
Output Main(uint id : SV_VertexID) {
  float2 positions[3] = { float2(-1,-1), float2(3,-1), float2(-1,3) };
  Output result; result.position = float4(positions[id % 3], 0.5, 1); result.pointSize = 1; return result;
})",
                              "Main", "vs_6_0", &vs),
            0);
  ASSERT_EQ(core.CreateShader("float4 Main() : SV_Target { return float4(1,0,0,1); }", "Main", "ps_6_0", &ps), 0);
  std::unique_ptr<Program> program;
  ASSERT_EQ(core.CreateProgram({IMAGE_FORMAT_R32G32B32A32_SFLOAT}, IMAGE_FORMAT_D32_SFLOAT, &program), 0);
  program->SetCullMode(CULL_MODE_NONE);
  program->BindShader(vs.get(), SHADER_TYPE_VERTEX);
  program->BindShader(ps.get(), SHADER_TYPE_PIXEL);
  program->Finalize();
  EXPECT_THROW(program->SetCullMode(CULL_MODE_BACK), std::logic_error);
  EXPECT_THROW(program->AddResourceBinding(RESOURCE_TYPE_UNIFORM_BUFFER, 1), std::logic_error);
  EXPECT_THROW(program->Finalize(), std::logic_error);
  std::unique_ptr<Shader> green_ps;
  ASSERT_EQ(core.CreateShader("float4 Main() : SV_Target { return float4(0,1,0,1); }", "Main", "ps_6_0", &green_ps), 0);
  std::unique_ptr<Program> green;
  ASSERT_EQ(core.CreateProgram({IMAGE_FORMAT_R32G32B32A32_SFLOAT}, IMAGE_FORMAT_D32_SFLOAT, &green), 0);
  green->SetCullMode(CULL_MODE_NONE);
  green->BindShader(vs.get(), SHADER_TYPE_VERTEX);
  green->BindShader(green_ps.get(), SHADER_TYPE_PIXEL);
  green->Finalize();
  green_ps.reset();
  // No application Shader remains when lazy PSOs are first constructed.
  vs.reset();
  ps.reset();
  auto *native = dynamic_cast<VulkanProgram *>(program.get());
  const auto triangles = native->PipelineForTopology(PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  EXPECT_EQ(triangles, native->PipelineForTopology(PRIMITIVE_TOPOLOGY_TRIANGLE_LIST));
  const auto strip = native->PipelineForTopology(PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);
  if (core.DynamicTopologyEnabled())
    EXPECT_EQ(triangles, strip);
  else
    EXPECT_NE(triangles, strip);
  EXPECT_NE(triangles, native->PipelineForTopology(PRIMITIVE_TOPOLOGY_LINE_LIST));

  // Recreate attachments at different sizes; reuse Programs across render passes
  // with CLEAR and LOAD, and exercise framebuffer invalidation on image deletion.
  for (int size : {8, 16, 8}) {
    std::unique_ptr<Image> color, depth;
    ASSERT_EQ(core.CreateImage(size, size, IMAGE_FORMAT_R32G32B32A32_SFLOAT, &color), 0);
    ASSERT_EQ(core.CreateImage(size, size, IMAGE_FORMAT_D32_SFLOAT, &depth), 0);
    std::unique_ptr<Buffer> indices;
    ASSERT_EQ(core.CreateBuffer(12, BUFFER_TYPE_STATIC, &indices), 0);
    const std::array<uint32_t, 3> index_data{0, 1, 2};
    indices->UploadData(index_data.data(), sizeof(index_data));
    std::unique_ptr<CommandContext> commands;
    core.CreateCommandContext(&commands);
    commands->CmdBindProgram(program.get());
    commands->CmdSetPrimitiveTopology(PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    ClearValue clear{};
    clear.color = {0, 0, 1, 1};
    commands->CmdClearImage(color.get(), clear);
    clear.depth = {1};
    commands->CmdClearImage(depth.get(), clear);
    commands->CmdBeginRendering({color.get()}, depth.get());
    commands->CmdSetViewport({0, 0, float(size), float(size), 0, 1});
    commands->CmdSetScissor({{0, 0}, {uint32_t(size), uint32_t(size)}});
    commands->CmdDraw(3, 1, 0, 0);
    commands->CmdEndRendering();
    commands->CmdBeginRendering({color.get()}, depth.get());
    // Equal-depth green fragments must fail LESS if the previous depth was preserved.
    commands->CmdBindProgram(green.get());
    commands->CmdBindIndexBuffer(indices.get(), 0);
    for (auto topology : {PRIMITIVE_TOPOLOGY_LINE_LIST, PRIMITIVE_TOPOLOGY_POINT_LIST,
                          PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, PRIMITIVE_TOPOLOGY_TRIANGLE_LIST}) {
      commands->CmdSetPrimitiveTopology(topology);
      commands->CmdDrawIndexed(3, 1, 0, 0, 0);
    }
    commands->CmdEndRendering();
    for (int replay = 0; replay != 2; ++replay) {
      ASSERT_EQ(core.SubmitCommandContext(commands.get()), 0);
      core.WaitGPU();
      std::vector<std::array<float, 4>> pixels(size * size);
      color->DownloadData(pixels.data());
      for (const auto &pixel : pixels)
        EXPECT_EQ(pixel, (std::array<float, 4>{1, 0, 0, 1}));
    }
  }
}

TEST_P(VulkanRenderingFallback, ClearAndLoadAcrossMultipleAndDepthOnlyAttachments) {
  VulkanCore core({1, true}, {bool(GetParam() & 1), bool(GetParam() & 2)});
  ASSERT_EQ(core.InitializeLogicalDeviceAutoSelect(false), 0);
  std::unique_ptr<Image> a, b, depth;
  ASSERT_EQ(core.CreateImage(4, 4, IMAGE_FORMAT_R32G32B32A32_SFLOAT, &a), 0);
  ASSERT_EQ(core.CreateImage(4, 4, IMAGE_FORMAT_R32G32B32A32_SFLOAT, &b), 0);
  ASSERT_EQ(core.CreateImage(4, 4, IMAGE_FORMAT_D32_SFLOAT, &depth), 0);
  std::unique_ptr<CommandContext> commands;
  core.CreateCommandContext(&commands);
  ClearValue clear{};
  clear.color = {1, 0, 0, 1};
  commands->CmdClearImage(a.get(), clear);
  clear.color = {0, 1, 0, 1};
  commands->CmdClearImage(b.get(), clear);
  commands->CmdBeginRendering({a.get(), b.get()}, nullptr);
  commands->CmdEndRendering();
  clear.depth = {0.25f};
  commands->CmdClearImage(depth.get(), clear);
  commands->CmdBeginRendering({}, depth.get());
  commands->CmdEndRendering();
  clear.color = {0, 0, 1, 1};
  commands->CmdClearImage(b.get(), clear);
  commands->CmdBeginRendering({a.get(), b.get()}, depth.get());
  commands->CmdEndRendering();
  ASSERT_EQ(core.SubmitCommandContext(commands.get()), 0);
  core.WaitGPU();
  std::array<std::array<float, 4>, 16> pixels;
  a->DownloadData(pixels.data());
  for (const auto &pixel : pixels)
    EXPECT_EQ(pixel, (std::array<float, 4>{1, 0, 0, 1}));
  b->DownloadData(pixels.data());
  for (const auto &pixel : pixels)
    EXPECT_EQ(pixel, (std::array<float, 4>{0, 0, 1, 1}));
  std::array<float, 16> depths;
  depth->DownloadData(depths.data());
  for (float value : depths)
    EXPECT_EQ(value, 0.25f);
}

INSTANTIATE_TEST_SUITE_P(Capabilities, VulkanRenderingFallback, testing::Values(0, 1, 2, 3));
}  // namespace
