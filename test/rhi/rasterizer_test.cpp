#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "grassland/rhi/rhi.h"
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#endif

namespace grassland::rhi {
class RhiRasterizer : public testing::TestWithParam<BackendAPI> {
 protected:
  ObjPtr<ICore> core;
  ObjPtr<IDevice> device;
  ObjPtr<IShader> vs, ps;

  void SetUp() override {
    core = CreateCore(GetParam(), {true});
    ASSERT_TRUE(core);
    device = core->CreateDeviceByAutoSelection();
    ASSERT_TRUE(device);
    const std::string source = R"(
      struct Input { RHI_INPUT(0,float2,position); };
      struct Output { float4 position:SV_Position; float pointSize:PSIZE; };
      Output VSMain(Input i) { Output o; o.position=float4(i.position,0,1); o.pointSize=1; return o; }
      float4 PSMain(bool front:SV_IsFrontFace):SV_Target0 { return front ? float4(0,1,0,1) : float4(1,0,0,1); }
    )";
    vs = device->CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
    ps = device->CreateShaderFromSource(source, "PSMain", SHADER_STAGE_FRAGMENT);
    ASSERT_TRUE(vs && ps);
  }

  ProgramDesc Description() {
    ProgramDesc desc;
    desc.vertex_shader = vs.Get();
    desc.fragment_shader = ps.Get();
    desc.vertex_bindings = {{8, false}};
    desc.vertex_attributes = {{0, 0, INPUT_TYPE_FLOAT2, 0}};
    return desc;
  }

  bool SupportsWireframe() {
#if defined(LONGMARCH_VULKAN_ENABLED)
    if (GetParam() == BACKEND_API_VULKAN) {
      VkPhysicalDeviceFeatures features{};
      vkGetPhysicalDeviceFeatures(static_cast<backend::vulkan::VulkanDevice *>(device.Get())->NativePhysicalDevice(),
                                  &features);
      return features.fillModeNonSolid;
    }
#endif
    return true;
  }

  void Render(const ProgramDesc &desc,
              const std::vector<std::array<float, 2>> &vertices,
              bool indexed,
              std::vector<uint8_t> &pixels) {
    ObjPtr<IProgram> program = device->CreateProgram(desc);
    ASSERT_TRUE(program);
    ObjPtr<IImage> image = device->CreateImage({64, 64, IMAGE_FORMAT_R8G8B8A8_UNORM});
    ObjPtr<IBuffer> buffer = device->CreateBuffer({vertices.size() * 8});
    ObjPtr<IBuffer> indices = device->CreateBuffer({vertices.size() * 4});
    ASSERT_TRUE(image && buffer && indices);
    ObjPtr<ICommandContext> commands = device->CreateCommandContext();
    ASSERT_TRUE(commands);
    ASSERT_TRUE(commands->CmdUploadBuffer(buffer.Get(), 0, vertices.data(), vertices.size() * 8));
    std::vector<uint32_t> index_data(vertices.size());
    for (uint32_t i = 0; i < index_data.size(); ++i)
      index_data[i] = i;
    ASSERT_TRUE(commands->CmdUploadBuffer(indices.Get(), 0, index_data.data(), index_data.size() * 4));
    ASSERT_TRUE(commands->CmdBindProgram(program.Get()));
    ASSERT_TRUE(commands->CmdBindVertexBuffer(0, buffer.Get()));
    ASSERT_TRUE(commands->CmdBindIndexBuffer(indices.Get()));
    ASSERT_TRUE(commands->CmdSetViewport({12, 8, 40, 40, 0, 1}));
    ASSERT_TRUE(commands->CmdSetScissor({0, 0, 64, 64}));
    // Binding before BeginRendering and a second rendering scope must both apply
    // the same program's rasterizer state (Metal encoder state is per scope).
    for (int pass = 0; pass < 2; ++pass) {
      ASSERT_TRUE(commands->CmdBeginRendering(
          {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}));
      if (indexed)
        ASSERT_TRUE(commands->CmdDrawIndexed(uint32_t(vertices.size())));
      else
        ASSERT_TRUE(commands->CmdDraw(uint32_t(vertices.size())));
      ASSERT_TRUE(commands->CmdEndRendering());
    }
    ObjPtr<IDownloadToken> token = commands->CmdDownloadImage(image.Get());
    ASSERT_TRUE(token);
    ObjPtr<ISubmissionToken> done = device->GetQueue(QUEUE_TYPE_GRAPHICS)->Submit(commands.Get());
    ASSERT_TRUE(done && done->WaitHost());
    pixels.resize(token->GetSize());
    ASSERT_TRUE(token->CopyTo(pixels.data(), pixels.size()));
  }
};

TEST_P(RhiRasterizer, WindingCullAndViewportAgreeWithGraphicsConvention) {
  const std::vector<std::array<float, 2>> ccw{{-.8f, -.8f}, {.8f, -.8f}, {0, .8f}};
  for (bool indexed : {false, true})
    for (bool reverse : {false, true})
      for (auto face : {FRONT_FACE_COUNTER_CLOCKWISE, FRONT_FACE_CLOCKWISE})
        for (auto cull : {CULL_MODE_NONE, CULL_MODE_FRONT, CULL_MODE_BACK}) {
          SCOPED_TRACE(std::to_string(indexed) + "/" + std::to_string(reverse) + "/" + std::to_string(face) + "/" +
                       std::to_string(cull));
          auto vertices = ccw;
          if (reverse)
            std::reverse(vertices.begin(), vertices.end());
          auto desc = Description();
          desc.rasterizer.front_face = face;
          desc.rasterizer.cull_mode = cull;
          std::vector<uint8_t> pixels;
          Render(desc, vertices, indexed, pixels);
          ASSERT_EQ(pixels.size(), 64 * 64 * 4);
          const bool front = (face == FRONT_FACE_COUNTER_CLOCKWISE) != reverse;
          const bool visible =
              cull == CULL_MODE_NONE || (cull == CULL_MODE_BACK && front) || (cull == CULL_MODE_FRONT && !front);
          const auto at = [&](int x, int y, int c) { return pixels[(y * 64 + x) * 4 + c]; };
          EXPECT_EQ(at(32, 28, 0), visible && !front ? 255 : 0);
          EXPECT_EQ(at(32, 28, 1), visible && front ? 255 : 0);
          // NDC +Y is at the top. The triangle is narrow at the top and wide below.
          EXPECT_EQ(at(19, 14, 0) + at(19, 14, 1), 0);
          EXPECT_EQ(at(19, 42, 0) + at(19, 42, 1), visible ? 255 : 0);
          EXPECT_EQ(at(4, 4, 0) + at(4, 4, 1), 0);
        }
}

TEST_P(RhiRasterizer, AllTopologiesSupportIndexedAndNonIndexedDraws) {
  for (auto topology : {PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
                        PRIMITIVE_TOPOLOGY_LINE_LIST, PRIMITIVE_TOPOLOGY_LINE_STRIP, PRIMITIVE_TOPOLOGY_POINT_LIST})
    for (bool indexed : {false, true}) {
      SCOPED_TRACE(std::to_string(topology) + "/" + std::to_string(indexed));
      std::vector<std::array<float, 2>> vertices{{-.7f, -.7f}, {.7f, -.7f}, {0, .7f}};
      if (topology == PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
        vertices = {{-.7f, -.7f}, {.7f, -.7f}, {-.7f, .7f}, {.7f, .7f}};
      auto desc = Description();
      desc.topology = topology;
      std::vector<uint8_t> pixels;
      Render(desc, vertices, indexed, pixels);
      ASSERT_EQ(pixels.size(), 64 * 64 * 4);
      size_t covered = 0;
      for (size_t i = 0; i < pixels.size(); i += 4)
        covered += pixels[i] != 0 || pixels[i + 1] != 0;
      if (topology == PRIMITIVE_TOPOLOGY_POINT_LIST)
        EXPECT_EQ(covered, vertices.size());
      else if (topology == PRIMITIVE_TOPOLOGY_LINE_LIST || topology == PRIMITIVE_TOPOLOGY_LINE_STRIP) {
        EXPECT_GT(covered, 10);
        EXPECT_LT(covered, 100);
      } else
        EXPECT_GT(covered, 200);
    }
}

TEST_P(RhiRasterizer, WireframeKeepsEdgesAndOmitsTriangleInterior) {
  auto desc = Description();
  desc.rasterizer.fill_mode = FILL_MODE_WIREFRAME;
  if (!SupportsWireframe()) {
    EXPECT_FALSE(ObjPtr<IProgram>(device->CreateProgram(desc)));
    return;
  }
  for (bool indexed : {false, true}) {
    std::vector<uint8_t> pixels;
    Render(desc, {{-.8f, -.8f}, {.8f, -.8f}, {0, .8f}}, indexed, pixels);
    ASSERT_EQ(pixels.size(), 64 * 64 * 4);
    EXPECT_EQ(pixels[(28 * 64 + 32) * 4] + pixels[(28 * 64 + 32) * 4 + 1], 0);
    size_t covered = 0;
    for (size_t i = 0; i < pixels.size(); i += 4)
      covered += pixels[i] != 0 || pixels[i + 1] != 0;
    EXPECT_GT(covered, 40);
    EXPECT_LT(covered, 180);
  }
}

TEST_P(RhiRasterizer, InvalidRasterizerEnumsAreRejected) {
  auto desc = Description();
  desc.topology = static_cast<PrimitiveTopology>(99);
  EXPECT_FALSE(ObjPtr<IProgram>(device->CreateProgram(desc)));
  desc = Description();
  desc.rasterizer.cull_mode = static_cast<CullMode>(99);
  EXPECT_FALSE(ObjPtr<IProgram>(device->CreateProgram(desc)));
  desc = Description();
  desc.rasterizer.front_face = static_cast<FrontFace>(99);
  EXPECT_FALSE(ObjPtr<IProgram>(device->CreateProgram(desc)));
  desc = Description();
  desc.rasterizer.fill_mode = static_cast<FillMode>(99);
  EXPECT_FALSE(ObjPtr<IProgram>(device->CreateProgram(desc)));
}

TEST_P(RhiRasterizer, SwitchingProgramsReappliesRasterizerStateInsideOneScope) {
  if (!SupportsWireframe())
    GTEST_SKIP();
  auto desc = Description();
  ObjPtr<IProgram> solid = device->CreateProgram(desc);
  desc.rasterizer.front_face = FRONT_FACE_CLOCKWISE;
  desc.rasterizer.cull_mode = CULL_MODE_BACK;
  ObjPtr<IProgram> culled = device->CreateProgram(desc);
  desc.rasterizer.cull_mode = CULL_MODE_FRONT;
  ObjPtr<IProgram> back = device->CreateProgram(desc);
  desc = Description();
  desc.rasterizer.fill_mode = FILL_MODE_WIREFRAME;
  ObjPtr<IProgram> wire = device->CreateProgram(desc);
  ASSERT_TRUE(solid && culled && back && wire);
  ObjPtr<IImage> image = device->CreateImage({64, 64, IMAGE_FORMAT_R8G8B8A8_UNORM});
  const float vertices[] = {-.8f, -.8f, .8f, -.8f, 0, .8f};
  ObjPtr<IBuffer> buffer = device->CreateBuffer({sizeof(vertices)});
  ASSERT_TRUE(image && buffer);
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands);
  ASSERT_TRUE(commands->CmdUploadBuffer(buffer.Get(), 0, vertices, sizeof(vertices)));
  ASSERT_TRUE(commands->CmdBeginRendering(
      {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}));
  const std::array<IProgram *, 4> programs{solid.Get(), culled.Get(), back.Get(), wire.Get()};
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(commands->CmdBindProgram(programs[i]));
    ASSERT_TRUE(commands->CmdBindVertexBuffer(0, buffer.Get()));
    ASSERT_TRUE(commands->CmdSetViewport({float(i % 2 * 32), float(i / 2 * 32), 32, 32, 0, 1}));
    ASSERT_TRUE(commands->CmdSetScissor({i % 2 * 32, i / 2 * 32, 32, 32}));
    ASSERT_TRUE(commands->CmdDraw(3));
  }
  ASSERT_TRUE(commands->CmdEndRendering());
  ObjPtr<IDownloadToken> token = commands->CmdDownloadImage(image.Get());
  ASSERT_TRUE(token);
  ObjPtr<ISubmissionToken> done = device->GetQueue(QUEUE_TYPE_GRAPHICS)->Submit(commands.Get());
  ASSERT_TRUE(done && done->WaitHost());
  std::vector<uint8_t> pixels(token->GetSize());
  ASSERT_TRUE(token->CopyTo(pixels.data(), pixels.size()));
  const std::array<std::array<uint8_t, 2>, 4> expected{{{0, 255}, {0, 0}, {255, 0}, {0, 0}}};
  for (int i = 0; i < 4; ++i) {
    const size_t offset = ((i / 2 * 32 + 16) * 64 + i % 2 * 32 + 16) * 4;
    EXPECT_EQ(pixels[offset], expected[i][0]);
    EXPECT_EQ(pixels[offset + 1], expected[i][1]);
  }
  size_t edges = 0;
  for (int y = 32; y < 64; ++y)
    for (int x = 32; x < 64; ++x)
      edges += pixels[(y * 64 + x) * 4 + 1] != 0;
  EXPECT_GT(edges, 30);
  EXPECT_LT(edges, 150);
}

#if !defined(NDEBUG)
TEST_P(RhiRasterizer, ViewportRequiresPositiveExtents) {
  for (auto extent : {std::array<float, 2>{0, 16}, {16, 0}, {-16, 16}, {16, -16}}) {
    ObjPtr<ICommandContext> commands = device->CreateCommandContext();
    ASSERT_TRUE(commands);
    EXPECT_FALSE(commands->CmdSetViewport({0, 0, extent[0], extent[1], 0, 1}));
    EXPECT_FALSE(ObjPtr<ISubmissionToken>(device->GetQueue(QUEUE_TYPE_GRAPHICS)->Submit(commands.Get())));
  }
}
#endif

const BackendAPI backends[] = {
#if defined(LONGMARCH_VULKAN_ENABLED)
    BACKEND_API_VULKAN,
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
    BACKEND_API_D3D12,
#endif
#if defined(LONGMARCH_METAL_ENABLED)
    BACKEND_API_METAL,
#endif
};
INSTANTIATE_TEST_SUITE_P(Backends, RhiRasterizer, testing::ValuesIn(backends));
}  // namespace grassland::rhi
