#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "grassland/rhi/rhi.h"

namespace grassland::rhi {
namespace {
struct FormatCase {
  ImageFormat format;
  uint32_t channels, bytes;
};

const FormatCase formats[] = {
    {IMAGE_FORMAT_R8_UINT, 1, 1},          {IMAGE_FORMAT_R8_INT, 1, 1},
    {IMAGE_FORMAT_R8_UNORM, 1, 1},         {IMAGE_FORMAT_R8G8_UINT, 2, 1},
    {IMAGE_FORMAT_R8G8_INT, 2, 1},         {IMAGE_FORMAT_R8G8_UNORM, 2, 1},
    {IMAGE_FORMAT_R8G8B8_UINT, 3, 1},      {IMAGE_FORMAT_R8G8B8_INT, 3, 1},
    {IMAGE_FORMAT_R8G8B8_UNORM, 3, 1},     {IMAGE_FORMAT_R8G8B8A8_UINT, 4, 1},
    {IMAGE_FORMAT_R8G8B8A8_INT, 4, 1},     {IMAGE_FORMAT_R8G8B8A8_UNORM, 4, 1},
    {IMAGE_FORMAT_R32_UINT, 1, 4},         {IMAGE_FORMAT_R32_INT, 1, 4},
    {IMAGE_FORMAT_R32_SFLOAT, 1, 4},       {IMAGE_FORMAT_R32G32_UINT, 2, 4},
    {IMAGE_FORMAT_R32G32_INT, 2, 4},       {IMAGE_FORMAT_R32G32_SFLOAT, 2, 4},
    {IMAGE_FORMAT_R32G32B32_UINT, 3, 4},   {IMAGE_FORMAT_R32G32B32_INT, 3, 4},
    {IMAGE_FORMAT_R32G32B32_SFLOAT, 3, 4}, {IMAGE_FORMAT_R32G32B32A32_UINT, 4, 4},
    {IMAGE_FORMAT_R32G32B32A32_INT, 4, 4}, {IMAGE_FORMAT_R32G32B32A32_SFLOAT, 4, 4},
};
}  // namespace

class RhiFormats : public testing::TestWithParam<BackendAPI> {};

TEST_P(RhiFormats, ColorUploadsAndSubrectangleDownloadsPreserveNativeBytes) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  for (auto c : formats) {
    SCOPED_TRACE(to_string(c.format));
    EXPECT_EQ(ImageFormatSize(c.format), c.channels * c.bytes);
    ObjPtr<IImage> image = device->CreateImage({11, 7, c.format});
    if (c.channels == 3 && !image)
      continue;  // Three-channel textures are not required by every native API.
    ASSERT_TRUE(image);
    const auto pixel = c.channels * c.bytes, pitch = 11 * pixel + 16;
    std::vector<uint8_t> source(pitch * 7);
    // Finite float bit patterns and nontrivial integer channel bytes.
    for (size_t i = 0; i < source.size(); ++i)
      source[i] = uint8_t(i % 61 + 1);
    ObjPtr<ICommandContext> commands = device->CreateCommandContext();
    ASSERT_TRUE(commands->CmdUploadImage(image.Get(), source.data(), pitch));
    ObjPtr<IDownloadToken> token = commands->CmdDownloadImage(image.Get(), {2, 1, 5, 3});
    ASSERT_TRUE(token);
    ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
    ASSERT_TRUE(done && done->WaitHost());
    ASSERT_EQ(token->GetSize(), 5 * 3 * pixel);
    std::vector<uint8_t> actual(token->GetSize());
    ASSERT_TRUE(token->CopyTo(actual.data(), actual.size()));
    for (int y = 0; y < 3; ++y)
      EXPECT_EQ(std::memcmp(actual.data() + y * 5 * pixel, source.data() + (y + 1) * pitch + 2 * pixel, 5 * pixel), 0);
  }
}

TEST_P(RhiFormats, IntegerClearsUseNumericValuesForTransferAndRenderPass) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  for (auto c : formats) {
    const bool signed_integer = IsSignedIntegerFormat(c.format);
    if (!signed_integer && !IsUnsignedIntegerFormat(c.format))
      continue;
    SCOPED_TRACE(to_string(c.format));
    ObjPtr<IImage> image = device->CreateImage({8, 8, c.format});
    if (c.channels == 3 && !image)
      continue;
    ASSERT_TRUE(image);
    ClearValue values;
    if (signed_integer)
      values.color.int32 = c.bytes == 1 ? std::array<int32_t, 4>{-7, 23, -65, 101}
                                        : std::array<int32_t, 4>{-16777217, INT32_MAX, INT32_MIN, 16777217};
    else
      values.color.uint32 = c.bytes == 1 ? std::array<uint32_t, 4>{7, 23, 65, 255}
                                         : std::array<uint32_t, 4>{UINT32_MAX, 16777217u, 0x80000001u, 0x12345678u};
    for (bool render_pass : {false, true}) {
      ObjPtr<ICommandContext> commands = device->CreateCommandContext();
      if (render_pass) {
        ASSERT_TRUE(commands->CmdBeginRendering(
            {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, values}}}));
        ASSERT_TRUE(commands->CmdEndRendering());
      } else {
        ASSERT_TRUE(commands->CmdClearImage(image.Get(), values));
      }
      ObjPtr<IDownloadToken> token = commands->CmdDownloadImage(image.Get(), {3, 2, 1, 1});
      ASSERT_TRUE(token);
      ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
      ASSERT_TRUE(done && done->WaitHost());
      std::array<uint8_t, 16> actual{};
      ASSERT_TRUE(token->CopyTo(actual.data(), actual.size()));
      for (uint32_t i = 0; i < c.channels; ++i) {
        const auto expected = signed_integer ? static_cast<uint32_t>(values.color.int32[i]) : values.color.uint32[i];
        if (c.bytes == 1)
          EXPECT_EQ(actual[i], static_cast<uint8_t>(expected));
        else {
          uint32_t v;
          std::memcpy(&v, actual.data() + 4 * i, 4);
          EXPECT_EQ(v, expected);
        }
      }
    }
  }
}

TEST_P(RhiFormats, FloatClearValuesPreserveFloatAndNormalizedColors) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  for (auto c : formats) {
    if (IsSignedIntegerFormat(c.format) || IsUnsignedIntegerFormat(c.format))
      continue;
    SCOPED_TRACE(to_string(c.format));
    ObjPtr<IImage> image = device->CreateImage({7, 5, c.format});
    if (c.channels == 3 && !image)
      continue;
    ASSERT_TRUE(image);
    const ClearValue value =
        c.bytes == 1 ? ClearValue{0.25f, 0.5f, 0.75f, 1.0f} : ClearValue{-1.25f, 0.5f, 16777216.0f, 4.0f};
    for (bool render_pass : {false, true}) {
      ObjPtr<ICommandContext> commands = device->CreateCommandContext();
      if (render_pass) {
        ASSERT_TRUE(
            commands->CmdBeginRendering({{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, value}}}));
        ASSERT_TRUE(commands->CmdEndRendering());
      } else
        ASSERT_TRUE(commands->CmdClearImage(image.Get(), value));
      ObjPtr<IDownloadToken> token = commands->CmdDownloadImage(image.Get(), {2, 1, 1, 1});
      ASSERT_TRUE(token);
      ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
      ASSERT_TRUE(done && done->WaitHost());
      std::array<uint8_t, 16> pixel{};
      ASSERT_TRUE(token->CopyTo(pixel.data(), pixel.size()));
      for (uint32_t channel = 0; channel < c.channels; ++channel) {
        if (c.bytes == 1)
          EXPECT_NEAR(pixel[channel], value.color.float32[channel] * 255, 0.5f);
        else {
          float actual;
          std::memcpy(&actual, pixel.data() + channel * 4, 4);
          EXPECT_FLOAT_EQ(actual, value.color.float32[channel]);
        }
      }
    }
  }
}

TEST_P(RhiFormats, D16DepthSubrectangleKeepsTwoBytePixels) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IImage> image = device->CreateImage({13, 7, IMAGE_FORMAT_D16_UNORM});
  ASSERT_TRUE(image);
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  RenderingInfo info;
  info.depth_stencil.image = image.Get();
  info.depth_stencil.depth_load_op = ATTACHMENT_LOAD_OP_CLEAR;
  info.depth_stencil.clear_value.depth_stencil.depth = 0.25f;
  ASSERT_TRUE(commands->CmdBeginRendering(info));
  ASSERT_TRUE(commands->CmdEndRendering());
  ObjPtr<IDownloadToken> token = commands->CmdDownloadImage(image.Get(), {3, 2, 5, 3});
  ASSERT_TRUE(token);
  ObjPtr<ISubmissionToken> done = device->GetQueue(QUEUE_TYPE_GRAPHICS)->Submit(commands.Get());
  ASSERT_TRUE(done && done->WaitHost());
  ASSERT_EQ(token->GetSize(), 30);
  std::array<uint16_t, 15> actual{};
  ASSERT_TRUE(token->CopyTo(actual.data(), sizeof(actual)));
  for (auto value : actual)
    EXPECT_NEAR(value, 16384, 1);
}

TEST_P(RhiFormats, FloatSignedAndUnsignedVertexInputsRenderEveryDimension) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IImage> image = device->CreateImage({8, 8, IMAGE_FORMAT_R8G8B8A8_UNORM});
  ASSERT_TRUE(image);
  for (int kind = 0; kind < 3; ++kind) {
    for (uint32_t n = 1; n <= 4; ++n) {
      const auto type = static_cast<InputType>(kind * 4 + n - 1);
      SCOPED_TRACE(to_string(type));
      EXPECT_EQ(InputTypeSize(type), n * 4);
      const std::string scalar = kind == 0 ? "float" : kind == 1 ? "int" : "uint";
      const std::string vector = scalar + (n == 1 ? "" : std::to_string(n));
      std::string expression;
      for (uint32_t j = 0; j < n; ++j) {
        if (j)
          expression += " && ";
        const int value = kind == 1 ? -int(j + 2) : int(j + 2);
        expression +=
            "input.value" + (n == 1 ? std::string{} : "." + std::string(1, "xyzw"[j])) + " == " + std::to_string(value);
      }
      const std::string source = "struct Input { RHI_INPUT(0," + vector + ",value); };" + R"(
        struct Output { float4 position:SV_Position; float ok:TEXCOORD0; };
        Output VSMain(Input input, uint id:SV_VertexID) {
          float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
          Output o; o.position=float4(p[id],0,1); o.ok=()" +
                                 expression + R"() ? 1.0 : 0.0; return o;
        }
        float4 PSMain(Output o):SV_Target0 { return float4(0,o.ok,0,1); }
      )";
      ObjPtr<IShader> vs = device->CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
      ObjPtr<IShader> ps = device->CreateShaderFromSource(source, "PSMain", SHADER_STAGE_FRAGMENT);
      ProgramDesc desc;
      desc.vertex_shader = vs.Get();
      desc.fragment_shader = ps.Get();
      desc.vertex_bindings = {{n * 4, false}};
      desc.vertex_attributes = {{0, 0, type, 0}};
      ObjPtr<IProgram> program = device->CreateProgram(desc);
      ASSERT_TRUE(program);
      std::vector<uint32_t> vertices(n * 3);
      for (uint32_t v = 0; v < 3; ++v)
        for (uint32_t j = 0; j < n; ++j) {
          if (kind == 0) {
            float value = float(j + 2);
            std::memcpy(&vertices[v * n + j], &value, 4);
          } else
            vertices[v * n + j] = kind == 1 ? uint32_t(-int(j + 2)) : j + 2;
        }
      ObjPtr<IBuffer> buffer = device->CreateBuffer({vertices.size() * 4});
      ASSERT_TRUE(buffer);
      ObjPtr<ICommandContext> commands = device->CreateCommandContext();
      ASSERT_TRUE(commands->CmdUploadBuffer(buffer.Get(), 0, vertices.data(), vertices.size() * 4));
      ASSERT_TRUE(commands->CmdBindProgram(program.Get()));
      ASSERT_TRUE(commands->CmdBindVertexBuffer(0, buffer.Get()));
      ASSERT_TRUE(commands->CmdBeginRendering(
          {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}));
      ASSERT_TRUE(commands->CmdSetViewport({0, 0, 8, 8, 0, 1}));
      ASSERT_TRUE(commands->CmdSetScissor({0, 0, 8, 8}));
      ASSERT_TRUE(commands->CmdDraw(3));
      ASSERT_TRUE(commands->CmdEndRendering());
      ObjPtr<IDownloadToken> token = commands->CmdDownloadImage(image.Get(), {4, 4, 1, 1});
      ASSERT_TRUE(token);
      ObjPtr<ISubmissionToken> done = device->GetQueue(QUEUE_TYPE_GRAPHICS)->Submit(commands.Get());
      ASSERT_TRUE(done && done->WaitHost());
      std::array<uint8_t, 4> pixel{};
      ASSERT_TRUE(token->CopyTo(pixel.data(), pixel.size()));
      EXPECT_EQ(pixel, (std::array<uint8_t, 4>{0, 255, 0, 255}));
    }
  }
}

TEST_P(RhiFormats, IntegerStorageWritesAndRenderTargetsPreserveValues) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  for (auto c : formats) {
    const bool signed_integer = IsSignedIntegerFormat(c.format);
    if (!signed_integer && !IsUnsignedIntegerFormat(c.format))
      continue;
    SCOPED_TRACE(to_string(c.format));
    ObjPtr<IImage> image = device->CreateImage({8, 8, c.format});
    if (c.channels == 3 && !image)
      continue;
    ASSERT_TRUE(image);
    const std::string scalar = signed_integer ? "int" : "uint";
    const std::string type = scalar + (c.channels == 1 ? "" : std::to_string(c.channels));
    const std::string literal = signed_integer ? "-17" : c.bytes == 4 ? "0xfedcba98u" : "201u";
    const std::string value = type + "(" + literal + ")";
    const std::string image_format = std::string(c.channels == 1   ? "r"
                                                 : c.channels == 2 ? "rg"
                                                                   : "rgba") +
                                     std::to_string(c.bytes * 8) + (signed_integer ? "i" : "ui");
    const std::string source =
        "[[vk::image_format(\"" + image_format + "\")]] RHI_RW_TEXTURE2D(0,0," + type + ", target);\n" +
        "[numthreads(8,8,1)] void CSMain(uint3 id:SV_DispatchThreadID) { target[id.xy]=" + value + "; }\n" +
        "float4 VSMain(uint id:SV_VertexID):SV_Position { float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[id],0,1); }\n" +
        type + " PSMain():SV_Target0 { return " + value + "; }";
    ObjPtr<IShader> cs = device->CreateShaderFromSource(source, "CSMain", SHADER_STAGE_COMPUTE);
    ComputeProgramDesc cd;
    cd.compute_shader = cs.Get();
    cd.pipeline_layout.groups = {{{{0, BINDING_RW_TEXTURE2D, 1, SHADER_VISIBILITY_COMPUTE}}}};
    ObjPtr<IComputeProgram> compute = device->CreateComputeProgram(cd);
    ASSERT_TRUE(compute);
    ObjPtr<IBindingGroup> group = device->CreateBindingGroup(cd.pipeline_layout.groups[0]);
    ASSERT_TRUE(group && group->BindResource(0, image.Get()));
    ObjPtr<IShader> vs =
        device->CreateShaderFromSource(source.substr(source.find("float4 VSMain")), "VSMain", SHADER_STAGE_VERTEX);
    ObjPtr<IShader> ps =
        device->CreateShaderFromSource(source.substr(source.find("float4 VSMain")), "PSMain", SHADER_STAGE_FRAGMENT);
    ProgramDesc pd;
    pd.vertex_shader = vs.Get();
    pd.fragment_shader = ps.Get();
    pd.color_formats = {c.format};
    ObjPtr<IProgram> graphics = device->CreateProgram(pd);
    ASSERT_TRUE(graphics);
    ClearValue zero;
    if (signed_integer)
      zero.color.int32 = {};
    else
      zero.color.uint32 = {};
    for (bool storage : {true, false}) {
      ObjPtr<ICommandContext> commands = device->CreateCommandContext();
      // Initialize attachment metadata before using a newly allocated image as storage.
      ASSERT_TRUE(commands->CmdClearImage(image.Get(), zero));
      if (storage) {
        ASSERT_TRUE(commands->CmdBindComputeProgram(compute.Get()));
        ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
        ASSERT_TRUE(commands->CmdDispatch(1));
      } else {
        ASSERT_TRUE(commands->CmdBindProgram(graphics.Get()));
        ASSERT_TRUE(
            commands->CmdBeginRendering({{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, zero}}}));
        ASSERT_TRUE(commands->CmdSetViewport({0, 0, 8, 8, 0, 1}));
        ASSERT_TRUE(commands->CmdSetScissor({0, 0, 8, 8}));
        ASSERT_TRUE(commands->CmdDraw(3));
        ASSERT_TRUE(commands->CmdEndRendering());
      }
      ObjPtr<IDownloadToken> token = commands->CmdDownloadImage(image.Get(), {4, 4, 1, 1});
      ASSERT_TRUE(token);
      ObjPtr<ISubmissionToken> done = device->GetQueue(QUEUE_TYPE_GRAPHICS)->Submit(commands.Get());
      ASSERT_TRUE(done && done->WaitHost());
      std::array<uint8_t, 16> actual{};
      ASSERT_TRUE(token->CopyTo(actual.data(), actual.size()));
      for (uint32_t i = 0; i < c.channels; ++i) {
        if (c.bytes == 1)
          EXPECT_EQ(actual[i], signed_integer ? uint8_t(-17) : uint8_t(201));
        else {
          uint32_t v;
          std::memcpy(&v, actual.data() + 4 * i, 4);
          EXPECT_EQ(v, signed_integer ? uint32_t(-17) : 0xfedcba98u);
        }
      }
    }
  }
}

TEST_P(RhiFormats, ClearValueSelectsDepthAndStencilForStandaloneAndRendering) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  for (auto format : {IMAGE_FORMAT_D16_UNORM, IMAGE_FORMAT_D32_SFLOAT, IMAGE_FORMAT_D24_UNORM_S8_UINT}) {
    SCOPED_TRACE(to_string(format));
    ObjPtr<IImage> image = device->CreateImage({7, 5, format});
    ASSERT_TRUE(image);
    for (bool render_pass : {false, true}) {
      ObjPtr<ICommandContext> commands = device->CreateCommandContext();
      ASSERT_TRUE(commands);
      const ClearValue value{ClearDepthStencilValue{0.25f, 0xa5}};
      ASSERT_TRUE(commands->CmdClearImage(image.Get(), ClearDepthStencilValue{0.75f, 0x12}));
      if (render_pass) {
        RenderingInfo info;
        info.depth_stencil.image = image.Get();
        info.depth_stencil.depth_load_op = ATTACHMENT_LOAD_OP_CLEAR;
        info.depth_stencil.stencil_load_op = ATTACHMENT_LOAD_OP_CLEAR;
        info.depth_stencil.clear_value = value;
        ASSERT_TRUE(commands->CmdBeginRendering(info));
        ASSERT_TRUE(commands->CmdEndRendering());
      } else
        ASSERT_TRUE(commands->CmdClearImage(image.Get(), value));
      ObjPtr<IDownloadToken> token = commands->CmdDownloadImage(image.Get(), {2, 1, 3, 2});
      ASSERT_TRUE(token);
      ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
      ASSERT_TRUE(done && done->WaitHost());
      std::vector<uint8_t> pixels(token->GetSize());
      ASSERT_TRUE(token->CopyTo(pixels.data(), pixels.size()));
      for (size_t i = 0; i < 6; ++i) {
        if (format == IMAGE_FORMAT_D16_UNORM) {
          uint16_t depth;
          std::memcpy(&depth, pixels.data() + 2 * i, 2);
          EXPECT_NEAR(depth, 16384, 1);
        } else if (format == IMAGE_FORMAT_D32_SFLOAT) {
          float depth;
          std::memcpy(&depth, pixels.data() + 4 * i, 4);
          EXPECT_FLOAT_EQ(depth, 0.25f);
        } else {
          uint32_t packed;
          std::memcpy(&packed, pixels.data() + 4 * i, 4);
          EXPECT_NEAR(packed & 0xffffff, 4194304, 1);
          EXPECT_EQ(packed >> 24, 0xa5u);
        }
      }
    }
  }
}

TEST_P(RhiFormats, IntegerClearsPreserveBoundDescriptorTablesAndOrderWrites) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IImage> image = device->CreateImage({4, 4, IMAGE_FORMAT_R32_UINT});
  ASSERT_TRUE(image);
  ObjPtr<IShader> shader = device->CreateShaderFromSource(R"(
    RHI_RW_TEXTURE2D(0, 0, uint, target);
    [numthreads(4, 4, 1)] void CSMain(uint3 id:SV_DispatchThreadID) {
      target[id.xy] += 1u;
    }
  )",
                                                          "CSMain", SHADER_STAGE_COMPUTE);
  ASSERT_TRUE(shader);
  ComputeProgramDesc desc;
  desc.compute_shader = shader.Get();
  desc.pipeline_layout.groups = {{{{0, BINDING_RW_TEXTURE2D, 1, SHADER_VISIBILITY_COMPUTE}}}};
  ObjPtr<IComputeProgram> program = device->CreateComputeProgram(desc);
  ASSERT_TRUE(program);
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ASSERT_TRUE(group && group->BindResource(0, image.Get()));
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands->CmdBindComputeProgram(program.Get()));
  ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
  std::array<ObjPtr<IDownloadToken>, 4> tokens;
  for (uint32_t i = 0; i < tokens.size(); ++i) {
    ClearValue value;
    value.color.uint32 = {0x80000001u + i * 4, 0, 0, 0};
    if (i % 2) {
      ASSERT_TRUE(
          commands->CmdBeginRendering({{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, value}}}));
      ASSERT_TRUE(commands->CmdEndRendering());
    } else
      ASSERT_TRUE(commands->CmdClearImage(image.Get(), value));
    // Keep the same program/group bound across the clear, including its heap changes.
    ASSERT_TRUE(commands->CmdDispatch(1));
    tokens[i] = commands->CmdDownloadImage(image.Get());
    ASSERT_TRUE(tokens[i]);
  }
  ObjPtr<ISubmissionToken> done = device->GetQueue(QUEUE_TYPE_GRAPHICS)->Submit(commands.Get());
  ASSERT_TRUE(done && done->WaitHost());
  for (uint32_t i = 0; i < tokens.size(); ++i) {
    std::array<uint32_t, 16> pixels{};
    ASSERT_TRUE(tokens[i]->CopyTo(pixels.data(), sizeof(pixels)));
    for (auto pixel : pixels)
      EXPECT_EQ(pixel, 0x80000002u + i * 4);
  }
}

const BackendAPI format_backends[] = {
#if defined(LONGMARCH_D3D12_ENABLED)
    BACKEND_API_D3D12,
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    BACKEND_API_VULKAN,
#endif
#if defined(LONGMARCH_METAL_ENABLED)
    BACKEND_API_METAL,
#endif
};
INSTANTIATE_TEST_SUITE_P(Backends, RhiFormats, testing::ValuesIn(format_backends));
}  // namespace grassland::rhi
