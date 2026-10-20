#include <gtest/gtest.h>

#include <cstdlib>
#if defined(LONGMARCH_D3D12_ENABLED)
#include <d3d12sdklayers.h>

#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#endif

#include "grassland/rhi/rhi.h"
#include "grassland/rhi/rhix/rhix.h"

#if defined(LONGMARCH_D3D12_ENABLED) || defined(LONGMARCH_VULKAN_ENABLED) || defined(LONGMARCH_METAL_ENABLED)
namespace grassland::rhi {
class RhiProgram : public testing::TestWithParam<BackendAPI> {};

TEST_P(RhiProgram, DeferredCompilationOwnsSourcesAndUsesInputMacros) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ShaderDesc source;
  source.source =
      "\xef\xbb\xbf"
      R"(
 struct Input { RHI_INPUT(0,float3,position); RHI_INPUT(1,float3,color); };
 struct Varying { float4 position : SV_Position; float3 color : COLOR0; };
 Varying VSMain(Input input) { Varying output; output.position=float4(input.position,1); output.color=input.color; return output; }
 float4 PSMain(Varying input) : SV_Target0 { return float4(input.color,1); }
 )";
  source.entry_point = "VSMain";
  ObjPtr<IShader> vertex = device->CreateShader(source);
  source.entry_point = "PSMain";
  source.stage = SHADER_STAGE_FRAGMENT;
  ObjPtr<IShader> fragment = device->CreateShader(source);
  ASSERT_TRUE(vertex);
  ASSERT_TRUE(fragment);
  ProgramDesc desc;
  desc.vertex_shader = vertex.Get();
  desc.fragment_shader = fragment.Get();
  desc.vertex_bindings = {{24, false}};
  desc.vertex_attributes = {{0, 0, VERTEX_FORMAT_FLOAT3, 0}, {1, 0, VERTEX_FORMAT_FLOAT3, 12}};
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  EXPECT_FALSE(program->IsFinalized());
  vertex.Release();
  fragment.Release();
  ASSERT_TRUE(program->Finalize());
  EXPECT_TRUE(program->IsFinalized());
  EXPECT_TRUE(program->Finalize());
}

TEST_P(RhiProgram, InvalidShaderIsRejectedAtFinalizeNotSourceCreation) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IShader> vertex = device->CreateShader({"invalid shader code", "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> fragment = device->CreateShader({"invalid shader code", "PSMain", SHADER_STAGE_FRAGMENT});
  ASSERT_TRUE(vertex);
  ASSERT_TRUE(fragment);
  ProgramDesc desc;
  desc.vertex_shader = vertex.Get();
  desc.fragment_shader = fragment.Get();
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  EXPECT_FALSE(program->Finalize());
  EXPECT_FALSE(program->IsFinalized());
}

TEST_P(RhiProgram, DrawAndIndexedDrawRetainResourcesUntilCompletion) {
  if (!std::getenv("LONGMARCH_RHI_WINDOW_TESTS"))
    GTEST_SKIP();
#if defined(LONGMARCH_D3D12_ENABLED)
  backend::d3d12::ComPtr<ID3D12Debug> debug;
  if (GetParam() == BACKEND_API_D3D12) {
    ASSERT_TRUE(SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))));
    debug->EnableDebugLayer();
  }
#endif
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
#if defined(LONGMARCH_D3D12_ENABLED)
  backend::d3d12::ComPtr<ID3D12InfoQueue> messages;
  if (GetParam() == BACKEND_API_D3D12) {
    ASSERT_TRUE(SUCCEEDED(static_cast<backend::d3d12::D3D12Device *>(device.Get())
                              ->NativeDevice()
                              ->QueryInterface(IID_PPV_ARGS(&messages))));
  }
#endif
  ObjPtr<IWindow> window = device->CreateGLFWWindow({320, 240, "RHI draw regression"});
  ASSERT_TRUE(window);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(queue);
  const std::string source = R"(
    struct Input { RHI_INPUT(0,float3,position); };
    float4 VSMain(Input input) : SV_Position { return float4(input.position,1); }
    float4 PSMain() : SV_Target0 { return float4(1,0,0,1); }
  )";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ASSERT_TRUE(vs);
  ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
  ASSERT_TRUE(ps);
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.vertex_bindings = {{12, false}};
  desc.vertex_attributes = {{0, 0, VERTEX_FORMAT_FLOAT3, 0}};
  {
    ObjPtr<IImage> frame = window->GetNextImage();
    ASSERT_TRUE(frame);
    desc.color_format = frame->GetDesc().format;
  }
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  ASSERT_TRUE(program->Finalize());
  vs.Release();
  ps.Release();
  const float vertices[] = {0, 0.5f, 0, -0.5f, -0.5f, 0, 0.5f, -0.5f, 0};
  const uint32_t indices[] = {0, 1, 2};
  ObjPtr<IBuffer> geometry = device->CreateBuffer({sizeof(vertices) + sizeof(indices)});
  ASSERT_TRUE(geometry);
  ObjPtr<ISubmissionToken> done;
  for (int frame = 0; frame < 12; ++frame) {
    ASSERT_TRUE(PollEvents());
    ObjPtr<IImage> image = window->GetNextImage();
    ASSERT_TRUE(image);
    ObjPtr<ICommandContext> commands = device->CreateCommandContext();
    ASSERT_TRUE(commands);
    ASSERT_TRUE(commands->CmdBindProgram(program.Get()));
    ASSERT_TRUE(commands->CmdBindVertexBuffer(0, geometry.Get()));
    ASSERT_TRUE(commands->CmdBindIndexBuffer(geometry.Get(), INDEX_FORMAT_UINT32, sizeof(vertices)));
    // Upload after binding exercises restoration of bound resources' draw usages.
    ASSERT_TRUE(commands->CmdUploadBuffer(geometry.Get(), 0, vertices, sizeof(vertices)));
    ASSERT_TRUE(commands->CmdUploadBuffer(geometry.Get(), sizeof(vertices), indices, sizeof(indices)));
    ASSERT_TRUE(commands->CmdSetViewport({0, 0, 320, 240, 0, 1}));
    ASSERT_TRUE(commands->CmdSetScissor({0, 0, 320, 240}));
    ASSERT_TRUE(commands->CmdBeginRendering(
        {image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0.6f, 0.7f, 0.8f, 1}}));
    ASSERT_TRUE(commands->CmdDraw(3));
    ASSERT_TRUE(commands->CmdEndRendering());
    // A LOAD pass is compatible with the same Program and framebuffer as CLEAR.
    ASSERT_TRUE(commands->CmdBeginRendering({image.Get()}));
    // Rebinding inside a scope must not split it or emit illegal Vulkan barriers.
    ASSERT_TRUE(commands->CmdBindVertexBuffer(0, geometry.Get()));
    ASSERT_TRUE(commands->CmdBindIndexBuffer(geometry.Get(), INDEX_FORMAT_UINT32, sizeof(vertices)));
    ASSERT_TRUE(commands->CmdDrawIndexed(3));
    ASSERT_TRUE(commands->CmdEndRendering());
    ASSERT_TRUE(commands->CmdPresent(window.Get()));
    if (frame == 11) {
      program.Release();
      geometry.Release();
      image.Release();
    }
    done = queue->Submit(commands.Get());
    ASSERT_TRUE(done);
  }
  ASSERT_TRUE(done->WaitHost());
#if defined(LONGMARCH_D3D12_ENABLED)
  if (messages) {
    for (UINT64 i = 0; i < messages->GetNumStoredMessages(); ++i) {
      SIZE_T size{};
      ASSERT_TRUE(SUCCEEDED(messages->GetMessage(i, nullptr, &size)));
      std::vector<uint8_t> storage(size);
      auto message = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
      ASSERT_TRUE(SUCCEEDED(messages->GetMessage(i, message, &size)));
      EXPECT_NE(message->Severity, D3D12_MESSAGE_SEVERITY_ERROR) << message->pDescription;
      EXPECT_NE(message->Severity, D3D12_MESSAGE_SEVERITY_CORRUPTION) << message->pDescription;
    }
  }
#endif
}

TEST_P(RhiProgram, ExplicitRenderingSupportsEmptyScopesAndAttachmentOperations) {
  if (!std::getenv("LONGMARCH_RHI_WINDOW_TESTS"))
    GTEST_SKIP();
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IWindow> window = device->CreateGLFWWindow({160, 120, "RHI rendering scopes"});
  ASSERT_TRUE(window);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(queue);
  ObjPtr<IImage> image = window->GetNextImage();
  ASSERT_TRUE(image);
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands);
  for (auto load : {ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_LOAD_OP_LOAD, ATTACHMENT_LOAD_OP_DISCARD}) {
    for (auto store : {ATTACHMENT_STORE_OP_STORE, ATTACHMENT_STORE_OP_DISCARD}) {
      // Initialize before LOAD, including after a previous DISCARD store.
      ASSERT_TRUE(commands->CmdClearColor(image.Get(), {0, 0, 0, 1}));
      ASSERT_TRUE(commands->CmdBeginRendering({image.Get(), load, store, {0.6f, 0.7f, 0.8f, 1}}));
      ASSERT_TRUE(commands->CmdEndRendering());
    }
  }
  ASSERT_TRUE(commands->CmdBeginRendering(
      {image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0.6f, 0.7f, 0.8f, 1}}));
  ASSERT_TRUE(commands->CmdEndRendering());
  ASSERT_TRUE(commands->CmdPresent(window.Get()));
  ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
  ASSERT_TRUE(done);
  ASSERT_TRUE(done->WaitHost());
}

#if !defined(NDEBUG)
TEST_P(RhiProgram, ExplicitRenderingRejectsUnbalancedScopesAndOutsideCommands) {
  if (!std::getenv("LONGMARCH_RHI_WINDOW_TESTS"))
    GTEST_SKIP();
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IWindow> window = device->CreateGLFWWindow({160, 120, "RHI scope contract"});
  ASSERT_TRUE(window);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(queue);
  ObjPtr<IImage> image = window->GetNextImage();
  ASSERT_TRUE(image);
  ObjPtr<IBuffer> buffer = device->CreateBuffer({16});
  ASSERT_TRUE(buffer);
  for (int operation = 0; operation < 7; ++operation) {
    ObjPtr<ICommandContext> commands = device->CreateCommandContext();
    ASSERT_TRUE(commands);
    if (operation == 0) {
      EXPECT_FALSE(commands->CmdEndRendering());
    } else if (operation == 1) {
      EXPECT_FALSE(commands->CmdDraw(3));
    } else {
      ASSERT_TRUE(commands->CmdBeginRendering({image.Get(), ATTACHMENT_LOAD_OP_CLEAR}));
      switch (operation) {
        case 2:
          EXPECT_FALSE(commands->CmdBeginRendering({image.Get()}));
          break;
        case 3:
          EXPECT_FALSE(commands->CmdClearColor(image.Get(), {0, 0, 0, 1}));
          break;
        case 4:
          EXPECT_FALSE(commands->CmdPresent(window.Get()));
          break;
        case 5: {
          const uint32_t value = 0;
          EXPECT_FALSE(commands->CmdUploadBuffer(buffer.Get(), 0, &value, sizeof(value)));
          break;
        }
        case 6:
          break;  // Submission must not implicitly end an open scope.
      }
    }
    ObjPtr<ISubmissionToken> rejected = queue->Submit(commands.Get());
    EXPECT_FALSE(rejected);
  }
}
#endif

static const BackendAPI program_backends[] = {
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
INSTANTIATE_TEST_SUITE_P(Backends, RhiProgram, testing::ValuesIn(program_backends));
}  // namespace grassland::rhi

#endif
