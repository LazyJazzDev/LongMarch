#include <gtest/gtest.h>

#include <cstdlib>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_program.h"
#if defined(LONGMARCH_D3D12_ENABLED)
#include <d3d12sdklayers.h>

#include "grassland/rhi/backend/d3d12/d3d12_binding_group.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_program.h"
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
    desc.color_formats = {frame->GetDesc().format};
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
        {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0.6f, 0.7f, 0.8f, 1}}}}));
    ASSERT_TRUE(commands->CmdDraw(3));
    ASSERT_TRUE(commands->CmdEndRendering());
    // A LOAD pass is compatible with the same Program and framebuffer as CLEAR.
    ASSERT_TRUE(commands->CmdBeginRendering({{{image.Get()}}}));
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
      ASSERT_TRUE(commands->CmdBeginRendering({{{image.Get(), load, store, {0.6f, 0.7f, 0.8f, 1}}}}));
      ASSERT_TRUE(commands->CmdEndRendering());
    }
  }
  ASSERT_TRUE(commands->CmdBeginRendering(
      {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0.6f, 0.7f, 0.8f, 1}}}}));
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
      ASSERT_TRUE(commands->CmdBeginRendering({{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR}}}));
      switch (operation) {
        case 2:
          EXPECT_FALSE(commands->CmdBeginRendering({{{image.Get()}}}));
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

TEST_P(RhiProgram, BindingLayoutRejectsIncorrectStageTypeCountAndRange) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  const std::string source = R"(
    struct Data {float4 color;};RHI_UNIFORM(0,0,Data,data);
    float4 VSMain(uint id:SV_VertexID):SV_Position {return float4(0,0,0,1);}
    float4 PSMain():SV_Target0 {return data.color;}
  )";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.pipeline_layout.groups = {{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_VERTEX}}}};
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  EXPECT_FALSE(program->Finalize());
  desc.pipeline_layout.groups[0].bindings[0].visibility = SHADER_VISIBILITY_FRAGMENT;
  program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  ASSERT_TRUE(program->Finalize());
  desc.pipeline_layout.groups[0].bindings[0].type = BINDING_TEXTURE2D;
  program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  EXPECT_FALSE(program->Finalize());
  // Reflection also checks declarations that bypass the generated macro.
  const std::string manual = (GetParam() == BACKEND_API_VULKAN ? "[[vk::binding(0,0)]] Texture2D<float4> textures[3];"
                                                               : "Texture2D<float4> textures[3]:register(t0,space0);") +
                             std::string(R"(
    float4 VSMain(uint id:SV_VertexID):SV_Position {return float4(0,0,0,1);}
    float4 PSMain():SV_Target0 {return textures[2].Load(int3(0,0,0));}
  )");
  vs = device->CreateShader({manual, "VSMain", SHADER_STAGE_VERTEX});
  ps = device->CreateShader({manual, "PSMain", SHADER_STAGE_FRAGMENT});
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.pipeline_layout.groups[0].bindings[0].count = 2;
  program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  EXPECT_FALSE(program->Finalize());
  ObjPtr<IBuffer> buffer = device->CreateBuffer({512});
  ASSERT_TRUE(buffer);
  BindingLayoutDesc layout{{{0, BINDING_UNIFORM_BUFFER}}};
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(layout);
  ASSERT_TRUE(group);
  EXPECT_FALSE(group->BindResource(0, BufferView{buffer.Get(), 1, 256}));
  EXPECT_FALSE(group->BindResource(0, BufferView{buffer.Get(), 512, 256}));
  EXPECT_TRUE(group->BindResource(0, BufferView{buffer.Get(), 256, 256}));
  EXPECT_FALSE(group->BindResource(1, buffer.Get()));
  layout.bindings[0].count = 2;
  EXPECT_EQ(group->GetLayout().bindings[0].count, 1u);
  ObjPtr<IBindingGroup> array = device->CreateBindingGroup(layout);
  ASSERT_TRUE(array);
  EXPECT_TRUE(array->BindResource(0, std::vector<IBuffer *>{buffer.Get(), buffer.Get()}));
  EXPECT_FALSE(array->BindResource(0, std::vector<IBuffer *>{buffer.Get(), buffer.Get()}, 1));
}

TEST_P(RhiProgram, BindingUpdatesAreTransactionalAndForkOnlyCapturedVersions) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IBuffer> a = device->CreateBuffer({512});
  ObjPtr<IBuffer> b = device->CreateBuffer({256});
  ObjPtr<IImage> image = device->CreateImage({1, 1, IMAGE_FORMAT_RGBA8_UNORM});
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  ASSERT_TRUE(image);
  BindingLayoutDesc layout{{{2, BINDING_UNIFORM_BUFFER, 2}, {7, BINDING_TEXTURE2D}, {9, BINDING_SAMPLER}}};
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(layout);
  ASSERT_TRUE(group);
  auto native = static_cast<backend::BindingGroup *>(group.Get());
  auto initial = native->CurrentVersion();
  const auto *stable_layout = &group->GetLayout();
  EXPECT_FALSE(initial->IsComplete());
  ASSERT_TRUE(group->BindResource(2, std::vector<BufferView>{{a.Get(), 0, 256}}, 1));
  ASSERT_TRUE(group->BindResource(2, BufferView{b.Get()}));
  ASSERT_TRUE(group->BindResource(7, image.Get()));
  ASSERT_TRUE(group->BindResource(9, SamplerDesc{}));
  EXPECT_TRUE(initial->IsComplete());
  EXPECT_EQ(native->CurrentVersion(), initial);
  // Native validation rejects the second slot; the first must not change either.
  EXPECT_FALSE(group->BindResource(2, std::vector<BufferView>{{a.Get(), 256, 256}, {a.Get(), 1, 256}}));
  EXPECT_EQ(initial->GetReferences()[0][0].Get(), b.Get());
  EXPECT_EQ(initial->GetReferences()[0][1].Get(), a.Get());
  EXPECT_FALSE(group->BindResource(7, b.Get()));
  EXPECT_FALSE(group->BindResource(2, image.Get()));
  EXPECT_FALSE(group->BindResource(9, std::vector<SamplerDesc>{{static_cast<SamplerFilter>(-1)}}));
  // A command's use-reference is the only reason to fork.
  auto captured = backend::ObjectPtr<backend::BindingGroup::Version>::Retain(initial);
  EXPECT_FALSE(group->BindResource(2, BufferView{a.Get(), 1, 256}));
  EXPECT_EQ(native->CurrentVersion(), initial);
  ASSERT_TRUE(group->BindResource(2, BufferView{a.Get(), 256, 256}));
  auto next = native->CurrentVersion();
  EXPECT_NE(next, initial);
  EXPECT_EQ(&group->GetLayout(), stable_layout);
  EXPECT_EQ(captured->GetReferences()[0][0].Get(), b.Get());
  EXPECT_EQ(next->GetReferences()[0][0].Get(), a.Get());
  ASSERT_TRUE(group->BindResource(9, SamplerDesc{SAMPLER_FILTER_NEAREST}));
  EXPECT_EQ(native->CurrentVersion(), next);
  EXPECT_TRUE(captured->IsInitialized(2, 0));
  EXPECT_TRUE(captured->GetReferences()[2].empty());
  // Capture of the unchanged version shares all descriptors and resources.
  auto again = backend::ObjectPtr<backend::BindingGroup::Version>::Retain(next);
  EXPECT_EQ(again.Get(), next);
  again.Reset();
  ASSERT_TRUE(group->BindResource(2, BufferView{a.Get(), 0, 256}));
  EXPECT_EQ(native->CurrentVersion(), next);
  ObjPtr<IBuffer> left = device->CreateBuffer({256});
  ObjPtr<IBuffer> right = device->CreateBuffer({512});
  ASSERT_TRUE(left);
  ASSERT_TRUE(right);
  ObjPtr<IBindingGroup> swap = device->CreateBindingGroup({{{0, BINDING_UNIFORM_BUFFER, 2}}});
  ASSERT_TRUE(swap);
  auto left_ptr = left.Get();
  auto right_ptr = right.Get();
  ASSERT_TRUE(swap->BindResource(0, std::vector<IBuffer *>{left_ptr, right_ptr}));
  left.Release();
  right.Release();
  ASSERT_TRUE(swap->BindResource(0, std::vector<IBuffer *>{right_ptr, left_ptr}));
  EXPECT_EQ(left_ptr->GetDesc().size, 256u);
  EXPECT_EQ(right_ptr->GetDesc().size, 512u);
  group.Release();
  a.Release();
  b.Release();
  image.Release();
  EXPECT_EQ(static_cast<IBuffer *>(captured->GetReferences()[0][0].Get())->GetDesc().size, 256u);
  EXPECT_EQ(static_cast<IImage *>(captured->GetReferences()[1][0].Get())->GetDesc().width, 1u);
}

TEST_P(RhiProgram, TypedBindingArraysPreserveSparseVersionsAndPartialUpdates) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IBuffer> buffer = device->CreateBuffer({512});
  ObjPtr<IImage> first = device->CreateImage({1, 1, IMAGE_FORMAT_RGBA8_UNORM});
  ObjPtr<IImage> second = device->CreateImage({2, 2, IMAGE_FORMAT_RGBA8_UNORM});
  ASSERT_TRUE(buffer);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  // Unsorted binding numbers and holes exercise cloning without reading unwritten descriptors.
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(
      {{{9, BINDING_SAMPLER, 3}, {2, BINDING_UNIFORM_BUFFER, 3}, {7, BINDING_TEXTURE2D, 3}}});
  ASSERT_TRUE(group);
  auto native = static_cast<backend::BindingGroup *>(group.Get());
  ASSERT_TRUE(group->BindResource(2, std::vector<BufferView>{{buffer.Get(), 256, 0}}, 1));
  ASSERT_TRUE(group->BindResource(7, std::vector<IImage *>{first.Get()}, 2));
  ASSERT_TRUE(group->BindResource(9, std::vector<SamplerDesc>{{SAMPLER_FILTER_NEAREST}}, 1));
  auto sparse = backend::ObjectPtr<backend::BindingGroup::Version>::Retain(native->CurrentVersion());
  ASSERT_FALSE(sparse->IsComplete());
  ASSERT_TRUE(group->BindResource(9, std::vector<SamplerDesc>{{SAMPLER_FILTER_LINEAR}, {SAMPLER_FILTER_NEAREST}}, 1));
  auto next = native->CurrentVersion();
  EXPECT_NE(next, sparse.Get());
  EXPECT_FALSE(next->IsInitialized(0, 0));
  EXPECT_EQ(next->GetReferences()[0][1].Get(), buffer.Get());
  EXPECT_FALSE(next->IsInitialized(0, 2));
  EXPECT_EQ(next->GetReferences()[1][2].Get(), first.Get());
  EXPECT_TRUE(sparse->IsInitialized(2, 1));
  EXPECT_TRUE(next->IsInitialized(2, 1));
  EXPECT_FALSE(sparse->IsInitialized(2, 2));
  EXPECT_TRUE(next->IsInitialized(2, 2));
  ASSERT_TRUE(group->BindResource(2, std::vector<BufferView>{{buffer.Get(), 0, 256}, {buffer.Get(), 256, 256}}, 0));
  ASSERT_TRUE(group->BindResource(2, std::vector<BufferView>{{buffer.Get(), 0, 256}}, 2));
  ASSERT_TRUE(group->BindResource(7, std::vector<IImage *>{first.Get(), second.Get()}, 0));
  ASSERT_TRUE(group->BindResource(9, SamplerDesc{}));
  EXPECT_TRUE(next->IsComplete());
  EXPECT_EQ(native->CurrentVersion(), next);
  EXPECT_FALSE(group->BindResource(7, std::vector<IImage *>{second.Get(), nullptr}));
  EXPECT_EQ(next->GetReferences()[1][0].Get(), first.Get());
  EXPECT_FALSE(
      group->BindResource(9, std::vector<SamplerDesc>{{SAMPLER_FILTER_NEAREST}, {static_cast<SamplerFilter>(-1)}}));
  EXPECT_EQ(native->CurrentVersion(), next);
  // Image array replacement must retain all inputs before dropping any old slot.
  auto first_ptr = first.Get();
  auto second_ptr = second.Get();
  first.Release();
  second.Release();
  ASSERT_TRUE(group->BindResource(7, std::vector<IImage *>{second_ptr, first_ptr}, 0));
  EXPECT_EQ(static_cast<IImage *>(next->GetReferences()[1][0].Get())->GetDesc().width, 2u);
  EXPECT_EQ(static_cast<IImage *>(next->GetReferences()[1][1].Get())->GetDesc().width, 1u);
}

TEST_P(RhiProgram, StructuredStridesMustMatchExplicitLayoutAcrossStages) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  const std::string vertex = R"(
    RHI_STRUCTURED_BUFFER(0,0,float4,positions);
    float4 VSMain(uint id:SV_VertexID):SV_Position { return positions[id]; }
  )";
  const std::string fragment = R"(
    RHI_STRUCTURED_BUFFER(0,0,float2,colors);
    float4 PSMain():SV_Target0 { return float4(colors[0],0,1); }
  )";
  ObjPtr<IShader> vs = device->CreateShader({vertex, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> ps = device->CreateShader({fragment, "PSMain", SHADER_STAGE_FRAGMENT});
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.pipeline_layout.groups = {{{{0, BINDING_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_ALL, 16}}}};
  ObjPtr<IProgram> mismatch = device->CreateProgram(desc);
  ASSERT_TRUE(mismatch);
  EXPECT_FALSE(mismatch->Finalize());
  EXPECT_FALSE(mismatch->IsFinalized());
  // Retrying must still reject the incompatible shader.
  EXPECT_FALSE(mismatch->Finalize());
  ps = device->CreateShader({R"(
    RHI_STRUCTURED_BUFFER(0,0,float4,colors);
    float4 PSMain():SV_Target0 { return colors[0]; }
  )",
                             "PSMain", SHADER_STAGE_FRAGMENT});
  desc.fragment_shader = ps.Get();
  ObjPtr<IProgram> matching = device->CreateProgram(desc);
  ASSERT_TRUE(matching);
  ASSERT_TRUE(matching->Finalize());

  const std::string source = R"(
    struct Element { float4 color; float scale; };
    RHI_STRUCTURED_BUFFER_ARRAY(0,0,Element,values);
    RHI_RW_STRUCTURED_BUFFER(0,1,Element,outputValues);
    RHI_STRUCTURED_BUFFER(0,2,float4,unusedValues);
    float4 VSMain(uint id:SV_VertexID):SV_Position {
      float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
      return float4(p[id],0,1);
    }
    float4 PSMain():SV_Target0 {
      outputValues[0]=values[1][1];return outputValues[0].color*outputValues[0].scale;
    }
  )";
  vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.pipeline_layout.groups = {{{{0, BINDING_STRUCTURED_BUFFER, 2, SHADER_VISIBILITY_FRAGMENT, 20},
                                   {1, BINDING_RW_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT, 20},
                                   {2, BINDING_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT, 16}}}};
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  ASSERT_TRUE(program->Finalize());
  desc.pipeline_layout.groups[0].bindings[0].stride = 32;
  ObjPtr<IProgram> padded = device->CreateProgram(desc);
  ASSERT_TRUE(padded);
  EXPECT_FALSE(padded->Finalize());
  EXPECT_TRUE(program->Finalize());
}

TEST_P(RhiProgram, StructuredLayoutStrideControlsCompatibilityAndRanges) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  BindingLayoutDesc layout{{{0, BINDING_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_ALL, 16}}};
  auto other = layout;
  other.bindings[0].stride = 8;
  EXPECT_FALSE(backend::SameBindingLayout(layout, other));
  other.bindings[0].stride = 0;
  EXPECT_FALSE(device->CreateBindingGroup(other));
  ObjPtr<IBuffer> buffer = device->CreateBuffer({256});
  ASSERT_TRUE(buffer);
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(layout);
  ASSERT_TRUE(group);
  EXPECT_FALSE(group->BindResource(0, BufferView{buffer.Get(), 4, 16}));
  EXPECT_FALSE(group->BindResource(0, BufferView{buffer.Get(), 0, 17}));
  EXPECT_TRUE(group->BindResource(0, BufferView{buffer.Get(), 0, 32}));
  layout.bindings[0].type = BINDING_BYTE_ADDRESS_BUFFER;
  other = layout;
  other.bindings[0].stride = 123;
  EXPECT_TRUE(backend::SameBindingLayout(layout, other));
  group = device->CreateBindingGroup(other);
  ASSERT_TRUE(group);
  EXPECT_FALSE(group->BindResource(0, BufferView{buffer.Get(), 1, 4}));
  EXPECT_FALSE(group->BindResource(0, BufferView{buffer.Get(), 0, 3}));
  EXPECT_TRUE(group->BindResource(0, BufferView{buffer.Get(), 0, 4}));

  const std::string source = R"(
    RHI_STRUCTURED_BUFFER(0,0,float2,values);
    float4 VSMain(uint id:SV_VertexID):SV_Position { return float4(values[id],0,1); }
    float4 PSMain():SV_Target0 { return 1; }
  )";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  layout.bindings[0].type = BINDING_STRUCTURED_BUFFER;
  layout.bindings[0].stride = 8;
  desc.pipeline_layout.groups = {layout};
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  ASSERT_TRUE(program->Finalize());
  layout.bindings[0].stride = 16;
  group = device->CreateBindingGroup(layout);
  ASSERT_TRUE(group);
  ASSERT_TRUE(group->BindResource(0, buffer.Get()));
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands);
  ASSERT_TRUE(commands->CmdBindProgram(program.Get()));
  EXPECT_FALSE(commands->CmdBindGroup(0, group.Get()));
}

#if defined(LONGMARCH_D3D12_ENABLED)
TEST(D3D12BindingGroup, CpuDescriptorsAreFinalBeforeProgramBinding) {
  using namespace backend::d3d12;
  ObjPtr<ICore> core = CreateCore(BACKEND_API_D3D12);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IBuffer> buffer = device->CreateBuffer({256});
  ASSERT_TRUE(buffer);
  BindingLayoutDesc layout{{{0, BINDING_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT, 16},
                            {1, BINDING_SAMPLER, 1, SHADER_VISIBILITY_FRAGMENT}}};
  auto make_program = [&](bool wide) {
    std::string source =
        wide ? "RHI_STRUCTURED_BUFFER(0,0,float4,values);" : "RHI_STRUCTURED_BUFFER(0,0,float2,values);";
    source += "float4 VSMain(uint id:SV_VertexID):SV_Position { return float4(0,0,0,1); }";
    source += wide ? "float4 PSMain():SV_Target0 { return values[0]; }"
                   : "float4 PSMain():SV_Target0 { return float4(values[0],0,1); }";
    ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
    ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
    ProgramDesc desc;
    desc.vertex_shader = vs.Get();
    desc.fragment_shader = ps.Get();
    desc.pipeline_layout.groups = {layout};
    desc.pipeline_layout.groups[0].bindings[0].stride = wide ? 16 : 8;
    return ObjPtr<IProgram>(device->CreateProgram(desc));
  };
  auto wide = make_program(true);
  auto narrow = make_program(false);
  auto same = make_program(true);
  ASSERT_TRUE(wide);
  ASSERT_TRUE(narrow);
  ASSERT_TRUE(same);
  ASSERT_TRUE(wide->Finalize());
  ASSERT_TRUE(narrow->Finalize());
  ASSERT_TRUE(same->Finalize());
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(layout);
  ASSERT_TRUE(group);
  ASSERT_TRUE(group->BindResource(0, buffer.Get()));
  ASSERT_TRUE(group->BindResource(1, SamplerDesc{}));
  auto native = static_cast<D3D12BindingGroup *>(group.Get());
  auto version = static_cast<D3D12BindingGroup::Version *>(native->CurrentVersion());
  auto a = version->ResourceDescriptors();
  ASSERT_TRUE(a);
  EXPECT_EQ(a->Heap()->GetDesc().Flags, D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
  EXPECT_EQ(version->SamplerDescriptors()->Heap()->GetDesc().Flags, D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
  auto sampler = version->SamplerDescriptors()->CPU().ptr;
  ObjPtr<ICommandContext> first = device->CreateCommandContext();
  ObjPtr<ICommandContext> second = device->CreateCommandContext();
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  ASSERT_TRUE(first->CmdBindProgram(wide.Get()));
  ASSERT_TRUE(first->CmdBindGroup(0, group.Get()));
  ASSERT_TRUE(second->CmdBindProgram(same.Get()));
  ASSERT_TRUE(second->CmdBindGroup(0, group.Get()));
  EXPECT_EQ(native->CurrentVersion(), version);
  EXPECT_EQ(version->ResourceDescriptors(), a);
  EXPECT_EQ(version->SamplerDescriptors()->CPU().ptr, sampler);
  ASSERT_TRUE(group->BindResource(1, SamplerDesc{SAMPLER_FILTER_NEAREST}));
  auto replacement = static_cast<D3D12BindingGroup::Version *>(native->CurrentVersion());
  EXPECT_NE(replacement, version);
  EXPECT_NE(replacement->SamplerDescriptors()->CPU().ptr, sampler);
  EXPECT_EQ(version->ResourceDescriptors(), a);
  ObjPtr<ICommandContext> rejected = device->CreateCommandContext();
  ASSERT_TRUE(rejected);
  ASSERT_TRUE(rejected->CmdBindProgram(narrow.Get()));
  EXPECT_FALSE(rejected->CmdBindGroup(0, group.Get()));
  // Discarding recorded commands must return their version leases, without submission.
  first.Release();
  second.Release();
}
#endif

INSTANTIATE_TEST_SUITE_P(Backends, RhiProgram, testing::ValuesIn(program_backends));
}  // namespace grassland::rhi

#endif
