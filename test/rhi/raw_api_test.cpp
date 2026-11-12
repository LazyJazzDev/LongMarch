#include <gtest/gtest.h>

#include <type_traits>

// This translation unit deliberately uses only public RHI headers.
#include "grassland/rhi/rhi.h"

namespace grassland::rhi {
namespace {
static_assert(std::is_abstract_v<ICore>);
static_assert(std::is_abstract_v<IDevice>);
static_assert(std::is_abstract_v<IWindow>);
static_assert(std::is_abstract_v<IImage>);
static_assert(std::is_abstract_v<IQueue>);
static_assert(std::is_abstract_v<ICommandContext>);
static_assert(std::is_abstract_v<ISubmissionToken>);

class RhiRaw : public testing::TestWithParam<BackendAPI> {};

TEST_P(RhiRaw, QueueIdentitySurvivesCallerReferenceRelease) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  IQueue *identities[3]{};
  for (auto type : {QUEUE_TYPE_MAIN, QUEUE_TYPE_ASSIST, QUEUE_TYPE_COMPUTE}) {
    ObjPtr<IQueue> queue = device->GetQueue(type);
    ASSERT_TRUE(queue);
    ObjPtr<IQueue> second = device->GetQueue(type);
    EXPECT_EQ(queue.Get(), second.Get());
    identities[type] = queue.Get();
  }
  EXPECT_NE(identities[0], identities[1]);
  EXPECT_NE(identities[0], identities[2]);
  EXPECT_NE(identities[1], identities[2]);
  for (auto type : {QUEUE_TYPE_MAIN, QUEUE_TYPE_ASSIST, QUEUE_TYPE_COMPUTE}) {
    ObjPtr<IQueue> queue = device->GetQueue(type);
    ASSERT_TRUE(queue);
    EXPECT_EQ(queue.Get(), identities[type]);
    EXPECT_EQ(queue->GetType(), type);
  }
}

TEST_P(RhiRaw, PointerFactoriesAndBorrowedSubmissionWorkWithoutWrappers) {
  ICore *core = CreateCore(GetParam());
  ASSERT_NE(core, nullptr);
#if !defined(NDEBUG)
  EXPECT_EQ(core->CreateDevice(-1), nullptr);
#endif
  IDevice *device = core->CreateDeviceByAutoSelection();
  ASSERT_NE(device, nullptr);
  IQueue *queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ASSERT_NE(queue, nullptr);
  ICommandContext *context = ObjPtr<IQueue>(device->GetQueue(QUEUE_TYPE_MAIN))->CreateCommandContext();
  ASSERT_NE(context, nullptr);
  ISubmissionToken *done = context->Submit();
  ASSERT_NE(done, nullptr);
#if !defined(NDEBUG)
  EXPECT_FALSE(context->CmdClearImage(nullptr, {0.0f, 0.0f, 0.0f, 1.0f}));
  EXPECT_FALSE(context->CmdPresent(nullptr));
  EXPECT_EQ(context->Submit(), nullptr);  // One-shot submission.
#endif
  context->Release();  // Submit borrowed the caller reference and retained work.
  EXPECT_TRUE(done->WaitHost());
  EXPECT_EQ(done->GetStatus(), SUBMISSION_STATUS_COMPLETED);
  done->Release();
  queue->Release();
  device->Release();
  core->Release();
}

TEST_P(RhiRaw, ContextSubmitsAfterCallerReleasesItsQueueReference) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  for (auto type : {QUEUE_TYPE_MAIN, QUEUE_TYPE_ASSIST, QUEUE_TYPE_COMPUTE}) {
    ObjPtr<IQueue> queue = device->GetQueue(type);
    ASSERT_TRUE(queue);
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    ASSERT_TRUE(commands);
    queue.Release();
    ObjPtr<ISubmissionToken> done = commands->Submit();
    ASSERT_TRUE(done);
    commands.Release();
    EXPECT_TRUE(done->WaitHost());
  }
}

TEST_P(RhiRaw, ComputeQueueClearsAndDispatchesAcrossMainQueueDependencies) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> main = device->GetQueue(QUEUE_TYPE_MAIN);
  ObjPtr<IQueue> compute = device->GetQueue(QUEUE_TYPE_COMPUTE);
  ASSERT_TRUE(main);
  ASSERT_TRUE(compute);
  ObjPtr<IBuffer> values = device->CreateBuffer({16});
  ObjPtr<IBuffer> parameters = device->CreateBuffer({256});
  ObjPtr<IImage> source = device->CreateImage({2, 2, IMAGE_FORMAT_R32G32B32A32_SFLOAT});
  ASSERT_TRUE(values);
  ASSERT_TRUE(parameters);
  ASSERT_TRUE(source);
  ObjPtr<IShader> shader = device->CreateShaderFromSource(R"(
    struct Parameters { uint delta; };
    RHI_RW_STRUCTURED_BUFFER(0,0,uint,values);
    RHI_TEXTURE2D(0,1,float4,source);
    RHI_UNIFORM(0,2,Parameters,parameters);
    [numthreads(1,1,1)] void CSMain(uint3 id:SV_DispatchThreadID) {
      values[id.x] += uint(source.Load(int3(0,0,0)).x) + parameters.delta;
    }
  )",
                                                          "CSMain", SHADER_STAGE_COMPUTE);
  ASSERT_TRUE(shader);
  ComputeProgramDesc desc;
  desc.compute_shader = shader.Get();
  desc.pipeline_layout.groups = {{{{0, BINDING_RW_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_COMPUTE, 4},
                                   {1, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_COMPUTE},
                                   {2, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_COMPUTE}}}};
  ObjPtr<IComputeProgram> program = device->CreateComputeProgram(desc);
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ASSERT_TRUE(program);
  ASSERT_TRUE(group);
  ASSERT_TRUE(group->BindResource(0, values.Get()));
  ASSERT_TRUE(group->BindResource(1, source.Get()));
  ASSERT_TRUE(group->BindResource(2, parameters.Get()));
  ObjPtr<ISubmissionToken> previous;
  for (uint32_t round = 0; round < 8; ++round) {
    uint32_t initial[4] = {round, round + 1, round + 2, round + 3};
    const uint32_t delta = round + 2;
    ObjPtr<ICommandContext> upload = main->CreateCommandContext();
    ASSERT_TRUE(upload);
    ASSERT_TRUE(upload->CmdUploadBuffer(values.Get(), 0, initial, sizeof(initial)));
    ASSERT_TRUE(upload->CmdUploadBuffer(parameters.Get(), 0, &delta, sizeof(delta)));
    SubmitInfo waits;
    if (previous)
      waits.waits.push_back(previous->GetEvent());
    ObjPtr<ISubmissionToken> prepared = upload->Submit(waits);
    ASSERT_TRUE(prepared);
    ObjPtr<ICommandContext> commands = compute->CreateCommandContext();
    ASSERT_TRUE(commands);
    const float color = float(round + 1);
    ASSERT_TRUE(commands->CmdClearImage(source.Get(), {color, 0.0f, 0.0f, 1.0f}));
    ASSERT_TRUE(commands->CmdBindComputeProgram(program.Get()));
    ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
    ASSERT_TRUE(commands->CmdDispatch(4, 1, 1));
    waits.waits = {prepared->GetEvent()};
    ObjPtr<ISubmissionToken> rendered = commands->Submit(waits);
    ASSERT_TRUE(rendered);
    ObjPtr<ICommandContext> read = main->CreateCommandContext();
    ASSERT_TRUE(read);
    ObjPtr<IDownloadToken> result = read->CmdDownloadBuffer(values.Get(), 0, sizeof(initial));
    ASSERT_TRUE(result);
    waits.waits = {rendered->GetEvent()};
    previous = read->Submit(waits);
    ASSERT_TRUE(previous);
    ASSERT_TRUE(previous->WaitHost());
    uint32_t actual[4]{};
    ASSERT_TRUE(result->CopyTo(actual, sizeof(actual)));
    for (uint32_t i = 0; i < 4; ++i)
      EXPECT_EQ(actual[i], initial[i] + uint32_t(color) + delta);
  }
}

TEST_P(RhiRaw, ComputeQueueRejectsGraphicsAndDepthClears) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> compute = device->GetQueue(QUEUE_TYPE_COMPUTE);
  ASSERT_TRUE(compute);
  ObjPtr<ICommandContext> graphics = compute->CreateCommandContext();
  ASSERT_TRUE(graphics);
  EXPECT_FALSE(graphics->CmdBeginRendering({}));
  EXPECT_EQ(graphics->Submit(), nullptr);
  ObjPtr<IImage> depth = device->CreateImage({2, 2, IMAGE_FORMAT_D32_SFLOAT});
  ASSERT_TRUE(depth);
  ObjPtr<ICommandContext> clear = compute->CreateCommandContext();
  ASSERT_TRUE(clear);
  EXPECT_FALSE(clear->CmdClearImage(depth.Get(), ClearValue{ClearDepthStencilValue{1.0f, 0}}));
  EXPECT_EQ(clear->Submit(), nullptr);
}

std::vector<BackendAPI> Backends() {
  return {
#ifdef LONGMARCH_D3D12_ENABLED
      BACKEND_API_D3D12,
#endif
#ifdef LONGMARCH_METAL_ENABLED
      BACKEND_API_METAL,
#endif
#ifdef LONGMARCH_VULKAN_ENABLED
      BACKEND_API_VULKAN,
#endif
  };
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiRaw, testing::ValuesIn(Backends()));
}  // namespace
}  // namespace grassland::rhi
