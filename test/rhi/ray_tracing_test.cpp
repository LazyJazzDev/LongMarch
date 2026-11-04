#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstring>
#include <thread>

#include "grassland/rhi/rhi.h"
#if defined(LONGMARCH_METAL_ENABLED)
#include "grassland/rhi/backend/metal/metal_acceleration_structure.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_image.h"
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
#include <d3d12sdklayers.h>

#include "grassland/rhi/backend/d3d12/d3d12_acceleration_structure.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/rhi/backend/vulkan/vulkan_acceleration_structure.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#endif
#if defined(LONGMARCH_D3D12_ENABLED) || defined(LONGMARCH_VULKAN_ENABLED) || defined(LONGMARCH_METAL_ENABLED)
namespace grassland::rhi {
namespace {
template <size_t N>
void ReadBuffer(IDevice *device, IBuffer *buffer, BackendAPI api, std::array<uint32_t, N> &output) {
#if defined(LONGMARCH_METAL_ENABLED)
  if (api == BACKEND_API_METAL) {
    using namespace backend::metal;
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto owner = static_cast<MetalDevice *>(device);
    auto readback = NS::TransferPtr(owner->NativeDevice()->newBuffer(sizeof(output), MTL::ResourceStorageModeShared));
    ASSERT_TRUE(readback);
    std::lock_guard<std::mutex> lock(owner->QueueMutex());
    auto commands = NS::RetainPtr(owner->NativeQueue()->commandBuffer());
    auto blit = commands->blitCommandEncoder();
    blit->copyFromBuffer(static_cast<MetalBuffer *>(buffer)->NativeBuffer(), 0, readback.get(), 0, sizeof(output));
    blit->endEncoding();
    commands->commit();
    commands->waitUntilCompleted();
    ASSERT_EQ(commands->status(), MTL::CommandBufferStatusCompleted);
    std::memcpy(output.data(), readback->contents(), sizeof(output));
    return;
  }
#endif

#if defined(LONGMARCH_D3D12_ENABLED)
  if (api == BACKEND_API_D3D12) {
    using namespace backend::d3d12;
    auto owner = static_cast<D3D12Device *>(device);
    auto source = static_cast<D3D12Buffer *>(buffer)->NativeBuffer();
    auto native = owner->NativeDevice();
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = sizeof(output);
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> readback;
    ASSERT_TRUE(SUCCEEDED(native->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback))));
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ASSERT_TRUE(SUCCEEDED(native->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))));
    ASSERT_TRUE(SUCCEEDED(
        native->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list))));
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = source;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(readback.Get(), 0, source, 0, sizeof(output));
    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    list->ResourceBarrier(1, &barrier);
    ASSERT_TRUE(SUCCEEDED(list->Close()));
    {
      std::lock_guard<std::mutex> lock(owner->QueueMutex());
      ID3D12CommandList *lists[] = {list.Get()};
      owner->NativeQueue()->ExecuteCommandLists(1, lists);
      ASSERT_TRUE(owner->WaitIdle());
    }
    void *mapped{};
    D3D12_RANGE range{0, sizeof(output)};
    ASSERT_TRUE(SUCCEEDED(readback->Map(0, &range, &mapped)));
    std::memcpy(output.data(), mapped, sizeof(output));
    D3D12_RANGE written{0, 0};
    readback->Unmap(0, &written);
    return;
  }
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
  if (api == BACKEND_API_VULKAN) {
    using namespace backend::vulkan;
    auto owner = static_cast<VulkanDevice *>(device);
    auto native = owner->NativeDevice();
    auto source = static_cast<VulkanBuffer *>(buffer)->NativeBuffer();
    VkBufferCreateInfo desc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    desc.size = sizeof(output);
    desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer readback{};
    ASSERT_EQ(vkCreateBuffer(native, &desc, nullptr, &readback), VK_SUCCESS);
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(native, readback, &requirements);
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(owner->NativePhysicalDevice(), &properties);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
      auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      if ((requirements.memoryTypeBits & (1u << i)) && (properties.memoryTypes[i].propertyFlags & flags) == flags) {
        type = i;
        break;
      }
    }
    ASSERT_NE(type, UINT32_MAX);
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = type;
    VkDeviceMemory memory{};
    ASSERT_EQ(vkAllocateMemory(native, &allocate, nullptr, &memory), VK_SUCCESS);
    ASSERT_EQ(vkBindBufferMemory(native, readback, memory, 0), VK_SUCCESS);
    VkCommandPool pool{};
    VkCommandBuffer commands{};
    ASSERT_TRUE(owner->AcquireCommandResources(pool, commands));
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = source;
    barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                         &barrier, 0, nullptr);
    VkBufferCopy copy{0, 0, sizeof(output)};
    vkCmdCopyBuffer(commands, source, readback, 1, &copy);
    barrier.buffer = readback;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                         &barrier, 0, nullptr);
    ASSERT_EQ(vkEndCommandBuffer(commands), VK_SUCCESS);
    uint64_t value{};
    {
      std::lock_guard<std::mutex> lock(owner->QueueMutex());
      value = owner->NextSerial();
      auto semaphore = owner->Timeline();
      VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
      timeline.signalSemaphoreValueCount = 1;
      timeline.pSignalSemaphoreValues = &value;
      VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit.pNext = &timeline;
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &commands;
      submit.signalSemaphoreCount = 1;
      submit.pSignalSemaphores = &semaphore;
      ASSERT_EQ(vkQueueSubmit(owner->NativeQueue(), 1, &submit, {}), VK_SUCCESS);
    }
    ASSERT_EQ(owner->Wait(value), VK_SUCCESS);
    void *mapped{};
    ASSERT_EQ(vkMapMemory(native, memory, 0, sizeof(output), 0, &mapped), VK_SUCCESS);
    std::memcpy(output.data(), mapped, sizeof(output));
    vkUnmapMemory(native, memory);
    owner->RecycleCommandResources(pool, commands);
    vkDestroyBuffer(native, readback, nullptr);
    vkFreeMemory(native, memory, nullptr);
  }
#endif
}

class RhiRayTracing : public testing::TestWithParam<BackendAPI> {
 protected:
  void CheckNativeErrors(IDevice *device) {
#if defined(LONGMARCH_D3D12_ENABLED)
    if (GetParam() == BACKEND_API_D3D12) {
      backend::d3d12::ComPtr<ID3D12InfoQueue> messages;
      ASSERT_TRUE(SUCCEEDED(
          static_cast<backend::d3d12::D3D12Device *>(device)->NativeDevice()->QueryInterface(IID_PPV_ARGS(&messages))));
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
};

TEST_P(RhiRayTracing, BuildsUpdatesCompactsAndQueriesRetainedGeometry) {
  const auto api = GetParam();
  ObjPtr<ICore> core = CreateCore(api, {true});
  ASSERT_TRUE(core);
  DeviceFeatureRequirement required;
  required.ray_query = true;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection(required);
  if (!device)
    GTEST_SKIP() << "Ray query unavailable";
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(queue);
  BottomLevelAccelerationStructureDesc desc;
  desc.geometries = {{3}};
  desc.allow_update = true;
  desc.allow_compaction = true;
  ObjPtr<IBottomLevelAccelerationStructure> blas = device->CreateBottomLevelAccelerationStructure(desc);
  ASSERT_TRUE(blas);
  ObjPtr<IBuffer> vertices = device->CreateBuffer({36});
  ASSERT_TRUE(vertices);
  const float positions[] = {-1, -1, 0, 0, 1, 0, 1, -1, 0};
  ObjPtr<ITopLevelAccelerationStructure> tlas = device->CreateTopLevelAccelerationStructure({1, true, true});
  ASSERT_TRUE(tlas);
  RayTracingInstance instance;
  instance.blas = blas.Get();
  instance.disable_triangle_culling = true;
  ObjPtr<ICommandContext> build = device->CreateCommandContext();
  ASSERT_TRUE(build);
  ASSERT_TRUE(build->CmdUploadBuffer(vertices.Get(), 0, positions, sizeof(positions)));
  ASSERT_TRUE(build->CmdBuildBottomLevelAccelerationStructure(blas.Get(), {{{vertices.Get()}, {}}}));
  ASSERT_TRUE(build->CmdBuildBottomLevelAccelerationStructure(blas.Get(), {{{vertices.Get()}, {}}},
                                                              ACCELERATION_STRUCTURE_UPDATE));
  ASSERT_TRUE(build->CmdQueryCompactedSize(blas.Get()));
  ObjPtr<ISubmissionToken> token = queue->Submit(build.Get());
  ASSERT_TRUE(token);
  build.Release();
  vertices.Release();
  ASSERT_TRUE(token->WaitHost());
  desc.compacted_size = blas->GetCompactedSize();
  ASSERT_GT(desc.compacted_size, 0u);
  ObjPtr<IBottomLevelAccelerationStructure> compact = device->CreateBottomLevelAccelerationStructure(desc);
  ASSERT_TRUE(compact);
  build = device->CreateCommandContext();
  ASSERT_TRUE(build);
  ASSERT_TRUE(build->CmdCompactAccelerationStructure(blas.Get(), compact.Get()));
  instance.blas = compact.Get();
  ASSERT_TRUE(build->CmdBuildTopLevelAccelerationStructure(tlas.Get(), {instance}));
  ASSERT_TRUE(build->CmdQueryCompactedSize(tlas.Get()));
  token = queue->Submit(build.Get());
  ASSERT_TRUE(token);
  build.Release();
  blas.Release();
  ASSERT_TRUE(token->WaitHost());
  auto top_desc = tlas->GetDesc();
  top_desc.compacted_size = tlas->GetCompactedSize();
  ASSERT_GT(top_desc.compacted_size, 0u);
  ObjPtr<ITopLevelAccelerationStructure> compact_top = device->CreateTopLevelAccelerationStructure(top_desc);
  ASSERT_TRUE(compact_top);
  build = device->CreateCommandContext();
  ASSERT_TRUE(build);
  ASSERT_TRUE(build->CmdCompactAccelerationStructure(tlas.Get(), compact_top.Get()));
  token = queue->Submit(build.Get());
  ASSERT_TRUE(token);
  ASSERT_TRUE(token->WaitHost());
  build.Release();
  tlas.Release();
  tlas = std::move(compact_top);
  {
    // Abandoning a TLAS update must not replace its submitted BLAS ownership.
    auto replacement_desc = desc;
    replacement_desc.compacted_size = 0;
    ObjPtr<IBottomLevelAccelerationStructure> replacement =
        device->CreateBottomLevelAccelerationStructure(replacement_desc);
    ASSERT_TRUE(replacement);
    ObjPtr<ICommandContext> abandoned = device->CreateCommandContext();
    ASSERT_TRUE(abandoned);
    auto next_instance = instance;
    next_instance.blas = replacement.Get();
    ASSERT_TRUE(
        abandoned->CmdBuildTopLevelAccelerationStructure(tlas.Get(), {next_instance}, ACCELERATION_STRUCTURE_UPDATE));
    abandoned.Release();
#if defined(LONGMARCH_METAL_ENABLED)
    if (api == BACKEND_API_METAL)
      EXPECT_EQ(static_cast<backend::metal::MetalTopLevelAccelerationStructure *>(tlas.Get())->BottomLevels()[0].Get(),
                compact.Get());
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
    if (api == BACKEND_API_D3D12)
      EXPECT_EQ(static_cast<backend::d3d12::D3D12TopLevelAccelerationStructure *>(tlas.Get())->BottomLevels()[0].Get(),
                compact.Get());
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    if (api == BACKEND_API_VULKAN)
      EXPECT_EQ(
          static_cast<backend::vulkan::VulkanTopLevelAccelerationStructure *>(tlas.Get())->BottomLevels()[0].Get(),
          compact.Get());
#endif
  }
  const char *source = R"(
RHI_ACCELERATION_STRUCTURE(0,0,scene);
RHI_RW_BYTE_ADDRESS_BUFFER(0,1,output);
[numthreads(1,1,1)]
void CSMain() {
 RayDesc ray; ray.Origin = float3(0,0,-2); ray.Direction = float3(0,0,1); ray.TMin = 0; ray.TMax = 10;
 RayQuery<RAY_FLAG_NONE> query;
 query.TraceRayInline(scene,RAY_FLAG_NONE,255,ray);
 while(query.Proceed()) {}
 output.Store(0,query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 123u : 456u);
}
)";
  PipelineLayoutDesc layout{{{{{0, BINDING_ACCELERATION_STRUCTURE, 1, SHADER_VISIBILITY_COMPUTE},
                               {1, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE}}}}};
  ObjPtr<IShader> shader = device->CreateShaderFromSource(source, "CSMain", SHADER_STAGE_COMPUTE);
  ASSERT_TRUE(shader);
  ObjPtr<IComputeProgram> program = device->CreateComputeProgram({shader.Get(), layout});
  ASSERT_TRUE(program);
  ObjPtr<IBuffer> output = device->CreateBuffer({4});
  ASSERT_TRUE(output);
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(layout.groups[0]);
  ASSERT_TRUE(group);
  ASSERT_TRUE(group->BindResource(0, tlas.Get()));
  ASSERT_TRUE(group->BindResource(1, output.Get()));
  for (int frame = 0; frame < 2; ++frame) {
    build = device->CreateCommandContext();
    ASSERT_TRUE(build);
    instance.transform[3] = frame ? 5.0f : 0.0f;
    ASSERT_TRUE(build->CmdBuildTopLevelAccelerationStructure(tlas.Get(), {instance}, ACCELERATION_STRUCTURE_UPDATE));
    ASSERT_TRUE(build->CmdBindComputeProgram(program.Get()));
    ASSERT_TRUE(build->CmdBindGroup(0, group.Get()));
    ASSERT_TRUE(build->CmdDispatch(1));
    token = queue->Submit(build.Get());
    ASSERT_TRUE(token);
    ASSERT_TRUE(token->WaitHost());
    build.Release();
    std::array<uint32_t, 1> result{};
    ReadBuffer(device.Get(), output.Get(), api, result);
    EXPECT_EQ(result[0], frame ? 456u : 123u);
  }
  compact.Release();
  CheckNativeErrors(device.Get());
}

TEST_P(RhiRayTracing, ProceduralQueriesUseArraysMasksAndCapturedBindings) {
  const auto api = GetParam();
  ObjPtr<ICore> core = CreateCore(api, {true});
  ASSERT_TRUE(core);
  DeviceFeatureRequirement required;
  required.ray_query = true;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection(required);
  if (!device)
    GTEST_SKIP() << "Ray query unavailable";
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(queue);
  BottomLevelAccelerationStructureDesc desc;
  desc.aabbs = {{1}};
  desc.allow_update = true;
  ObjPtr<IBottomLevelAccelerationStructure> blas = device->CreateBottomLevelAccelerationStructure(desc);
  ASSERT_TRUE(blas);
  const float bounds[] = {-1, -1, -1, 1, 1, 1};
  ObjPtr<IBuffer> input = device->CreateBuffer({sizeof(bounds)});
  ASSERT_TRUE(input);
  std::array<ObjPtr<ITopLevelAccelerationStructure>, 2> tops;
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands);
  ASSERT_TRUE(commands->CmdUploadBuffer(input.Get(), 0, bounds, sizeof(bounds)));
  ASSERT_TRUE(commands->CmdBuildBottomLevelAccelerationStructure(blas.Get(), {{{input.Get()}, {}}}));
  ASSERT_TRUE(commands->CmdBuildBottomLevelAccelerationStructure(blas.Get(), {{{input.Get()}, {}}},
                                                                 ACCELERATION_STRUCTURE_UPDATE));
  for (uint32_t i = 0; i < tops.size(); ++i) {
    tops[i] = device->CreateTopLevelAccelerationStructure({1});
    ASSERT_TRUE(tops[i]);
    RayTracingInstance instance;
    instance.blas = blas.Get();
    instance.custom_index = 71 + i;
    instance.mask = 1 << i;
    instance.transform[3] = float(i * 4);
    ASSERT_TRUE(commands->CmdBuildTopLevelAccelerationStructure(tops[i].Get(), {instance}));
  }
  const char *source = R"(
RHI_ACCELERATION_STRUCTURE_ARRAY(0,3,scenes);
RHI_RW_BYTE_ADDRESS_BUFFER(0,8,output);
[numthreads(1,1,1)] void CSMain(uint3 id : SV_DispatchThreadID) {
 uint index = id.x & 1;
 RayDesc ray; ray.Origin=float3(index*4,0,-3); ray.Direction=float3(0,0,1); ray.TMin=0; ray.TMax=10;
 RayQuery<RAY_FLAG_NONE> query;
 query.TraceRayInline(scenes[index], RAY_FLAG_NONE, id.x < 2 ? (1u << index) : (2u >> index), ray);
 while (query.Proceed()) {
   if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
     query.CommitProceduralPrimitiveHit(2.0);
 }
 output.Store(id.x*4, query.CommittedStatus()==COMMITTED_PROCEDURAL_PRIMITIVE_HIT ? query.CommittedInstanceID() : 0);
}
)";
  // Unused AS entries before the active array exercise argument-buffer padding.
  PipelineLayoutDesc layout{{{{{0, BINDING_ACCELERATION_STRUCTURE, 2, SHADER_VISIBILITY_COMPUTE},
                               {3, BINDING_ACCELERATION_STRUCTURE, 2, SHADER_VISIBILITY_COMPUTE},
                               {8, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE}}}}};
  ObjPtr<IShader> shader = device->CreateShaderFromSource(source, "CSMain", SHADER_STAGE_COMPUTE);
  ASSERT_TRUE(shader);
  ObjPtr<IComputeProgram> program = device->CreateComputeProgram({shader.Get(), layout});
  ASSERT_TRUE(program);
  ObjPtr<IBuffer> output = device->CreateBuffer({16});
  ASSERT_TRUE(output);
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(layout.groups[0]);
  ASSERT_TRUE(group);
  std::vector<ITopLevelAccelerationStructure *> structures{tops[0].Get(), tops[1].Get()};
  ASSERT_TRUE(group->BindResource(0, structures));
  ASSERT_TRUE(group->BindResource(3, structures));
  ASSERT_TRUE(group->BindResource(8, output.Get()));
  ASSERT_TRUE(commands->CmdBindComputeProgram(program.Get()));
  ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
  ASSERT_TRUE(group->BindResource(3, std::vector<ITopLevelAccelerationStructure *>{tops[1].Get()}, 0));
  ASSERT_TRUE(commands->CmdDispatch(4));
  ObjPtr<ISubmissionToken> token = queue->Submit(commands.Get());
  ASSERT_TRUE(token);
  commands.Release();
  input.Release();
  blas.Release();
  tops = {};
  group.Release();
  ASSERT_TRUE(token->WaitHost());
  std::array<uint32_t, 4> result{};
  ReadBuffer(device.Get(), output.Get(), api, result);
  EXPECT_EQ(result, (std::array<uint32_t, 4>{71, 72, 0, 0}));
  CheckNativeErrors(device.Get());
}

TEST_P(RhiRayTracing, CustomShaderBindingTablesRouteRecordsAndRestoreDefault) {
  const auto api = GetParam();
  ObjPtr<ICore> core = CreateCore(api, {true});
  ASSERT_TRUE(core);
  DeviceFeatureRequirement required;
  required.ray_tracing_pipeline = true;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection(required);
  if (!device)
    GTEST_SKIP() << "Ray pipeline unavailable";
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(queue);
  BottomLevelAccelerationStructureDesc triangles;
  triangles.geometries = {{3, 12, 3, INDEX_FORMAT_UINT16}};
  BottomLevelAccelerationStructureDesc boxes;
  boxes.aabbs = {{1, 24, false}};
  ObjPtr<IBottomLevelAccelerationStructure> triangle = device->CreateBottomLevelAccelerationStructure(triangles);
  ASSERT_TRUE(triangle);
  ObjPtr<IBottomLevelAccelerationStructure> sphere = device->CreateBottomLevelAccelerationStructure(boxes);
  ASSERT_TRUE(sphere);
  ObjPtr<ITopLevelAccelerationStructure> tlas = device->CreateTopLevelAccelerationStructure({2});
  ASSERT_TRUE(tlas);
  ObjPtr<IBuffer> input = device->CreateBuffer({128});
  ASSERT_TRUE(input);
  const float vertices[] = {-0.5f, -0.5f, 0, 0, 0.5f, 0, 0.5f, -0.5f, 0};
  const uint16_t indices[] = {0, 1, 2};
  const float bounds[] = {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f};
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands);
  ASSERT_TRUE(commands->CmdUploadBuffer(input.Get(), 0, vertices, sizeof(vertices)));
  ASSERT_TRUE(commands->CmdUploadBuffer(input.Get(), 40, indices, sizeof(indices)));
  ASSERT_TRUE(commands->CmdUploadBuffer(input.Get(), 64, bounds, sizeof(bounds)));
  ASSERT_TRUE(commands->CmdBuildBottomLevelAccelerationStructure(
      triangle.Get(), {{{input.Get(), 0, sizeof(vertices)}, {input.Get(), 40, sizeof(indices)}}}));
  ASSERT_TRUE(
      commands->CmdBuildBottomLevelAccelerationStructure(sphere.Get(), {{{input.Get(), 64, sizeof(bounds)}, {}}}));
  RayTracingInstance a;
  a.blas = triangle.Get();
  a.disable_triangle_culling = true;
  RayTracingInstance b;
  b.blas = sphere.Get();
  b.transform[3] = 2;
  b.hit_group_offset = 1;
  ASSERT_TRUE(commands->CmdBuildTopLevelAccelerationStructure(tlas.Get(), {a, b}));
  const char *source = R"(
RHI_ACCELERATION_STRUCTURE(0,0,scene);
RHI_RW_BYTE_ADDRESS_BUFFER(0,1,output);
struct Payload { uint result; };
struct Attributes { float2 unused; };
[shader("raygeneration")]
void RayGen() {
 uint id = DispatchRaysIndex().x;
 RayDesc ray; ray.Origin = float3(id*2,0,-2); ray.Direction = float3(0,0,1); ray.TMin = 0; ray.TMax = 10;
 Payload p; p.result = 0;
 TraceRay(scene,RAY_FLAG_NONE,255,0,1,0,ray,p);
 CallShader(0,p);
 output.Store(id*4,p.result);
}
[shader("callable")]
void Callable(inout Payload p) { p.result += 1; }
[shader("miss")]
void Miss(inout Payload p) { p.result = 333; }
[shader("closesthit")]
void Triangle(inout Payload p, BuiltInTriangleIntersectionAttributes attr) { p.result = 111; }
[shader("closesthit")]
void AlternateTriangle(inout Payload p, BuiltInTriangleIntersectionAttributes attr) { p.result = 444; }
[shader("miss")]
void AlternateMiss(inout Payload p) { p.result = 555; }
[shader("callable")]
void AlternateCallable(inout Payload p) { p.result += 10; }
[shader("intersection")]
void Intersection() {
 float3 o = ObjectRayOrigin(); float3 d = ObjectRayDirection();
 float b = dot(o,d); float delta = b*b-dot(d,d)*(dot(o,o)-0.25);
 if (delta >= 0) { Attributes a; a.unused = 0; ReportHit((-b-sqrt(delta))/dot(d,d),0,a); }
}
[shader("closesthit")]
void Sphere(inout Payload p, Attributes attr) { p.result = 222; }
)";
  ObjPtr<IShader> raygen = device->CreateShaderFromSource(source, "RayGen", SHADER_STAGE_RAY_GENERATION);
  ASSERT_TRUE(raygen);
  ObjPtr<IShader> miss = device->CreateShaderFromSource(source, "Miss", SHADER_STAGE_MISS);
  ASSERT_TRUE(miss);
  ObjPtr<IShader> hit = device->CreateShaderFromSource(source, "Triangle", SHADER_STAGE_CLOSEST_HIT);
  ASSERT_TRUE(hit);
  ObjPtr<IShader> sphere_hit = device->CreateShaderFromSource(source, "Sphere", SHADER_STAGE_CLOSEST_HIT);
  ASSERT_TRUE(sphere_hit);
  ObjPtr<IShader> intersection = device->CreateShaderFromSource(source, "Intersection", SHADER_STAGE_INTERSECTION);
  ASSERT_TRUE(intersection);
  RayTracingProgramDesc desc;
  desc.ray_generation = raygen.Get();
  desc.miss_shaders = {miss.Get()};
  desc.hit_groups = {{hit.Get(), nullptr, nullptr}, {sphere_hit.Get(), nullptr, intersection.Get()}};
  ObjPtr<IShader> callable = device->CreateShaderFromSource(source, "Callable", SHADER_STAGE_CALLABLE);
  ASSERT_TRUE(callable);
  desc.callable_shaders = {callable.Get()};
  ObjPtr<IShader> alternate_hit = device->CreateShaderFromSource(source, "AlternateTriangle", SHADER_STAGE_CLOSEST_HIT);
  ObjPtr<IShader> alternate_miss = device->CreateShaderFromSource(source, "AlternateMiss", SHADER_STAGE_MISS);
  ObjPtr<IShader> alternate_callable =
      device->CreateShaderFromSource(source, "AlternateCallable", SHADER_STAGE_CALLABLE);
  ASSERT_TRUE(alternate_hit);
  ASSERT_TRUE(alternate_miss);
  ASSERT_TRUE(alternate_callable);
  desc.hit_groups.push_back({alternate_hit.Get(), nullptr, nullptr});
  desc.miss_shaders.push_back(alternate_miss.Get());
  desc.callable_shaders.push_back(alternate_callable.Get());
  desc.max_payload_size = 4;
  desc.pipeline_layout.groups = {{{{0, BINDING_ACCELERATION_STRUCTURE, 1, SHADER_VISIBILITY_RAY_GENERATION},
                                   {1, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_RAY_GENERATION}}}};
  ObjPtr<IRayTracingProgram> program = device->CreateRayTracingProgram(desc);
  ASSERT_TRUE(program);
  EXPECT_EQ(program->CreateShaderBindingTable({{2}, {}, {}}), nullptr);
  EXPECT_EQ(program->CreateShaderBindingTable({{}, {3}, {}}), nullptr);
  EXPECT_EQ(program->CreateShaderBindingTable({{}, {}, {2}}), nullptr);
  ObjPtr<IShaderBindingTable> empty = program->CreateShaderBindingTable({});
  ASSERT_TRUE(empty);
  empty.Release();
  ShaderBindingTableDesc table_desc{{1, 0, 1}, {2, 1, 2}, {1, 0, 1}};
  ObjPtr<IShaderBindingTable> custom = program->CreateShaderBindingTable(table_desc);
  ASSERT_TRUE(custom);
  EXPECT_EQ(custom->GetProgram(), program.Get());
  EXPECT_EQ(custom->GetDesc().hit_group_indices, table_desc.hit_group_indices);
  ObjPtr<IBuffer> output = device->CreateBuffer({12});
  ASSERT_TRUE(output);
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ASSERT_TRUE(group);
  ASSERT_TRUE(group->BindResource(0, tlas.Get()));
  ASSERT_TRUE(group->BindResource(1, output.Get()));
  ObjPtr<IRayTracingProgram> other = device->CreateRayTracingProgram(desc);
  ASSERT_TRUE(other);
  ObjPtr<ICommandContext> rejected = device->CreateCommandContext();
  ASSERT_TRUE(rejected->CmdBindRayTracingProgram(other.Get()));
  ASSERT_TRUE(rejected->CmdBindGroup(0, group.Get()));
  EXPECT_FALSE(rejected->CmdDispatchRays(3, 1, 1, custom.Get()));
  EXPECT_FALSE(queue->Submit(rejected.Get()));
  rejected.Release();
  other.Release();
  ASSERT_TRUE(commands->CmdBindRayTracingProgram(program.Get()));
  ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
  ASSERT_TRUE(commands->CmdDispatchRays(3, 1));
  ObjPtr<ISubmissionToken> token = queue->Submit(commands.Get());
  ASSERT_TRUE(token && token->WaitHost());
  std::array<uint32_t, 3> result{};
  ReadBuffer(device.Get(), output.Get(), api, result);
  EXPECT_EQ(result, (std::array<uint32_t, 3>{112, 223, 334}));

  commands = device->CreateCommandContext();
  a.hit_group_offset = 2;  // Exercise a repeated record beyond the original two groups.
  ASSERT_TRUE(commands->CmdBuildTopLevelAccelerationStructure(tlas.Get(), {a, b}));
  ASSERT_TRUE(commands->CmdBindRayTracingProgram(program.Get()));
  ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
  ASSERT_TRUE(commands->CmdDispatchRays(3, 1, 1, custom.Get()));
  custom.Release();  // The recorded command must keep the SBT buffer alive.
  token = queue->Submit(commands.Get());
  ASSERT_TRUE(token);
  commands.Release();
  ASSERT_TRUE(token->WaitHost());
  ReadBuffer(device.Get(), output.Get(), api, result);
  EXPECT_EQ(result, (std::array<uint32_t, 3>{454, 232, 565}));

  commands = device->CreateCommandContext();
  a.hit_group_offset = 0;
  ASSERT_TRUE(commands->CmdBuildTopLevelAccelerationStructure(tlas.Get(), {a, b}));
  ASSERT_TRUE(commands->CmdBindRayTracingProgram(program.Get()));
  ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
  ASSERT_TRUE(commands->CmdDispatchRays(3, 1));
  token = queue->Submit(commands.Get());
  ASSERT_TRUE(token);
  commands.Release();
  group.Release();
  program.Release();
  ASSERT_TRUE(token->WaitHost());
  ReadBuffer(device.Get(), output.Get(), api, result);
  EXPECT_EQ(result, (std::array<uint32_t, 3>{112, 223, 334}));
  CheckNativeErrors(device.Get());
}

TEST_P(RhiRayTracing, CompactedLargeBlasSupportsInPlaceUpdate) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  DeviceFeatureRequirement required;
  required.ray_query = true;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection(required);
  if (!device)
    GTEST_SKIP();
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(queue);
  constexpr uint32_t count = 1024;
  std::vector<float> positions(count * 9);
  for (uint32_t i = 0; i < count; ++i) {
    float x = float(i % 32), y = float(i / 32);
    float triangle[] = {x, y, 0, x + 0.8f, y, 0, x, y + 0.8f, 0};
    std::memcpy(positions.data() + i * 9, triangle, sizeof(triangle));
  }
  ObjPtr<IBuffer> vertices = device->CreateBuffer({positions.size() * sizeof(float)});
  ASSERT_TRUE(vertices);
  BottomLevelAccelerationStructureDesc desc;
  desc.geometries = {{count * 3}};
  desc.allow_update = true;
  desc.allow_compaction = true;
  ObjPtr<IBottomLevelAccelerationStructure> original = device->CreateBottomLevelAccelerationStructure(desc);
  ASSERT_TRUE(original);
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands);
  ASSERT_TRUE(commands->CmdUploadBuffer(vertices.Get(), 0, positions.data(), positions.size() * sizeof(float)));
  ASSERT_TRUE(commands->CmdBuildBottomLevelAccelerationStructure(original.Get(), {{{vertices.Get()}, {}}}));
  ASSERT_TRUE(commands->CmdQueryCompactedSize(original.Get()));
  ObjPtr<ISubmissionToken> token = queue->Submit(commands.Get());
  ASSERT_TRUE(token);
  ASSERT_TRUE(token->WaitHost());
  commands.Release();
  desc.compacted_size = original->GetCompactedSize();
  ASSERT_GT(desc.compacted_size, 0u);
  ObjPtr<IBottomLevelAccelerationStructure> compact = device->CreateBottomLevelAccelerationStructure(desc);
  ASSERT_TRUE(compact);
  commands = device->CreateCommandContext();
  ASSERT_TRUE(commands);
  ASSERT_TRUE(commands->CmdCompactAccelerationStructure(original.Get(), compact.Get()));
  for (size_t i = 2; i < positions.size(); i += 3)
    positions[i] = 0.25f;
  ASSERT_TRUE(commands->CmdUploadBuffer(vertices.Get(), 0, positions.data(), positions.size() * sizeof(float)));
  ASSERT_TRUE(commands->CmdBuildBottomLevelAccelerationStructure(compact.Get(), {{{vertices.Get()}, {}}},
                                                                 ACCELERATION_STRUCTURE_UPDATE));
  token = queue->Submit(commands.Get());
  ASSERT_TRUE(token);
  ASSERT_TRUE(token->WaitHost());
  commands.Release();
  CheckNativeErrors(device.Get());
}

TEST_P(RhiRayTracing, RebuildsTlasWithChangingInstanceCountWithinCapacity) {
  const auto api = GetParam();
  ObjPtr<ICore> core = CreateCore(api, {true});
  ASSERT_TRUE(core);
  DeviceFeatureRequirement required;
  required.ray_query = true;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection(required);
  if (!device)
    GTEST_SKIP();
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(queue);
  BottomLevelAccelerationStructureDesc desc;
  desc.geometries = {{3}};
  ObjPtr<IBottomLevelAccelerationStructure> blas = device->CreateBottomLevelAccelerationStructure(desc);
  ASSERT_TRUE(blas);
  ObjPtr<IBuffer> vertices = device->CreateBuffer({36});
  ASSERT_TRUE(vertices);
  const float positions[] = {-1, -1, 0, 0, 1, 0, 1, -1, 0};
  ObjPtr<ITopLevelAccelerationStructure> tlas = device->CreateTopLevelAccelerationStructure({4, true});
  ASSERT_TRUE(tlas);
  ObjPtr<ICommandContext> build = device->CreateCommandContext();
  ASSERT_TRUE(build);
  ASSERT_TRUE(build->CmdUploadBuffer(vertices.Get(), 0, positions, sizeof(positions)));
  ASSERT_TRUE(build->CmdBuildBottomLevelAccelerationStructure(blas.Get(), {{{vertices.Get()}, {}}}));
  ObjPtr<ISubmissionToken> token = queue->Submit(build.Get());
  ASSERT_TRUE(token);
  ASSERT_TRUE(token->WaitHost());
  const char *source = R"(
RHI_ACCELERATION_STRUCTURE(0,0,scene);
RHI_RW_BYTE_ADDRESS_BUFFER(0,1,output);
[numthreads(1,1,1)] void CSMain(uint3 id : SV_DispatchThreadID) {
 RayDesc ray; ray.Origin=float3(id.x*3,0,-2); ray.Direction=float3(0,0,1); ray.TMin=0; ray.TMax=10;
 RayQuery<RAY_FLAG_NONE> query;
 query.TraceRayInline(scene,RAY_FLAG_NONE,255,ray);
 while(query.Proceed()) {}
 output.Store(id.x*4,query.CommittedStatus()==COMMITTED_TRIANGLE_HIT ? query.CommittedInstanceID()+1 : 0);
}
)";
  PipelineLayoutDesc layout{{{{{0, BINDING_ACCELERATION_STRUCTURE, 1, SHADER_VISIBILITY_COMPUTE},
                               {1, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE}}}}};
  ObjPtr<IShader> shader = device->CreateShaderFromSource(source, "CSMain", SHADER_STAGE_COMPUTE);
  ASSERT_TRUE(shader);
  ObjPtr<IComputeProgram> program = device->CreateComputeProgram({shader.Get(), layout});
  ASSERT_TRUE(program);
  ObjPtr<IBuffer> output = device->CreateBuffer({16});
  ASSERT_TRUE(output);
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(layout.groups[0]);
  ASSERT_TRUE(group);
  ASSERT_TRUE(group->BindResource(0, tlas.Get()));
  ASSERT_TRUE(group->BindResource(1, output.Get()));
  for (uint32_t count : {1u, 4u, 2u, 3u, 1u}) {
    std::vector<RayTracingInstance> instances(count);
    for (uint32_t i = 0; i < count; ++i) {
      instances[i].blas = blas.Get();
      instances[i].custom_index = i;
      instances[i].transform[3] = float(i * 3);
      instances[i].disable_triangle_culling = true;
    }
    build = device->CreateCommandContext();
    ASSERT_TRUE(build);
    ASSERT_TRUE(build->CmdBuildTopLevelAccelerationStructure(tlas.Get(), instances));
    ASSERT_TRUE(build->CmdBuildTopLevelAccelerationStructure(tlas.Get(), instances, ACCELERATION_STRUCTURE_UPDATE));
    ASSERT_TRUE(build->CmdBindComputeProgram(program.Get()));
    ASSERT_TRUE(build->CmdBindGroup(0, group.Get()));
    ASSERT_TRUE(build->CmdDispatch(4));
    token = queue->Submit(build.Get());
    ASSERT_TRUE(token);
    ASSERT_TRUE(token->WaitHost());
    std::array<uint32_t, 4> result{};
    ReadBuffer(device.Get(), output.Get(), api, result);
    for (uint32_t i = 0; i < 4; ++i)
      EXPECT_EQ(result[i], i < count ? i + 1 : 0);
  }
  RayTracingInstance instance;
  instance.blas = blas.Get();
  build = device->CreateCommandContext();
  ASSERT_TRUE(build);
  EXPECT_FALSE(
      build->CmdBuildTopLevelAccelerationStructure(tlas.Get(), {instance, instance}, ACCELERATION_STRUCTURE_UPDATE));
  build = device->CreateCommandContext();
  ASSERT_TRUE(build);
  EXPECT_FALSE(build->CmdBuildTopLevelAccelerationStructure(tlas.Get(), std::vector<RayTracingInstance>(5, instance)));
  CheckNativeErrors(device.Get());
}

TEST_P(RhiRayTracing, ScratchPoolSharesStorageAndShrinksAfterDemandExpires) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  DeviceFeatureRequirement required;
  required.ray_query = true;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection(required);
  if (!device)
    GTEST_SKIP();
  auto check = [](auto &pool) {
    auto initial_buffer = pool.Acquire(1);
    ASSERT_TRUE(initial_buffer);
    EXPECT_EQ(initial_buffer->GetDesc().size, 65536u);
    EXPECT_EQ(pool.Acquire(65536).Get(), initial_buffer.Get());
    auto large = pool.Acquire(700000);
    ASSERT_TRUE(large);
    EXPECT_EQ(large->GetDesc().size, 1048576u);
    EXPECT_EQ(pool.Acquire(1).Get(), large.Get());
    EXPECT_NE(large.Get(), initial_buffer.Get());
    EXPECT_FALSE(pool.Acquire(UINT64_MAX));
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    // Exactly 50% is not below the shrink threshold, and remains in the peak window.
    EXPECT_EQ(pool.Acquire(524288).Get(), large.Get());
    EXPECT_EQ(pool.Acquire(1).Get(), large.Get());
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    auto shrunk = pool.Acquire(65537);
    ASSERT_TRUE(shrunk);
    EXPECT_EQ(shrunk->GetDesc().size, 131072u);
    EXPECT_NE(shrunk.Get(), large.Get());
    EXPECT_EQ(pool.Acquire(1).Get(), shrunk.Get());
    // Old borrowers remain usable after replacement and clearing the cache.
    EXPECT_TRUE(large->NativeBuffer());
    EXPECT_TRUE(initial_buffer->NativeBuffer());
    pool.Clear();
    EXPECT_TRUE(shrunk->NativeBuffer());
    auto next = pool.Acquire(1);
    ASSERT_TRUE(next);
    EXPECT_EQ(next->GetDesc().size, 65536u);
    EXPECT_NE(next.Get(), shrunk.Get());
  };
#if defined(LONGMARCH_D3D12_ENABLED)
  if (GetParam() == BACKEND_API_D3D12)
    check(static_cast<backend::d3d12::D3D12Device *>(device.Get())->ScratchPool());
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
  if (GetParam() == BACKEND_API_VULKAN)
    check(static_cast<backend::vulkan::VulkanDevice *>(device.Get())->ScratchPool());
#endif
#if defined(LONGMARCH_METAL_ENABLED)
  if (GetParam() == BACKEND_API_METAL)
    check(static_cast<backend::metal::MetalDevice *>(device.Get())->ScratchPool());
#endif
  CheckNativeErrors(device.Get());
}

TEST_P(RhiRayTracing, SharedScratchSurvivesResizingAndOutOfOrderContextSubmission) {
  const auto api = GetParam();
  ObjPtr<ICore> core = CreateCore(api, {true});
  ASSERT_TRUE(core);
  DeviceFeatureRequirement required;
  required.ray_query = true;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection(required);
  if (!device)
    GTEST_SKIP();
  ObjPtr<IQueue> graphics = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IQueue> transfer = device->GetQueue(QUEUE_TYPE_TRANSFER);
  ASSERT_TRUE(graphics);
  ASSERT_TRUE(transfer);
  constexpr uint32_t count = 32;
  const float positions[] = {-1, -1, 0, 0, 1, 0, 1, -1, 0};
  ObjPtr<IBuffer> vertices = device->CreateBuffer({sizeof(positions)});
  ASSERT_TRUE(vertices);
  ObjPtr<ICommandContext> upload = device->CreateCommandContext();
  ASSERT_TRUE(upload);
  ASSERT_TRUE(upload->CmdUploadBuffer(vertices.Get(), 0, positions, sizeof(positions)));
  ObjPtr<ISubmissionToken> uploaded = transfer->Submit(upload.Get());
  ASSERT_TRUE(uploaded);
  upload.Release();
  ASSERT_TRUE(uploaded->WaitHost());
  BottomLevelAccelerationStructureDesc desc;
  desc.geometries = {{3}};
  std::array<ObjPtr<IBottomLevelAccelerationStructure>, count> blases;
  std::array<ObjPtr<ICommandContext>, 4> contexts;
  for (uint32_t batch = 0; batch < contexts.size(); ++batch) {
    contexts[batch] = device->CreateCommandContext();
    ASSERT_TRUE(contexts[batch]);
    for (uint32_t i = batch * 8; i < (batch + 1) * 8; ++i) {
      blases[i] = device->CreateBottomLevelAccelerationStructure(desc);
      ASSERT_TRUE(blases[i]);
      ASSERT_TRUE(contexts[batch]->CmdBuildBottomLevelAccelerationStructure(blases[i].Get(), {{{vertices.Get()}, {}}}));
    }
    if (batch == 1) {
      // Replace the cache while unsubmitted commands still reference the old buffer.
#if defined(LONGMARCH_METAL_ENABLED)
      if (api == BACKEND_API_METAL)
        ASSERT_TRUE(static_cast<backend::metal::MetalDevice *>(device.Get())->ScratchPool().Acquire(4 * 1024 * 1024));
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
      if (api == BACKEND_API_D3D12)
        ASSERT_TRUE(static_cast<backend::d3d12::D3D12Device *>(device.Get())->ScratchPool().Acquire(4 * 1024 * 1024));
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
      if (api == BACKEND_API_VULKAN)
        ASSERT_TRUE(static_cast<backend::vulkan::VulkanDevice *>(device.Get())->ScratchPool().Acquire(4 * 1024 * 1024));
#endif
    }
  }
  std::vector<ObjPtr<ISubmissionToken>> tokens;
  for (uint32_t i : {3u, 1u, 0u, 2u}) {
    auto queue = i % 2 ? transfer.Get() : graphics.Get();
    tokens.emplace_back(queue->Submit(contexts[i].Get()));
    ASSERT_TRUE(tokens.back());
    contexts[i].Release();
  }
  ObjPtr<ITopLevelAccelerationStructure> tlas = device->CreateTopLevelAccelerationStructure({count});
  ASSERT_TRUE(tlas);
  std::vector<RayTracingInstance> instances(count);
  for (uint32_t i = 0; i < count; ++i) {
    instances[i].blas = blases[i].Get();
    instances[i].transform[3] = float(i * 3);
    instances[i].custom_index = i;
    instances[i].disable_triangle_culling = true;
  }
  const char *source = R"(
RHI_ACCELERATION_STRUCTURE(0,0,scene);
RHI_RW_BYTE_ADDRESS_BUFFER(0,1,output);
[numthreads(1,1,1)] void CSMain(uint3 id : SV_DispatchThreadID) {
 RayDesc ray; ray.Origin=float3(id.x*3,0,-2); ray.Direction=float3(0,0,1); ray.TMin=0; ray.TMax=10;
 RayQuery<RAY_FLAG_NONE> query;
 query.TraceRayInline(scene,RAY_FLAG_NONE,255,ray);
 while(query.Proceed()) {}
 output.Store(id.x*4,query.CommittedStatus()==COMMITTED_TRIANGLE_HIT ? query.CommittedInstanceID()+1 : 0);
}
)";
  PipelineLayoutDesc layout{{{{{0, BINDING_ACCELERATION_STRUCTURE, 1, SHADER_VISIBILITY_COMPUTE},
                               {1, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE}}}}};
  ObjPtr<IShader> shader = device->CreateShaderFromSource(source, "CSMain", SHADER_STAGE_COMPUTE);
  ASSERT_TRUE(shader);
  ObjPtr<IComputeProgram> program = device->CreateComputeProgram({shader.Get(), layout});
  ASSERT_TRUE(program);
  ObjPtr<IBuffer> output = device->CreateBuffer({count * sizeof(uint32_t)});
  ASSERT_TRUE(output);
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(layout.groups[0]);
  ASSERT_TRUE(group);
  ASSERT_TRUE(group->BindResource(0, tlas.Get()));
  ASSERT_TRUE(group->BindResource(1, output.Get()));
  ObjPtr<ICommandContext> query = device->CreateCommandContext();
  ASSERT_TRUE(query);
  ASSERT_TRUE(query->CmdBuildTopLevelAccelerationStructure(tlas.Get(), instances));
  ASSERT_TRUE(query->CmdBindComputeProgram(program.Get()));
  ASSERT_TRUE(query->CmdBindGroup(0, group.Get()));
  ASSERT_TRUE(query->CmdDispatch(count));
  ObjPtr<ISubmissionToken> completed = graphics->Submit(query.Get());
  ASSERT_TRUE(completed);
  query.Release();
  ASSERT_TRUE(completed->WaitHost());
  for (const auto &token : tokens)
    ASSERT_TRUE(token->WaitHost());
  std::array<uint32_t, count> hits{};
  ReadBuffer(device.Get(), output.Get(), api, hits);
  for (uint32_t i = 0; i < count; ++i)
    EXPECT_EQ(hits[i], i + 1);
  CheckNativeErrors(device.Get());
}

TEST_P(RhiRayTracing, ProgramCreationRejectsInvalidDescriptionAndShader) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  DeviceFeatureRequirement required;
  required.ray_tracing_pipeline = true;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection(required);
  if (!device)
    GTEST_SKIP();
  ObjPtr<IRayTracingProgram> program = device->CreateRayTracingProgram({});
  EXPECT_FALSE(program);
  ObjPtr<IShader> invalid =
      device->CreateShaderFromSource("invalid shader code", "RayGen", SHADER_STAGE_RAY_GENERATION);
  ASSERT_TRUE(invalid);
  RayTracingProgramDesc desc;
  desc.ray_generation = invalid.Get();
  program = device->CreateRayTracingProgram(desc);
  EXPECT_FALSE(program);
  ObjPtr<IShader> raygen = device->CreateShaderFromSource("[shader(\"raygeneration\")] void RayGen() {}", "RayGen",
                                                          SHADER_STAGE_RAY_GENERATION);
  ASSERT_TRUE(raygen);
  desc.ray_generation = raygen.Get();
  desc.max_recursion_depth = 0;
  program = device->CreateRayTracingProgram(desc);
  EXPECT_FALSE(program);
  desc.max_recursion_depth = 1;
  program = device->CreateRayTracingProgram(desc);
  ASSERT_TRUE(program);
  raygen.Release();
  ObjPtr<IShaderBindingTable> table = program->CreateShaderBindingTable({});
  ASSERT_TRUE(table);
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands);
  ASSERT_TRUE(commands->CmdBindRayTracingProgram(program.Get()));
  ASSERT_TRUE(commands->CmdDispatchRays(1, 1, 1, table.Get()));
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(queue);
  ObjPtr<ISubmissionToken> token = queue->Submit(commands.Get());
  ASSERT_TRUE(token);
  commands.Release();
  ASSERT_TRUE(token->WaitHost());
  CheckNativeErrors(device.Get());
}

constexpr BackendAPI backends[] = {
#if defined(LONGMARCH_METAL_ENABLED)
    BACKEND_API_METAL,
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
    BACKEND_API_D3D12,
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    BACKEND_API_VULKAN,
#endif
};
INSTANTIATE_TEST_SUITE_P(Backends, RhiRayTracing, testing::ValuesIn(backends));
}  // namespace
}  // namespace grassland::rhi
#endif
