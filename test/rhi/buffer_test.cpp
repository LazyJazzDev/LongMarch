#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstring>
#include <thread>

#include "grassland/rhi/rhi.h"
#if defined(LONGMARCH_METAL_ENABLED)
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_image.h"
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
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

class RhiBuffer : public testing::TestWithParam<BackendAPI> {};

TEST_P(RhiBuffer, UploadSnapshotsCpuDataAndCopySurvivesCallerRelease) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  EXPECT_EQ(device->CreateBuffer({0}), nullptr);
  ObjPtr<IQueue> transfer = device->GetQueue(QUEUE_TYPE_TRANSFER);
  ASSERT_TRUE(transfer);
  ObjPtr<IQueue> graphics = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(graphics);
  std::array<uint32_t, 16> expected{};
  for (uint32_t i = 0; i < expected.size(); ++i)
    expected[i] = i * 137 + 5;
  ObjPtr<IBuffer> source = device->CreateBuffer({sizeof(expected)});
  ASSERT_TRUE(source);
  ObjPtr<IBuffer> target = device->CreateBuffer({sizeof(expected)});
  ASSERT_TRUE(target);
  ObjPtr<ICommandContext> upload = device->CreateCommandContext();
  ASSERT_TRUE(upload);
  auto bytes = expected;
  ASSERT_TRUE(upload->CmdUploadBuffer(source.Get(), 0, bytes.data(), sizeof(bytes)));
  bytes.fill(0);
  ObjPtr<ISubmissionToken> ready = transfer->Submit(upload.Get());
  ASSERT_TRUE(ready);
  upload.Release();
  ObjPtr<ICommandContext> copy = device->CreateCommandContext();
  ASSERT_TRUE(copy);
  ASSERT_TRUE(copy->CmdCopyBuffer(source.Get(), 0, target.Get(), 0, sizeof(expected)));
  source.Release();
  SubmitInfo info;
  info.waits = {ready.Get()};
  ObjPtr<ISubmissionToken> done = graphics->Submit(copy.Get(), info);
  ASSERT_TRUE(done);
  copy.Release();
  ASSERT_TRUE(done->WaitHost());
  std::array<uint32_t, 16> actual{};
  ReadBuffer(device.Get(), target.Get(), GetParam(), actual);
  EXPECT_EQ(actual, expected);
}

TEST_P(RhiBuffer, AbandonedUploadsReturnStorageAndLeasesDoNotOverlap) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IBuffer> target = device->CreateBuffer({64});
  ASSERT_TRUE(target);
  auto check = [&](auto &pool) {
    auto first = pool.Acquire(64);
    ASSERT_TRUE(first);
    auto identity = first->NativeBuffer();
    auto second = pool.Acquire(64);
    ASSERT_TRUE(second);
    EXPECT_NE(identity, second->NativeBuffer());
    first.Reset();
    {
      ObjPtr<ICommandContext> context = device->CreateCommandContext();
      std::array<uint32_t, 16> bytes{};
      ASSERT_TRUE(context->CmdUploadBuffer(target.Get(), 0, bytes.data(), sizeof(bytes)));
    }
    auto reused = pool.Acquire(64);
    ASSERT_TRUE(reused);
    EXPECT_EQ(identity, reused->NativeBuffer());
    reused.Reset();
    second.Reset();
  };
#if defined(LONGMARCH_D3D12_ENABLED)
  if (GetParam() == BACKEND_API_D3D12)
    check(static_cast<backend::d3d12::D3D12Device *>(device.Get())->UploadPool());
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
  if (GetParam() == BACKEND_API_VULKAN)
    check(static_cast<backend::vulkan::VulkanDevice *>(device.Get())->UploadPool());
#endif
#if defined(LONGMARCH_METAL_ENABLED)
  if (GetParam() == BACKEND_API_METAL)
    check(static_cast<backend::metal::MetalDevice *>(device.Get())->UploadPool());
#endif
}

TEST_P(RhiBuffer, UploadClassesReuseIndependentlyAndReleaseExpiredPeakCapacity) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto check = [&](auto &pool) {
    pool.Clear();
    EXPECT_FALSE(pool.Acquire(UINT64_MAX));
    auto large = pool.Acquire(65537);
    ASSERT_TRUE(large);
    EXPECT_EQ(large->GetDesc().size, 131072u);
    auto large_identity = large->NativeBuffer();
    large.Reset();
    auto first = pool.Acquire(1);
    ASSERT_TRUE(first);
    EXPECT_EQ(first->GetDesc().size, 65536u);
    auto second = pool.Acquire(65536);
    auto third = pool.Acquire(64);
    auto fourth = pool.Acquire(64);
    ASSERT_TRUE(second);
    ASSERT_TRUE(third);
    ASSERT_TRUE(fourth);
    auto third_identity = third->NativeBuffer();
    first.Reset();
    second.Reset();
    third.Reset();
    EXPECT_EQ(pool.GetBufferCounts(64).first, 4u);
    EXPECT_EQ(pool.GetBufferCounts(64).second, 1u);
    auto reused = pool.Acquire(32);
    ASSERT_TRUE(reused);
    EXPECT_EQ(reused->NativeBuffer(), third_identity);
    reused.Reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    auto recent = pool.Acquire(64);
    ASSERT_TRUE(recent);
    fourth.Reset();
    EXPECT_EQ(pool.GetBufferCounts(64).first, 2u);
    EXPECT_EQ(pool.GetBufferCounts(64).second, 1u);
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    recent.Reset();
    EXPECT_EQ(pool.GetBufferCounts(64).first, 0u);
    EXPECT_EQ(pool.GetBufferCounts(64).second, 0u);
    EXPECT_EQ(pool.GetBufferCounts(65537).first, 1u);
    auto again = pool.Acquire(65537);
    ASSERT_TRUE(again);
    EXPECT_EQ(again->NativeBuffer(), large_identity);
    pool.Clear();  // Only idle storage may be cleared while a lease exists.
    EXPECT_EQ(pool.GetBufferCounts(65537).first, 1u);
    EXPECT_EQ(again->NativeBuffer(), large_identity);
    again.Reset();
    pool.Clear();
    EXPECT_EQ(pool.GetBufferCounts(65537).first, 0u);
  };
#if defined(LONGMARCH_D3D12_ENABLED)
  if (GetParam() == BACKEND_API_D3D12)
    check(static_cast<backend::d3d12::D3D12Device *>(device.Get())->UploadPool());
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
  if (GetParam() == BACKEND_API_VULKAN)
    check(static_cast<backend::vulkan::VulkanDevice *>(device.Get())->UploadPool());
#endif
#if defined(LONGMARCH_METAL_ENABLED)
  if (GetParam() == BACKEND_API_METAL)
    check(static_cast<backend::metal::MetalDevice *>(device.Get())->UploadPool());
#endif
}

TEST_P(RhiBuffer, UploadLeaseForwardsBufferAndReturnsAfterLastCommandReference) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_TRANSFER);
  ASSERT_TRUE(queue);
  std::array<uint32_t, 16> expected{};
  for (uint32_t i = 0; i < expected.size(); ++i)
    expected[i] = i * 31 + 7;
  ObjPtr<IBuffer> target = device->CreateBuffer({sizeof(expected)});
  ASSERT_TRUE(target);
  auto check = [&](auto &pool) {
    auto lease = pool.Acquire(sizeof(expected));
    ASSERT_TRUE(lease);
    EXPECT_EQ(lease->Owner(), device.Get());
    ASSERT_TRUE(lease->Write(expected.data(), sizeof(expected)));
#if defined(LONGMARCH_D3D12_ENABLED)
    if (GetParam() == BACKEND_API_D3D12) {
      auto upload = dynamic_cast<backend::d3d12::D3D12UploadPool::Lease *>(lease.Get());
      ASSERT_TRUE(upload);
      expected[7] = 12345;
      ASSERT_TRUE(upload->Write(&expected[7], sizeof(uint32_t), 7 * sizeof(uint32_t)));
      EXPECT_FALSE(upload->Write(expected.data(), sizeof(uint32_t), upload->GetDesc().size - 1));
      EXPECT_FALSE(upload->Write(expected.data(), sizeof(uint32_t), UINT64_MAX));
    }
#endif
    auto alias = lease;
    lease.Reset();
    EXPECT_EQ(pool.GetBufferCounts(sizeof(expected)).second, 1u);
    ObjPtr<ICommandContext> commands = device->CreateCommandContext();
    ASSERT_TRUE(commands);
    ASSERT_TRUE(commands->CmdCopyBuffer(alias.Get(), 0, target.Get(), 0, sizeof(expected)));
    alias.Reset();
    // The common resource table keeps the lease occupied after all caller references go away.
    EXPECT_EQ(pool.GetBufferCounts(sizeof(expected)).second, 1u);
    ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
    ASSERT_TRUE(done);
    commands.Release();
    ASSERT_TRUE(done->WaitHost());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (pool.GetBufferCounts(sizeof(expected)).second && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_EQ(pool.GetBufferCounts(sizeof(expected)).second, 0u);
    std::array<uint32_t, 16> actual{};
    ReadBuffer(device.Get(), target.Get(), GetParam(), actual);
    EXPECT_EQ(actual, expected);
  };
#if defined(LONGMARCH_D3D12_ENABLED)
  if (GetParam() == BACKEND_API_D3D12)
    check(static_cast<backend::d3d12::D3D12Device *>(device.Get())->UploadPool());
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
  if (GetParam() == BACKEND_API_VULKAN)
    check(static_cast<backend::vulkan::VulkanDevice *>(device.Get())->UploadPool());
#endif
#if defined(LONGMARCH_METAL_ENABLED)
  if (GetParam() == BACKEND_API_METAL)
    check(static_cast<backend::metal::MetalDevice *>(device.Get())->UploadPool());
#endif
}

const BackendAPI backends[] = {
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
TEST_P(RhiBuffer, EightBindingGroupsDispatchAndRejectNinthGroup) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(queue);
  std::string source;
  std::string body;
  ComputeProgramDesc desc;
  for (uint32_t i = 0; i < 8; ++i) {
    const auto index = std::to_string(i);
    source += "RHI_RW_STRUCTURED_BUFFER(" + index + ", 0, uint, values" + index + ");\n";
    body += "values" + index + "[0] = " + std::to_string(i + 1) + ";\n";
    desc.pipeline_layout.groups.push_back({{{0, BINDING_RW_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_COMPUTE, 4}}});
  }
  source += "[numthreads(1, 1, 1)] void CSMain() {\n" + body + "}\n";
  ObjPtr<IShader> shader = device->CreateShader({source, "CSMain", SHADER_STAGE_COMPUTE});
  ASSERT_TRUE(shader);
  desc.compute_shader = shader.Get();
  ObjPtr<IComputeProgram> program = device->CreateComputeProgram(desc);
  ASSERT_TRUE(program);
  ASSERT_TRUE(program->Finalize());
  ObjPtr<ICommandContext> context = device->CreateCommandContext();
  ASSERT_TRUE(context);
  ASSERT_TRUE(context->CmdBindComputeProgram(program.Get()));
  std::array<ObjPtr<IBuffer>, 8> buffers;
  for (uint32_t i = 0; i < buffers.size(); ++i) {
    buffers[i] = device->CreateBuffer({4});
    ASSERT_TRUE(buffers[i]);
    ObjPtr<IBindingGroup> group = device->CreateBindingGroup(desc.pipeline_layout.groups[i]);
    ASSERT_TRUE(group);
    ASSERT_TRUE(group->BindResource(0, buffers[i].Get()));
    ASSERT_TRUE(context->CmdBindGroup(i, group.Get()));
  }
  ASSERT_TRUE(context->CmdDispatch(1));
  ObjPtr<ISubmissionToken> token = queue->Submit(context.Get());
  ASSERT_TRUE(token);
  ASSERT_TRUE(token->WaitHost());
  for (uint32_t i = 0; i < buffers.size(); ++i) {
    std::array<uint32_t, 1> actual{};
    ReadBuffer(device.Get(), buffers[i].Get(), GetParam(), actual);
    EXPECT_EQ(actual[0], i + 1);
  }
  desc.pipeline_layout.groups.push_back(desc.pipeline_layout.groups.front());
  ObjPtr<IComputeProgram> invalid = device->CreateComputeProgram(desc);
  EXPECT_TRUE(!invalid || !invalid->Finalize());
}

TEST_P(RhiBuffer, ComputeDispatchDependenciesAndBindingSnapshots) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(queue);
  const std::string source = R"(
    RHI_RW_STRUCTURED_BUFFER(0, 0, uint, values);
    [numthreads(4, 1, 1)] void CSMain(uint3 id : SV_DispatchThreadID) {
      values[id.x] = values[id.x] * 2 + 1;
    }
  )";
  ObjPtr<IShader> shader = device->CreateShader({source, "CSMain", SHADER_STAGE_COMPUTE});
  ASSERT_TRUE(shader);
  ComputeProgramDesc desc;
  desc.compute_shader = shader.Get();
  desc.pipeline_layout.groups = {{{{0, BINDING_RW_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_COMPUTE, 4}}}};
  ObjPtr<IComputeProgram> program = device->CreateComputeProgram(desc);
  ASSERT_TRUE(program);
  shader.Release();
  ASSERT_TRUE(program->Finalize());
  EXPECT_TRUE(program->Finalize());
  ObjPtr<IBuffer> first = device->CreateBuffer({64});
  ObjPtr<IBuffer> second = device->CreateBuffer({64});
  ObjPtr<IBuffer> output = device->CreateBuffer({64});
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  ASSERT_TRUE(output);
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ASSERT_TRUE(group);
  ASSERT_TRUE(group->BindResource(0, first.Get()));
  std::array<uint32_t, 16> input{};
  for (uint32_t i = 0; i < 16; ++i)
    input[i] = i;
  ObjPtr<ICommandContext> ctx = device->CreateCommandContext();
  ASSERT_TRUE(ctx);
  ASSERT_TRUE(ctx->CmdUploadBuffer(first.Get(), 0, input.data(), sizeof(input)));
  ASSERT_TRUE(ctx->CmdUploadBuffer(second.Get(), 0, input.data(), sizeof(input)));
  ASSERT_TRUE(ctx->CmdBindComputeProgram(program.Get()));
  ASSERT_TRUE(ctx->CmdBindGroup(0, group.Get()));
  // The captured first version must remain bound even if the frontend group changes.
  ASSERT_TRUE(group->BindResource(0, second.Get()));
  ASSERT_TRUE(ctx->CmdDispatch(4));
  ASSERT_TRUE(ctx->CmdDispatch(4));
  ASSERT_TRUE(ctx->CmdCopyBuffer(first.Get(), 0, output.Get(), 0, 64));
  ASSERT_TRUE(ctx->CmdBindGroup(0, group.Get()));
  ASSERT_TRUE(ctx->CmdDispatch(4));
  first.Release();
  group.Release();
  program.Release();
  ObjPtr<ISubmissionToken> token = queue->Submit(ctx.Get());
  ASSERT_TRUE(token);
  ASSERT_TRUE(token->WaitHost());
  std::array<uint32_t, 16> actual{};
  ReadBuffer(device.Get(), output.Get(), GetParam(), actual);
  for (uint32_t i = 0; i < 16; ++i)
    EXPECT_EQ(actual[i], i * 4 + 3);
  ReadBuffer(device.Get(), second.Get(), GetParam(), actual);
  for (uint32_t i = 0; i < 16; ++i)
    EXPECT_EQ(actual[i], i * 2 + 1);
  // A new context sees the completed writes in the same physical buffer.
  shader = device->CreateShader({source, "CSMain", SHADER_STAGE_COMPUTE});
  ASSERT_TRUE(shader);
  program = device->CreateComputeProgram({shader.Get(), desc.pipeline_layout});
  ASSERT_TRUE(program);
  ASSERT_TRUE(program->Finalize());
  group = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ASSERT_TRUE(group);
  ASSERT_TRUE(group->BindResource(0, second.Get()));
  ctx = device->CreateCommandContext();
  ASSERT_TRUE(ctx);
  ASSERT_TRUE(ctx->CmdBindComputeProgram(program.Get()));
  ASSERT_TRUE(ctx->CmdBindGroup(0, group.Get()));
  ASSERT_TRUE(ctx->CmdDispatch(4));
  token = queue->Submit(ctx.Get());
  ASSERT_TRUE(token);
  ASSERT_TRUE(token->WaitHost());
  ReadBuffer(device.Get(), second.Get(), GetParam(), actual);
  for (uint32_t i = 0; i < 16; ++i)
    EXPECT_EQ(actual[i], i * 4 + 3);
}

TEST_P(RhiBuffer, ComputeDispatchUsesReflectedThreeDimensionalThreadgroups) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(queue);
  ObjPtr<IShader> shader = device->CreateShader({R"(
    RHI_RW_STRUCTURED_BUFFER(0, 0, uint, values);
    [numthreads(2, 3, 2)] void CSMain(uint3 id : SV_DispatchThreadID) {
      uint index = id.x + 4 * (id.y + 6 * id.z);
      values[index] += index + 1;
    }
  )",
                                                 "CSMain", SHADER_STAGE_COMPUTE});
  ComputeProgramDesc desc;
  desc.compute_shader = shader.Get();
  desc.pipeline_layout.groups = {{{{0, BINDING_RW_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_COMPUTE, 4}}}};
  ObjPtr<IComputeProgram> program = device->CreateComputeProgram(desc);
  ASSERT_TRUE(program);
  ASSERT_TRUE(program->Finalize());
  std::array<uint32_t, 96> values{};
  ObjPtr<IBuffer> buffer = device->CreateBuffer({sizeof(values)});
  ASSERT_TRUE(buffer);
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ASSERT_TRUE(group);
  ASSERT_TRUE(group->BindResource(0, buffer.Get()));
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands);
  ASSERT_TRUE(commands->CmdUploadBuffer(buffer.Get(), 0, values.data(), sizeof(values)));
  ASSERT_TRUE(commands->CmdBindComputeProgram(program.Get()));
  ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
  ASSERT_TRUE(commands->CmdDispatch(2, 2, 2));
  ASSERT_TRUE(commands->CmdDispatch(0, 2, 2));
  ASSERT_TRUE(commands->CmdDispatch(2, 0, 2));
  ASSERT_TRUE(commands->CmdDispatch(2, 2, 0));
  ObjPtr<ISubmissionToken> token = queue->Submit(commands.Get());
  ASSERT_TRUE(token);
  ASSERT_TRUE(token->WaitHost());
  ReadBuffer(device.Get(), buffer.Get(), GetParam(), values);
  for (size_t i = 0; i < values.size(); ++i)
    EXPECT_EQ(values[i], i + 1);
}

TEST_P(RhiBuffer, ComputeFinalizeRejectsWrongStageVisibilityAndStride) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IShader> shader = device->CreateShader({R"(
    RHI_RW_STRUCTURED_BUFFER(0, 0, float3, values);
    [numthreads(1, 1, 1)] void CSMain() { values[0] = float3(1, 2, 3); }
  )",
                                                 "CSMain", SHADER_STAGE_COMPUTE});
  ASSERT_TRUE(shader);
  ComputeProgramDesc desc;
  desc.compute_shader = shader.Get();
  desc.pipeline_layout.groups = {{{{0, BINDING_RW_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_VERTEX, 12}}}};
  ObjPtr<IComputeProgram> program = device->CreateComputeProgram(desc);
  ASSERT_TRUE(program);
  EXPECT_FALSE(program->Finalize());
  desc.pipeline_layout.groups[0].bindings[0].visibility = SHADER_VISIBILITY_COMPUTE;
  desc.pipeline_layout.groups[0].bindings[0].stride = 16;
  program = device->CreateComputeProgram(desc);
  ASSERT_TRUE(program);
  EXPECT_FALSE(program->Finalize());
  desc.pipeline_layout.groups[0].bindings[0].stride = 12;
  program = device->CreateComputeProgram(desc);
  ASSERT_TRUE(program);
  EXPECT_TRUE(program->Finalize());
  ObjPtr<IShader> vertex =
      device->CreateShader({"float4 VSMain():SV_Position { return 0; }", "VSMain", SHADER_STAGE_VERTEX});
  desc.compute_shader = vertex.Get();
  program = device->CreateComputeProgram(desc);
  ASSERT_TRUE(program);
  EXPECT_FALSE(program->Finalize());
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiBuffer, testing::ValuesIn(backends));
}  // namespace
}  // namespace grassland::rhi

#endif
