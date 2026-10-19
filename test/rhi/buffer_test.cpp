#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstring>
#include <thread>

#include "grassland/rhi/rhi.h"
#if defined(LONGMARCH_D3D12_ENABLED)
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#endif
#if defined(LONGMARCH_D3D12_ENABLED) || defined(LONGMARCH_VULKAN_ENABLED)
namespace grassland::rhi {
namespace {
void ReadBuffer(IDevice *device, IBuffer *buffer, BackendAPI api, std::array<uint32_t, 16> &output) {
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
}

const BackendAPI backends[] = {
#if defined(LONGMARCH_D3D12_ENABLED)
    BACKEND_API_D3D12,
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    BACKEND_API_VULKAN,
#endif
};
INSTANTIATE_TEST_SUITE_P(Backends, RhiBuffer, testing::ValuesIn(backends));
}  // namespace
}  // namespace grassland::rhi

#endif
