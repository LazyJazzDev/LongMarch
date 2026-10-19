#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstring>
#include <thread>

#include "grassland/rhi/rhi.h"
#if defined(LONGMARCH_D3D12_ENABLED)
#include <d3d12sdklayers.h>

#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_image.h"
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_image.h"
#endif
#if defined(LONGMARCH_D3D12_ENABLED) || defined(LONGMARCH_VULKAN_ENABLED)
namespace grassland::rhi {
namespace {
void ReadPixel(IDevice *device, IImage *image, BackendAPI api, uint32_t x, uint32_t y, std::array<uint8_t, 4> &output) {
#if defined(LONGMARCH_D3D12_ENABLED)
  if (api == BACKEND_API_D3D12) {
    using namespace backend::d3d12;
    auto owner = static_cast<D3D12Device *>(device);
    auto source = static_cast<D3D12Image *>(image)->NativeImage();
    auto native = owner->NativeDevice();
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
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
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = readback.Get();
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint.Footprint = {source->GetDesc().Format, 1, 1, 1, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT};
    D3D12_TEXTURE_COPY_LOCATION origin{};
    origin.pResource = source;
    origin.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_BOX box{x, y, 0, x + 1, y + 1, 1};
    list->CopyTextureRegion(&destination, 0, 0, 0, &origin, &box);
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
    auto source = static_cast<VulkanImage *>(image)->NativeImage();
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
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = source;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageOffset = {static_cast<int32_t>(x), static_cast<int32_t>(y), 0};
    copy.imageExtent = {1, 1, 1};
    vkCmdCopyImageToBuffer(commands, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &copy);
    std::swap(barrier.oldLayout, barrier.newLayout);
    std::swap(barrier.srcAccessMask, barrier.dstAccessMask);
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
    VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    host.srcQueueFamilyIndex = host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    host.buffer = readback;
    host.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &host,
                         0, nullptr);
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

class RhiImage : public testing::TestWithParam<BackendAPI> {};

TEST_P(RhiImage, OffscreenClearDrawReadbackAndReuse) {
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
  EXPECT_EQ(device->CreateImage({0, 64, IMAGE_FORMAT_RGBA8_UNORM}), nullptr);
  EXPECT_EQ(device->CreateImage({64, 64, IMAGE_FORMAT_UNKNOWN}), nullptr);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(queue);
  const std::string source = R"(
    float4 VSMain(uint id : SV_VertexID) : SV_Position {
      float2 positions[3] = {float2(0,0.75),float2(-0.75,-0.75),float2(0.75,-0.75)};
      return float4(positions[id],0,1);
    }
    float4 PSMain() : SV_Target0 { return float4(1,0,0,1); }
  )";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
  ASSERT_TRUE(vs);
  ASSERT_TRUE(ps);
  for (auto format : {IMAGE_FORMAT_RGBA8_UNORM, IMAGE_FORMAT_BGRA8_UNORM}) {
    ObjPtr<IImage> image = device->CreateImage({64, 64, format});
    ASSERT_TRUE(image);
    ProgramDesc desc;
    desc.vertex_shader = vs.Get();
    desc.fragment_shader = ps.Get();
    desc.color_format = format;
    ObjPtr<IProgram> program = device->CreateProgram(desc);
    ASSERT_TRUE(program);
    ASSERT_TRUE(program->Finalize());
    for (int frame = 0; frame < 3; ++frame) {
      ObjPtr<ICommandContext> commands = device->CreateCommandContext();
      ASSERT_TRUE(commands);
      ASSERT_TRUE(commands->CmdClearColor(image.Get(), {0.6f, 0.7f, 0.8f, 1}));
      ASSERT_TRUE(commands->CmdBindProgram(program.Get()));
      ASSERT_TRUE(commands->CmdBeginRendering({image.Get(), ATTACHMENT_LOAD_OP_LOAD, ATTACHMENT_STORE_OP_STORE}));
      ASSERT_TRUE(commands->CmdSetViewport({0, 0, 64, 64, 0, 1}));
      ASSERT_TRUE(commands->CmdSetScissor({0, 0, 64, 64}));
      ASSERT_TRUE(commands->CmdDraw(3));
      ASSERT_TRUE(commands->CmdEndRendering());
      ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
      ASSERT_TRUE(done);
      commands.Release();
      ASSERT_TRUE(done->WaitHost());
      std::array<uint8_t, 4> pixel{};
      ReadPixel(device.Get(), image.Get(), GetParam(), 32, 32, pixel);
      const size_t red = format == IMAGE_FORMAT_RGBA8_UNORM ? 0 : 2;
      const size_t blue = 2 - red;
      EXPECT_EQ(pixel[red], 255);
      EXPECT_EQ(pixel[1], 0);
      EXPECT_EQ(pixel[blue], 0);
      EXPECT_EQ(pixel[3], 255);
      ReadPixel(device.Get(), image.Get(), GetParam(), 0, 0, pixel);
      EXPECT_NEAR(pixel[red], 153, 1);
      EXPECT_NEAR(pixel[1], 179, 1);
      EXPECT_NEAR(pixel[blue], 204, 1);
      EXPECT_EQ(pixel[3], 255);
    }
  }
}

const BackendAPI backends[] = {
#if defined(LONGMARCH_D3D12_ENABLED)
    BACKEND_API_D3D12,
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    BACKEND_API_VULKAN,
#endif
};
INSTANTIATE_TEST_SUITE_P(Backends, RhiImage, testing::ValuesIn(backends));
}  // namespace
}  // namespace grassland::rhi

#endif
