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
#include <d3d12sdklayers.h>

#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_image.h"
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_image.h"
#endif
#if defined(LONGMARCH_D3D12_ENABLED) || defined(LONGMARCH_VULKAN_ENABLED) || defined(LONGMARCH_METAL_ENABLED)
namespace grassland::rhi {
namespace {
void ReadPixel(IDevice *device, IImage *image, BackendAPI api, uint32_t x, uint32_t y, std::array<uint8_t, 4> &output) {
#if defined(LONGMARCH_METAL_ENABLED)
  if (api == BACKEND_API_METAL) {
    using namespace backend::metal;
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto owner = static_cast<MetalDevice *>(device);
    auto readback = NS::TransferPtr(owner->NativeDevice()->newBuffer(256, MTL::ResourceStorageModeShared));
    ASSERT_TRUE(readback);
    std::lock_guard<std::mutex> lock(owner->QueueMutex());
    auto commands = NS::RetainPtr(owner->NativeQueue()->commandBuffer());
    auto blit = commands->blitCommandEncoder();
    blit->copyFromTexture(static_cast<MetalImage *>(image)->NativeImage(), 0, 0, MTL::Origin::Make(x, y, 0),
                          MTL::Size::Make(1, 1, 1), readback.get(), 0, 256, 256);
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

class RhiImage : public testing::TestWithParam<BackendAPI> {
 protected:
  void SetUp() override {
#if defined(LONGMARCH_D3D12_ENABLED)
    if (GetParam() == BACKEND_API_D3D12) {
      backend::d3d12::ComPtr<ID3D12Debug> debug;
      ASSERT_TRUE(SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))));
      debug->EnableDebugLayer();
    }
#endif
  }

  void Track(IDevice *device) {
#if defined(LONGMARCH_D3D12_ENABLED)
    if (GetParam() == BACKEND_API_D3D12)
      ASSERT_TRUE(SUCCEEDED(static_cast<backend::d3d12::D3D12Device *>(device)->NativeDevice()->QueryInterface(
          IID_PPV_ARGS(&messages_))));
#endif
  }

  void TearDown() override {
#if defined(LONGMARCH_D3D12_ENABLED)
    if (messages_)
      for (UINT64 i = 0; i < messages_->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
        SIZE_T size = 0;
        messages_->GetMessage(i, nullptr, &size);
        std::vector<uint8_t> storage(size);
        auto message = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
        messages_->GetMessage(i, message, &size);
        if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR)
          ADD_FAILURE() << message->pDescription;
      }
    messages_.Reset();
#endif
  }
#if defined(LONGMARCH_D3D12_ENABLED)
  backend::d3d12::ComPtr<ID3D12InfoQueue> messages_;
#endif
};

TEST_P(RhiImage, StorageUsageFollowsNativeFormatSupport) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  if (GetParam() == BACKEND_API_METAL)
    GTEST_SKIP() << "Metal binding groups are not implemented.";
  for (auto format :
       {IMAGE_FORMAT_RGBA8_UNORM, IMAGE_FORMAT_BGRA8_UNORM, IMAGE_FORMAT_D32_FLOAT, IMAGE_FORMAT_D24_UNORM_S8_UINT}) {
    ObjPtr<IImage> image = device->CreateImage({16, 16, format});
    ASSERT_TRUE(image);
    const auto native = static_cast<backend::Image *>(image.Get());
    bool supported = false;
#if defined(LONGMARCH_D3D12_ENABLED)
    if (GetParam() == BACKEND_API_D3D12) {
      auto owner = static_cast<backend::d3d12::D3D12Device *>(device.Get());
      supported = backend::d3d12::SupportsImageStorage(owner->NativeDevice(), format);
      EXPECT_FALSE(backend::d3d12::SupportsImageStorage(owner->NativeDevice(), IMAGE_FORMAT_UNKNOWN));
    }
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    if (GetParam() == BACKEND_API_VULKAN) {
      auto owner = static_cast<backend::vulkan::VulkanDevice *>(device.Get());
      supported = backend::vulkan::SupportsImageStorage(owner->NativePhysicalDevice(), format);
      EXPECT_FALSE(backend::vulkan::SupportsImageStorage(owner->NativePhysicalDevice(), IMAGE_FORMAT_UNKNOWN));
    }
#endif
    EXPECT_EQ(native->SupportsStorage(), supported);
    if (IsDepthFormat(format))
      EXPECT_FALSE(supported);
    BindingLayoutDesc layout{{{0, BINDING_RW_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}}};
    ObjPtr<IBindingGroup> group = device->CreateBindingGroup(layout);
    ASSERT_TRUE(group);
    EXPECT_EQ(group->BindResource(0, image.Get()), supported);
  }
}

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
  Track(device.Get());
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
      ASSERT_TRUE(commands->CmdBeginRendering({{{image.Get(), ATTACHMENT_LOAD_OP_LOAD, ATTACHMENT_STORE_OP_STORE}}}));
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

TEST_P(RhiImage, IndexedInstancedAttributesAndOffsetsProduceExpectedPixels) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IImage> image = device->CreateImage({64, 64, IMAGE_FORMAT_RGBA8_UNORM});
  ASSERT_TRUE(image);
  const std::string source = R"(
    struct Input { RHI_INPUT(3,float2,position); RHI_INPUT(5,float4,color); };
    struct Varying { float4 position : SV_Position; float4 color : COLOR0; };
    Varying VSMain(Input input) { Varying result; result.position=float4(input.position,0,1); result.color=input.color; return result; }
    float4 PSMain(Varying input) : SV_Target0 { return input.color; }
  )";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.vertex_bindings = {{8, false}, {16, true}};
  desc.vertex_attributes = {{3, 0, VERTEX_FORMAT_FLOAT2, 0}, {5, 1, VERTEX_FORMAT_FLOAT4, 0}};
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  ASSERT_TRUE(program->Finalize());
  // Skip prefixes with binding/index offsets, then use base vertex and base instance.
  const float vertices[] = {9, 9, 9, 9, 0, .75f, -.75f, -.75f, .75f, -.75f};
  const float colors[] = {0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1};
  const uint16_t indices[] = {99, 99, 0, 1, 2};
  ObjPtr<IBuffer> vertex = device->CreateBuffer({sizeof(vertices)});
  ObjPtr<IBuffer> color = device->CreateBuffer({sizeof(colors)});
  ObjPtr<IBuffer> index = device->CreateBuffer({sizeof(indices)});
  ASSERT_TRUE(vertex);
  ASSERT_TRUE(color);
  ASSERT_TRUE(index);
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands->CmdUploadBuffer(vertex.Get(), 0, vertices, sizeof(vertices)));
  ASSERT_TRUE(commands->CmdUploadBuffer(color.Get(), 0, colors, sizeof(colors)));
  ASSERT_TRUE(commands->CmdUploadBuffer(index.Get(), 0, indices, sizeof(indices)));
  ASSERT_TRUE(commands->CmdBindProgram(program.Get()));
  ASSERT_TRUE(commands->CmdBindVertexBuffer(0, vertex.Get(), 8));
  ASSERT_TRUE(commands->CmdBindVertexBuffer(1, color.Get(), 16));
  ASSERT_TRUE(commands->CmdBindIndexBuffer(index.Get(), INDEX_FORMAT_UINT16, 2));
  ASSERT_TRUE(commands->CmdSetViewport({0, 0, 64, 64, 0, 1}));
  ASSERT_TRUE(commands->CmdSetScissor({0, 0, 64, 64}));
  ASSERT_TRUE(commands->CmdBeginRendering(
      {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}));
  ASSERT_TRUE(commands->CmdDrawIndexed(3, 1, 1, 1, 1));
  ASSERT_TRUE(commands->CmdEndRendering());
  // A smaller attachment must clip an inherited scissor before any rebinding.
  ObjPtr<IImage> smaller = device->CreateImage({32, 32, IMAGE_FORMAT_RGBA8_UNORM});
  ASSERT_TRUE(smaller);
  ASSERT_TRUE(commands->CmdBeginRendering(
      {{{smaller.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 1, 0, 1}}}}));
  ASSERT_TRUE(commands->CmdEndRendering());
  ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
  ASSERT_TRUE(done);
  ASSERT_TRUE(done->WaitHost());
  std::array<uint8_t, 4> pixel{};
  ReadPixel(device.Get(), image.Get(), GetParam(), 32, 32, pixel);
  EXPECT_EQ(pixel, (std::array<uint8_t, 4>{255, 0, 0, 255}));
  ReadPixel(device.Get(), image.Get(), GetParam(), 0, 0, pixel);
  EXPECT_EQ(pixel, (std::array<uint8_t, 4>{0, 0, 0, 255}));
}

TEST_P(RhiImage, MultipleTargetsAndDepthRejectOccludedFragments) {
  if (GetParam() == BACKEND_API_METAL)
    GTEST_SKIP() << "Metal implementation is deferred.";
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IImage> first = device->CreateImage({64, 64, IMAGE_FORMAT_RGBA8_UNORM});
  ObjPtr<IImage> second = device->CreateImage({64, 64, IMAGE_FORMAT_RGBA8_UNORM});
  ObjPtr<IImage> depth = device->CreateImage({64, 64, IMAGE_FORMAT_D32_FLOAT});
  ASSERT_TRUE(first && second && depth);
  const std::string source = R"(
    struct V { float4 position : SV_Position; float nearValue : COLOR0; };
    V VSMain(uint id : SV_VertexID) {
      float2 p[3] = {float2(0,.75), float2(-.75,-.75),float2(.75,-.75)};
      V v; v.nearValue = id < 3 ? 1 : 0; v.position=float4(p[id%3],id < 3 ? .2 : .8,1); return v;
    }
    struct P { float4 a : SV_Target0; float4 b : SV_Target1; };
    P PSMain(V v) { P p; p.a=float4(v.nearValue,1-v.nearValue,0,1); p.b=float4(0,1-v.nearValue,v.nearValue,1); return p; }
  )";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.color_formats = {IMAGE_FORMAT_RGBA8_UNORM, IMAGE_FORMAT_RGBA8_UNORM};
  desc.depth_stencil.format = IMAGE_FORMAT_D32_FLOAT;
  desc.depth_stencil.depth_test = true;
  desc.depth_stencil.depth_write = true;
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  ASSERT_TRUE(program->Finalize());
  for (int repeat = 0; repeat < 3; ++repeat) {
    ObjPtr<ICommandContext> commands = device->CreateCommandContext();
    ASSERT_TRUE(commands->CmdBindProgram(program.Get()));
    RenderingInfo rendering;
    rendering.color_attachments = {{first.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}},
                                   {second.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}};
    rendering.depth_stencil.image = depth.Get();
    rendering.depth_stencil.depth_load_op = ATTACHMENT_LOAD_OP_CLEAR;
    ASSERT_TRUE(commands->CmdBeginRendering(rendering));
    ASSERT_TRUE(commands->CmdSetViewport({0, 0, 64, 64, 0, 1}));
    ASSERT_TRUE(commands->CmdSetScissor({0, 0, 64, 64}));
    ASSERT_TRUE(commands->CmdDraw(6));
    ASSERT_TRUE(commands->CmdEndRendering());
    ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
    ASSERT_TRUE(done);
    commands.Release();
    ASSERT_TRUE(done->WaitHost());
    std::array<uint8_t, 4> pixel{};
    ReadPixel(device.Get(), first.Get(), GetParam(), 32, 32, pixel);
    EXPECT_EQ(pixel, (std::array<uint8_t, 4>{255, 0, 0, 255}));
    ReadPixel(device.Get(), second.Get(), GetParam(), 32, 32, pixel);
    EXPECT_EQ(pixel, (std::array<uint8_t, 4>{0, 0, 255, 255}));
  }
}

TEST_P(RhiImage, BindingVersionsPreserveRangesAndArraysAcrossHeapPages) {
  if (GetParam() == BACKEND_API_METAL)
    GTEST_SKIP() << "Metal implementation is deferred.";
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IImage> target = device->CreateImage({64, 64, IMAGE_FORMAT_RGBA8_UNORM});
  ObjPtr<IImage> red = device->CreateImage({1, 1, IMAGE_FORMAT_RGBA8_UNORM});
  ObjPtr<IImage> green = device->CreateImage({1, 1, IMAGE_FORMAT_RGBA8_UNORM});
  ObjPtr<IBuffer> uniform = device->CreateBuffer({512});
  ObjPtr<IBuffer> factors = device->CreateBuffer({256});
  ASSERT_TRUE(target && red && green && uniform && factors);
  const std::string source = R"(
    struct Selection { uint textureIndex; };
    RHI_UNIFORM(0,3,Selection,selection);
    RHI_TEXTURE2D_ARRAY(1,7,float4,textures);
    RHI_SAMPLER(1,9,nearestSampler);
    RHI_STRUCTURED_BUFFER_ARRAY(1,11,float4,factors);
    float4 VSMain(uint id:SV_VertexID):SV_Position {
      float2 positions[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(positions[id],0,1);
    }
    float4 PSMain():SV_Target0 {return (selection.textureIndex==0?textures[0].Sample(nearestSampler,float2(.5,.5)):textures[1].Sample(nearestSampler,float2(.5,.5)))*factors[1][0];}
  )";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.pipeline_layout.groups = {{{{3, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT}}},
                                 {{{7, BINDING_TEXTURE2D, 2, SHADER_VISIBILITY_FRAGMENT},
                                   {9, BINDING_SAMPLER, 1, SHADER_VISIBILITY_FRAGMENT},
                                   {11, BINDING_STORAGE_BUFFER, 2, SHADER_VISIBILITY_FRAGMENT}}}};
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  ASSERT_TRUE(program->Finalize());
  ObjPtr<IBindingGroup> selection = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ObjPtr<IBindingGroup> material = device->CreateBindingGroup(desc.pipeline_layout.groups[1]);
  ASSERT_TRUE(selection);
  ASSERT_TRUE(material);
  ASSERT_TRUE(material->BindResource(7, std::vector<IImage *>{red.Get(), green.Get()}));
  ASSERT_TRUE(material->BindResource(9, SamplerDesc{}));
  ASSERT_TRUE(material->BindResource(11, std::vector<BufferView>{{factors.Get(), 0, 16}, {factors.Get(), 16, 16}}));
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  const uint32_t red_pixel = 0xff0000ff, green_pixel = 0xff00ff00;
  ASSERT_TRUE(commands->CmdUploadImage(red.Get(), &red_pixel, 4));
  ASSERT_TRUE(commands->CmdUploadImage(green.Get(), &green_pixel, 4));
  std::array<uint32_t, 128> constants{};
  constants[64] = 1;
  const float values[] = {0, 0, 0, 0, 1, 1, 1, 1};
  ASSERT_TRUE(commands->CmdUploadBuffer(uniform.Get(), 0, constants.data(), sizeof(constants)));
  ASSERT_TRUE(commands->CmdUploadBuffer(factors.Get(), 0, values, sizeof(values)));
  ASSERT_TRUE(commands->CmdBindProgram(program.Get()));
  ASSERT_TRUE(commands->CmdBindGroup(1, material.Get()));
  // Changes after binding must not affect this context, including render-pass restarts.
  ASSERT_TRUE(material->BindResource(7, std::vector<IImage *>{green.Get(), red.Get()}));
  ASSERT_TRUE(material->BindResource(9, SamplerDesc{SAMPLER_FILTER_NEAREST}));
  ASSERT_TRUE(material->BindResource(11, std::vector<BufferView>{{factors.Get(), 0, 16}, {factors.Get(), 0, 16}}));
  ASSERT_TRUE(commands->CmdBeginRendering(
      {{{target.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}));
  ASSERT_TRUE(commands->CmdSetViewport({0, 0, 64, 64, 0, 1}));
  // Exceeds both resource/sampler page capacities. Previous draw tables must stay alive.
  for (int i = 0; i < 1100; ++i) {
    ASSERT_TRUE(selection->BindResource(3, BufferView{uniform.Get(), uint64_t(i % 2 ? 256 : 0), 256}));
    ASSERT_TRUE(commands->CmdBindGroup(0, selection.Get()));
    // D3D12 builds tables at Draw, but must use the version captured at BindGroup.
    if (i != 1099)
      ASSERT_TRUE(selection->BindResource(3, BufferView{uniform.Get(), uint64_t(i % 2 ? 0 : 256), 256}));
    ASSERT_TRUE(commands->CmdSetScissor({i % 2 ? 32 : 0, 0, 32, 64}));
    ASSERT_TRUE(commands->CmdDraw(3));
  }
  ASSERT_TRUE(commands->CmdEndRendering());
  material.Release();
  factors.Release();
  red.Release();
  green.Release();
  program.Release();
  // Hold the native queue so rebinding is tested while the submission is truly
  // pending, rather than relying on timing on a fast GPU. The guard also opens
  // the gate on assertion failure, before Device teardown waits for the queue.
  std::shared_ptr<void> gate;
#if defined(LONGMARCH_D3D12_ENABLED)
  if (GetParam() == BACKEND_API_D3D12) {
    auto native = static_cast<backend::d3d12::D3D12Device *>(device.Get());
    backend::d3d12::ComPtr<ID3D12Fence> fence;
    ASSERT_TRUE(SUCCEEDED(native->NativeDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))));
    gate = std::shared_ptr<void>(native, [native, fence](void *) {
      EXPECT_TRUE(SUCCEEDED(fence->Signal(1)));
      std::lock_guard<std::mutex> lock(native->QueueMutex());
      EXPECT_TRUE(native->WaitIdle());
    });
    std::lock_guard<std::mutex> lock(native->QueueMutex());
    ASSERT_TRUE(SUCCEEDED(native->NativeQueue()->Wait(fence.Get(), 1)));
  }
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
  if (GetParam() == BACKEND_API_VULKAN) {
    auto native = static_cast<backend::vulkan::VulkanDevice *>(device.Get());
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    create.pNext = &type;
    VkSemaphore semaphore{};
    ASSERT_EQ(vkCreateSemaphore(native->NativeDevice(), &create, nullptr, &semaphore), VK_SUCCESS);
    gate = std::shared_ptr<void>(native, [native, semaphore](void *) {
      VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
      signal.semaphore = semaphore;
      signal.value = 1;
      EXPECT_EQ(vkSignalSemaphore(native->NativeDevice(), &signal), VK_SUCCESS);
      std::lock_guard<std::mutex> lock(native->QueueMutex());
      EXPECT_EQ(vkQueueWaitIdle(native->NativeQueue()), VK_SUCCESS);
      vkDestroySemaphore(native->NativeDevice(), semaphore, nullptr);
    });
    const uint64_t value = 1;
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timeline.waitSemaphoreValueCount = 1;
    timeline.pWaitSemaphoreValues = &value;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.pNext = &timeline;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &semaphore;
    submit.pWaitDstStageMask = &stage;
    std::lock_guard<std::mutex> lock(native->QueueMutex());
    ASSERT_EQ(vkQueueSubmit(native->NativeQueue(), 1, &submit, VK_NULL_HANDLE), VK_SUCCESS);
  }
#endif
  ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
  ASSERT_TRUE(done);
  EXPECT_EQ(done->GetStatus(), SUBMISSION_STATUS_PENDING);
  ASSERT_TRUE(selection->BindResource(3, BufferView{uniform.Get(), 0, 256}));
  selection.Release();
  uniform.Release();
  commands.Release();
  gate.reset();
  ASSERT_TRUE(done->WaitHost());
  std::array<uint8_t, 4> pixel{};
  ReadPixel(device.Get(), target.Get(), GetParam(), 16, 32, pixel);
  EXPECT_EQ(pixel, (std::array<uint8_t, 4>{255, 0, 0, 255}));
  ReadPixel(device.Get(), target.Get(), GetParam(), 48, 32, pixel);
  EXPECT_EQ(pixel, (std::array<uint8_t, 4>{0, 255, 0, 255}));
}

TEST_P(RhiImage, ReflectedStructuredStrideReadsSecondPaddedElement) {
  if (GetParam() == BACKEND_API_METAL)
    GTEST_SKIP() << "Metal bindings are deferred.";
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IImage> target = device->CreateImage({32, 32, IMAGE_FORMAT_RGBA8_UNORM});
  ObjPtr<IBuffer> buffer = device->CreateBuffer({256});
  ASSERT_TRUE(queue);
  ASSERT_TRUE(target);
  ASSERT_TRUE(buffer);
  const std::string source = R"(
    struct Element { float4 color; float scale; };
    RHI_STRUCTURED_BUFFER(0,0,Element,values);
    RHI_STRUCTURED_BUFFER(0,1,float4,unusedValues);
    float4 VSMain(uint id:SV_VertexID):SV_Position {
      float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[id],0,1);
    }
    float4 PSMain():SV_Target0 { return values[1].color*values[1].scale; }
  )";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.pipeline_layout.groups = {{{{0, BINDING_STORAGE_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT},
                                   {1, BINDING_STORAGE_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT}}}};
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  ASSERT_TRUE(program->Finalize());
  // The same Slang struct has a 20-byte DXIL stride and a padded 32-byte SPIR-V stride.
  const uint32_t stride = GetParam() == BACKEND_API_D3D12 ? 20 : 32;
  std::array<uint8_t, 256> data{};
  const float value[] = {1, 0, 0, 1, 1};
  std::memcpy(data.data() + stride, value, sizeof(value));
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ASSERT_TRUE(group);
  ASSERT_TRUE(group->BindResource(0, BufferView{buffer.Get(), 0, stride * 2}));
  // No stride is needed for an optimized-out binding, even for an arbitrary byte view.
  ASSERT_TRUE(group->BindResource(1, BufferView{buffer.Get(), 0, 3}));
  if (GetParam() == BACKEND_API_D3D12) {
    ObjPtr<IBindingGroup> invalid = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
    ASSERT_TRUE(invalid);
    ASSERT_TRUE(invalid->BindResource(0, BufferView{buffer.Get(), 4, stride * 2}));
    ASSERT_TRUE(invalid->BindResource(1, buffer.Get()));
    ObjPtr<ICommandContext> rejected = device->CreateCommandContext();
    ASSERT_TRUE(rejected);
    ASSERT_TRUE(rejected->CmdBindProgram(program.Get()));
    EXPECT_FALSE(rejected->CmdBindGroup(0, invalid.Get()));
  }
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands);
  ASSERT_TRUE(commands->CmdUploadBuffer(buffer.Get(), 0, data.data(), data.size()));
  ASSERT_TRUE(commands->CmdBindProgram(program.Get()));
  ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
  ASSERT_TRUE(commands->CmdBeginRendering(
      {{{target.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}));
  ASSERT_TRUE(commands->CmdSetViewport({0, 0, 32, 32, 0, 1}));
  ASSERT_TRUE(commands->CmdSetScissor({0, 0, 32, 32}));
  ASSERT_TRUE(commands->CmdDraw(3));
  ASSERT_TRUE(commands->CmdEndRendering());
  ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
  ASSERT_TRUE(done);
  ASSERT_TRUE(done->WaitHost());
  std::array<uint8_t, 4> pixel{};
  ReadPixel(device.Get(), target.Get(), GetParam(), 16, 16, pixel);
  EXPECT_EQ(pixel, (std::array<uint8_t, 4>{255, 0, 0, 255}));
}

TEST_P(RhiImage, StorageImageAndBufferBecomeReadableAfterRendering) {
  if (GetParam() == BACKEND_API_METAL)
    GTEST_SKIP() << "Metal implementation is deferred.";
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IImage> target = device->CreateImage({64, 64, IMAGE_FORMAT_RGBA8_UNORM});
  ObjPtr<IImage> storage = device->CreateImage({64, 64, IMAGE_FORMAT_RGBA8_UNORM});
  ObjPtr<IBuffer> buffer = device->CreateBuffer({64 * 64 * 16});
  ASSERT_TRUE(target && storage && buffer);
  const std::string source = R"(
    #ifdef WRITE_PASS
    RHI_RW_TEXTURE2D(0,0,float4,outputImage);
    RHI_RW_STRUCTURED_BUFFER(0,1,float4,outputBuffer);
    #else
    RHI_TEXTURE2D(0,0,float4,inputImage);
    RHI_STRUCTURED_BUFFER(0,1,float4,inputBuffer);
    #endif
    float4 VSMain(uint id:SV_VertexID):SV_Position {float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[id],0,1);}
    #ifdef WRITE_PASS
    float4 WritePS(float4 pos:SV_Position):SV_Target0 {uint2 xy=uint2(pos.xy);outputImage[xy]=float4(1,0,0,1);outputBuffer[xy.y*64+xy.x]=float4(0,0,1,0);return 0;}
    #else
    float4 ReadPS(float4 pos:SV_Position):SV_Target0 {uint2 xy=uint2(pos.xy);return inputImage.Load(int3(xy,0))+inputBuffer[xy.y*64+xy.x];}
    #endif
  )";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> write = device->CreateShader({"#define WRITE_PASS\n" + source, "WritePS", SHADER_STAGE_FRAGMENT});
  ObjPtr<IShader> read = device->CreateShader({source, "ReadPS", SHADER_STAGE_FRAGMENT});
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = write.Get();
  desc.pipeline_layout.groups = {{{{0, BINDING_RW_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT},
                                   {1, BINDING_RW_STORAGE_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT}}},
                                 {{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT},
                                   {1, BINDING_STORAGE_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT}}}};
  const auto write_layout = desc.pipeline_layout.groups[0], read_layout = desc.pipeline_layout.groups[1];
  desc.pipeline_layout.groups = {write_layout};
  // Each entry is compiled against its own authoritative layout.
  vs = device->CreateShader({"#define WRITE_PASS\n" + source, "VSMain", SHADER_STAGE_VERTEX});
  desc.vertex_shader = vs.Get();
  ObjPtr<IProgram> writer = device->CreateProgram(desc);
  vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = read.Get();
  desc.pipeline_layout.groups = {read_layout};
  ObjPtr<IProgram> reader = device->CreateProgram(desc);
  ASSERT_TRUE(writer && reader);
  ASSERT_TRUE(writer->Finalize());
  ASSERT_TRUE(reader->Finalize());
  ObjPtr<IBindingGroup> wg = device->CreateBindingGroup(write_layout);
  ObjPtr<IBindingGroup> rg = device->CreateBindingGroup(read_layout);
  ASSERT_TRUE(wg);
  ASSERT_TRUE(rg);
  for (auto group : {wg.Get(), rg.Get()}) {
    ASSERT_TRUE(group->BindResource(0, storage.Get()));
    ASSERT_TRUE(group->BindResource(1, BufferView{buffer.Get()}));
  }
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  for (auto program : {writer.Get(), reader.Get()}) {
    ASSERT_TRUE(commands->CmdBindProgram(program));
    ASSERT_TRUE(commands->CmdBindGroup(0, program == writer.Get() ? wg.Get() : rg.Get()));
    ASSERT_TRUE(commands->CmdBeginRendering(
        {{{target.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}));
    ASSERT_TRUE(commands->CmdSetViewport({0, 0, 64, 64, 0, 1}));
    ASSERT_TRUE(commands->CmdSetScissor({0, 0, 64, 64}));
    ASSERT_TRUE(commands->CmdDraw(3));
    ASSERT_TRUE(commands->CmdEndRendering());
  }
  ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
  ASSERT_TRUE(done);
  commands.Release();
  ASSERT_TRUE(done->WaitHost());
  std::array<uint8_t, 4> pixel{};
  ReadPixel(device.Get(), target.Get(), GetParam(), 32, 32, pixel);
  EXPECT_EQ(pixel, (std::array<uint8_t, 4>{255, 0, 255, 255}));
}

TEST_P(RhiImage, StencilSurvivesDepthOnlyPassAndMasksColorPass) {
  if (GetParam() == BACKEND_API_METAL)
    GTEST_SKIP() << "Metal implementation is deferred.";
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IImage> target = device->CreateImage({64, 64, IMAGE_FORMAT_RGBA8_UNORM});
  ObjPtr<IImage> depth = device->CreateImage({64, 64, IMAGE_FORMAT_D24_UNORM_S8_UINT});
  ASSERT_TRUE(target && depth);
  const std::string source = R"(
    float4 VSMain(uint id:SV_VertexID):SV_Position {float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[id],.5,1);}
    void MaskPS() {}
    float4 PSMain():SV_Target0 {return float4(0,1,0,1);}
  )";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> mask = device->CreateShader({source, "MaskPS", SHADER_STAGE_FRAGMENT});
  ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = mask.Get();
  desc.color_format = IMAGE_FORMAT_UNKNOWN;
  desc.depth_stencil.format = IMAGE_FORMAT_D24_UNORM_S8_UINT;
  desc.depth_stencil.stencil_test = true;
  desc.depth_stencil.stencil_reference = 5;
  desc.depth_stencil.front.pass_op = desc.depth_stencil.back.pass_op = STENCIL_REPLACE;
  ObjPtr<IProgram> writer = device->CreateProgram(desc);
  desc.color_format = IMAGE_FORMAT_RGBA8_UNORM;
  desc.fragment_shader = ps.Get();
  desc.depth_stencil.front.pass_op = desc.depth_stencil.back.pass_op = STENCIL_KEEP;
  desc.depth_stencil.front.compare = desc.depth_stencil.back.compare = COMPARE_EQUAL;
  ObjPtr<IProgram> reader = device->CreateProgram(desc);
  ASSERT_TRUE(writer && reader);
  ASSERT_TRUE(writer->Finalize());
  ASSERT_TRUE(reader->Finalize());
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  RenderingInfo rendering;
  rendering.depth_stencil.image = depth.Get();
  rendering.depth_stencil.depth_load_op = ATTACHMENT_LOAD_OP_CLEAR;
  rendering.depth_stencil.stencil_load_op = ATTACHMENT_LOAD_OP_CLEAR;
  ASSERT_TRUE(commands->CmdBindProgram(writer.Get()));
  ASSERT_TRUE(commands->CmdBeginRendering(rendering));
  ASSERT_TRUE(commands->CmdSetViewport({0, 0, 64, 64, 0, 1}));
  ASSERT_TRUE(commands->CmdSetScissor({0, 0, 32, 64}));
  ASSERT_TRUE(commands->CmdDraw(3));
  ASSERT_TRUE(commands->CmdEndRendering());
  rendering.color_attachments = {{target.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}};
  rendering.depth_stencil.depth_load_op = rendering.depth_stencil.stencil_load_op = ATTACHMENT_LOAD_OP_LOAD;
  ASSERT_TRUE(commands->CmdBindProgram(reader.Get()));
  ASSERT_TRUE(commands->CmdBeginRendering(rendering));
  ASSERT_TRUE(commands->CmdSetScissor({0, 0, 64, 64}));
  ASSERT_TRUE(commands->CmdDraw(3));
  ASSERT_TRUE(commands->CmdEndRendering());
  ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
  ASSERT_TRUE(done);
  commands.Release();
  ASSERT_TRUE(done->WaitHost());
  std::array<uint8_t, 4> pixel{};
  ReadPixel(device.Get(), target.Get(), GetParam(), 16, 32, pixel);
  EXPECT_EQ(pixel, (std::array<uint8_t, 4>{0, 255, 0, 255}));
  ReadPixel(device.Get(), target.Get(), GetParam(), 48, 32, pixel);
  EXPECT_EQ(pixel, (std::array<uint8_t, 4>{0, 0, 0, 255}));
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
INSTANTIATE_TEST_SUITE_P(Backends, RhiImage, testing::ValuesIn(backends));
}  // namespace
}  // namespace grassland::rhi

#endif
