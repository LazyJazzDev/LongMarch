#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <thread>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/rhi.h"
#if defined(LONGMARCH_METAL_ENABLED)
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_image.h"
#include "grassland/rhi/backend/metal/metal_util.h"
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
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
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
#if defined(LONGMARCH_METAL_ENABLED)
    if (GetParam() == BACKEND_API_METAL) {
      supported = backend::metal::SupportsImageStorage(format);
      EXPECT_FALSE(backend::metal::SupportsImageStorage(IMAGE_FORMAT_UNKNOWN));
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

#if defined(LONGMARCH_D3D12_ENABLED)
TEST(RhiD3D12Image, AttachmentLeasesFollowImagesRetainedByCommands) {
  ObjPtr<ICore> core = CreateCore(BACKEND_API_D3D12, {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto owner = static_cast<backend::d3d12::D3D12Device *>(device.Get());
  ObjPtr<IImage> color = device->CreateImage({8, 8, IMAGE_FORMAT_RGBA8_UNORM});
  ObjPtr<IImage> depth = device->CreateImage({8, 8, IMAGE_FORMAT_D32_FLOAT});
  ASSERT_TRUE(color && depth);
  auto handle = static_cast<backend::d3d12::D3D12Image *>(color.Get())->AttachmentView().ptr;
  EXPECT_NE(handle, 0u);
  EXPECT_EQ(owner->RTVPool().GetPageCounts().second, 1u);
  EXPECT_EQ(owner->DSVPool().GetPageCounts().second, 1u);
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands);
  ASSERT_TRUE(commands->CmdClearColor(color.Get(), {0, 0, 0, 1}));
  RenderingInfo rendering;
  rendering.color_attachments = {{color.Get(), ATTACHMENT_LOAD_OP_LOAD, ATTACHMENT_STORE_OP_STORE}};
  rendering.depth_stencil.image = depth.Get();
  rendering.depth_stencil.depth_load_op = ATTACHMENT_LOAD_OP_CLEAR;
  ASSERT_TRUE(commands->CmdBeginRendering(rendering));
  ASSERT_TRUE(commands->CmdEndRendering());
  EXPECT_EQ(static_cast<backend::d3d12::D3D12Image *>(color.Get())->AttachmentView().ptr, handle);
  color.Release();
  depth.Release();
  EXPECT_EQ(owner->RTVPool().GetPageCounts().second, 1u);
  EXPECT_EQ(owner->DSVPool().GetPageCounts().second, 1u);
  commands.Release();
  EXPECT_EQ(owner->RTVPool().GetPageCounts().second, 0u);
  EXPECT_EQ(owner->DSVPool().GetPageCounts().second, 0u);
}
#endif

TEST_P(RhiImage, UploadRowsPreservePixelsWithDifferentSourceAndDestinationPitch) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(queue);
  ObjPtr<IImage> image = device->CreateImage({3, 3, IMAGE_FORMAT_RGBA8_UNORM});
  ASSERT_TRUE(image);
  // Twelve useful bytes per row, with eight bytes of caller-owned padding.
  // D3D12 uses a different, 256-byte destination pitch for this image.
  std::array<uint8_t, 60> source;
  source.fill(0xcd);
  for (uint32_t y = 0; y < 3; ++y)
    for (uint32_t x = 0; x < 3; ++x) {
      const std::array<uint8_t, 4> pixel{uint8_t(x * 70), uint8_t(y * 80), 37, 255};
      std::memcpy(source.data() + y * 20 + x * 4, pixel.data(), pixel.size());
    }
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands);
  ASSERT_TRUE(commands->CmdUploadImage(image.Get(), source.data(), 20));
  source.fill(0);  // Recording must already own the uploaded bytes.
  ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
  ASSERT_TRUE(done);
  commands.Release();
  ASSERT_TRUE(done->WaitHost());
  for (uint32_t y = 0; y < 3; ++y)
    for (uint32_t x = 0; x < 3; ++x) {
      std::array<uint8_t, 4> pixel{};
      ReadPixel(device.Get(), image.Get(), GetParam(), x, y, pixel);
      EXPECT_EQ(pixel, (std::array<uint8_t, 4>{uint8_t(x * 70), uint8_t(y * 80), 37, 255}));
    }
}

TEST_P(RhiImage, OffscreenClearDrawReadbackAndReuse) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
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
    desc.color_formats = {format};
    ObjPtr<IProgram> program = device->CreateProgram(desc);
    ASSERT_TRUE(program);
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

TEST_P(RhiImage, FragmentRayQueriesRetainIndirectBlasResources) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  DeviceFeatureRequirement required;
  required.ray_query = true;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection(required);
  if (!device)
    GTEST_SKIP() << "Ray query unavailable";
#if defined(LONGMARCH_METAL_ENABLED)
  if (GetParam() == BACKEND_API_METAL &&
      !static_cast<backend::metal::MetalDevice *>(device.Get())->NativeDevice()->supportsRaytracingFromRender())
    GTEST_SKIP() << "Render-stage ray query unavailable";
#endif
  Track(device.Get());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IImage> image = device->CreateImage({16, 16, IMAGE_FORMAT_RGBA8_UNORM});
  ASSERT_TRUE(queue && image);
  BottomLevelAccelerationStructureDesc bottom;
  bottom.geometries = {{3}};
  ObjPtr<IBottomLevelAccelerationStructure> blas = device->CreateBottomLevelAccelerationStructure(bottom);
  ObjPtr<ITopLevelAccelerationStructure> tlas = device->CreateTopLevelAccelerationStructure({1});
  const float positions[] = {-1, -1, 0, 0, 1, 0, 1, -1, 0};
  ObjPtr<IBuffer> vertices = device->CreateBuffer({sizeof(positions)});
  ASSERT_TRUE(blas && tlas && vertices);
  ObjPtr<ICommandContext> build = device->CreateCommandContext();
  ASSERT_TRUE(build);
  ASSERT_TRUE(build->CmdUploadBuffer(vertices.Get(), 0, positions, sizeof(positions)));
  ASSERT_TRUE(build->CmdBuildBottomLevelAccelerationStructure(blas.Get(), {{{vertices.Get()}, {}}}));
  RayTracingInstance instance;
  instance.blas = blas.Get();
  instance.disable_triangle_culling = true;
  ASSERT_TRUE(build->CmdBuildTopLevelAccelerationStructure(tlas.Get(), {instance}));
  ObjPtr<ISubmissionToken> built = queue->Submit(build.Get());
  ASSERT_TRUE(built);
  build.Release();
  ASSERT_TRUE(built->WaitHost());
  blas.Release();
  vertices.Release();
  const char *source = R"(
RHI_ACCELERATION_STRUCTURE(0,0,scene);
float4 VSMain(uint id:SV_VertexID):SV_Position {
 float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
 return float4(p[id],0,1);
}
float4 PSMain():SV_Target0 {
 RayDesc ray; ray.Origin=float3(0,0,-2); ray.Direction=float3(0,0,1); ray.TMin=0; ray.TMax=10;
 RayQuery<RAY_FLAG_NONE> query; query.TraceRayInline(scene,RAY_FLAG_NONE,255,ray);
 while(query.Proceed()) {}
 return query.CommittedStatus()==COMMITTED_TRIANGLE_HIT ? float4(0,1,0,1) : float4(1,0,0,1);
}
)";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
  ASSERT_TRUE(vs && ps);
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.pipeline_layout.groups = {{{{0, BINDING_ACCELERATION_STRUCTURE, 1, SHADER_VISIBILITY_FRAGMENT}}}};
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ASSERT_TRUE(group);
  ASSERT_TRUE(group->BindResource(0, tlas.Get()));
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands);
  ASSERT_TRUE(commands->CmdBindProgram(program.Get()));
  ASSERT_TRUE(commands->CmdBeginRendering(
      {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}));
  ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
  ASSERT_TRUE(commands->CmdSetViewport({0, 0, 16, 16, 0, 1}));
  ASSERT_TRUE(commands->CmdSetScissor({0, 0, 16, 16}));
  ASSERT_TRUE(commands->CmdDraw(3));
  ASSERT_TRUE(commands->CmdEndRendering());
  group.Release();
  tlas.Release();
  ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
  ASSERT_TRUE(done);
  commands.Release();
  ASSERT_TRUE(done->WaitHost());
  std::array<uint8_t, 4> pixel{};
  ReadPixel(device.Get(), image.Get(), GetParam(), 8, 8, pixel);
  EXPECT_EQ(pixel, (std::array<uint8_t, 4>{0, 255, 0, 255}));
}

TEST_P(RhiImage, EightBindingGroupsDoNotOverlapVertexBuffers) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IImage> image = device->CreateImage({16, 16, IMAGE_FORMAT_RGBA8_UNORM});
  ASSERT_TRUE(queue && image);
  ProgramDesc desc;
  std::string source;
  std::string sum;
  for (uint32_t i = 0; i < 8; ++i) {
    source += "RHI_STRUCTURED_BUFFER(" + std::to_string(i) + ",0,float4,color" + std::to_string(i) + ");\n";
    sum += (i ? "+" : "") + std::string("color") + std::to_string(i) + "[0]";
    desc.pipeline_layout.groups.push_back({{{0, BINDING_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_ALL, 16}}});
  }
  source += "struct Input { RHI_INPUT(0,float2,position); };\n";
  source += "float4 VSMain(Input input):SV_Position { return float4(input.position,0,1); }\n";
  source += "float4 PSMain():SV_Target0 { return " + sum + "; }\n";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
  ASSERT_TRUE(vs && ps);
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.vertex_bindings = {{8}};
  desc.vertex_attributes = {{0, 0, VERTEX_FORMAT_FLOAT2, 0}};
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  const float vertices[] = {-1, -1, 3, -1, -1, 3};
  ObjPtr<IBuffer> vertex = device->CreateBuffer({sizeof(vertices)});
  ASSERT_TRUE(vertex);
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands);
  ASSERT_TRUE(commands->CmdUploadBuffer(vertex.Get(), 0, vertices, sizeof(vertices)));
  ASSERT_TRUE(commands->CmdBindProgram(program.Get()));
  ASSERT_TRUE(commands->CmdBindVertexBuffer(0, vertex.Get()));
  for (uint32_t i = 0; i < 8; ++i) {
    ObjPtr<IBuffer> color = device->CreateBuffer({16});
    ObjPtr<IBindingGroup> group = device->CreateBindingGroup(desc.pipeline_layout.groups[i]);
    ASSERT_TRUE(color && group);
    const float value[] = {0.125f, 0, 0.125f, 0.125f};
    ASSERT_TRUE(commands->CmdUploadBuffer(color.Get(), 0, value, sizeof(value)));
    ASSERT_TRUE(group->BindResource(0, color.Get()));
    ASSERT_TRUE(commands->CmdBindGroup(i, group.Get()));
  }
  ASSERT_TRUE(commands->CmdBeginRendering(
      {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}));
  ASSERT_TRUE(commands->CmdSetViewport({0, 0, 16, 16, 0, 1}));
  ASSERT_TRUE(commands->CmdSetScissor({0, 0, 16, 16}));
  ASSERT_TRUE(commands->CmdDraw(3));
  ASSERT_TRUE(commands->CmdEndRendering());
  ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
  ASSERT_TRUE(done);
  commands.Release();
  ASSERT_TRUE(done->WaitHost());
  std::array<uint8_t, 4> pixel{};
  ReadPixel(device.Get(), image.Get(), GetParam(), 8, 8, pixel);
  EXPECT_EQ(pixel, (std::array<uint8_t, 4>{255, 0, 255, 255}));
}

TEST_P(RhiImage, IndexedInstancedAttributesAndOffsetsProduceExpectedPixels) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
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
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IImage> first = device->CreateImage({64, 64, IMAGE_FORMAT_RGBA8_UNORM});
  // Keep an intervening view alive so MRT cannot assume contiguous handles.
  ObjPtr<IImage> spacer = device->CreateImage({64, 64, IMAGE_FORMAT_RGBA8_UNORM});
  ASSERT_TRUE(spacer);
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
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
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
    RHI_SAMPLER(2,1,changingSampler);
    RHI_STRUCTURED_BUFFER_ARRAY(1,11,float4,factors);
    float4 VSMain(uint id:SV_VertexID):SV_Position {
      float2 positions[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(positions[id],0,1);
    }
    float4 PSMain():SV_Target0 {return (selection.textureIndex==0?textures[0].Sample(nearestSampler,float2(.5,.5)):textures[1].Sample(changingSampler,float2(.5,.5)))*factors[1][0];}
  )";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.pipeline_layout.groups = {{{{3, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT}}},
                                 {{{7, BINDING_TEXTURE2D, 2, SHADER_VISIBILITY_FRAGMENT},
                                   {9, BINDING_SAMPLER, 1, SHADER_VISIBILITY_FRAGMENT},
                                   {11, BINDING_STRUCTURED_BUFFER, 2, SHADER_VISIBILITY_FRAGMENT, 16}}},
                                 {{{1, BINDING_SAMPLER, 1, SHADER_VISIBILITY_FRAGMENT}}}};
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  ObjPtr<IBindingGroup> selection = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ObjPtr<IBindingGroup> material = device->CreateBindingGroup(desc.pipeline_layout.groups[1]);
  ObjPtr<IBindingGroup> sampling = device->CreateBindingGroup(desc.pipeline_layout.groups[2]);
  ASSERT_TRUE(sampling);
  ASSERT_TRUE(selection);
  ASSERT_TRUE(material);
  ASSERT_TRUE(material->BindResource(7, std::vector<IImage *>{red.Get(), green.Get()}));
  ASSERT_TRUE(material->BindResource(9, ObjPtr<ISampler>(device->CreateSampler(SamplerDesc{})).Get()));
  ASSERT_TRUE(material->BindResource(11, std::vector<BufferView>{{factors.Get(), 0, 16}, {factors.Get(), 16, 16}}));
  // Fork only the buffer binding, then destroy the source version. Copied image
  // and sampler descriptors must keep their native handles valid for GPU use.
  auto original = backend::ObjectPtr<backend::BindingGroup::Version>::Retain(
      static_cast<backend::BindingGroup *>(material.Get())->CurrentVersion());
  ASSERT_TRUE(material->BindResource(11, std::vector<BufferView>{{factors.Get(), 0, 16}}, 0));
  EXPECT_NE(static_cast<backend::BindingGroup *>(material.Get())->CurrentVersion(), original.Get());
  original.Reset();
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
  ASSERT_TRUE(
      material->BindResource(9, ObjPtr<ISampler>(device->CreateSampler(SamplerDesc{SAMPLER_FILTER_NEAREST})).Get()));
  ASSERT_TRUE(material->BindResource(11, std::vector<BufferView>{{factors.Get(), 0, 16}, {factors.Get(), 0, 16}}));
  ASSERT_TRUE(commands->CmdBeginRendering(
      {{{target.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}));
  ASSERT_TRUE(commands->CmdSetViewport({0, 0, 64, 64, 0, 1}));
  // Resource-only and sampler-only groups change independently. The material
  // stays bound across both page rollovers and must be migrated from its old version.
  const int draws = GetParam() == BACKEND_API_D3D12 ? 4200 : 1100;
  for (int i = 0; i < draws; ++i) {
    if (i < 300) {
      SamplerDesc sampler;
      sampler.filter = i % 2 ? SAMPLER_FILTER_LINEAR : SAMPLER_FILTER_NEAREST;
      ASSERT_TRUE(sampling->BindResource(1, ObjPtr<ISampler>(device->CreateSampler(sampler)).Get()));
      ASSERT_TRUE(commands->CmdBindGroup(2, sampling.Get()));
    }
    ASSERT_TRUE(selection->BindResource(3, BufferView{uniform.Get(), uint64_t(i % 2 ? 256 : 0), 256}));
    ASSERT_TRUE(commands->CmdBindGroup(0, selection.Get()));
    // D3D12 builds tables at Draw, but must use the version captured at BindGroup.
    if (i != draws - 1)
      ASSERT_TRUE(selection->BindResource(3, BufferView{uniform.Get(), uint64_t(i % 2 ? 0 : 256), 256}));
    ASSERT_TRUE(commands->CmdSetScissor({i % 2 ? 32 : 0, 0, 32, 64}));
    ASSERT_TRUE(commands->CmdDraw(3));
  }
#if defined(LONGMARCH_D3D12_ENABLED)
  if (GetParam() == BACKEND_API_D3D12) {
    auto native = static_cast<backend::d3d12::D3D12Device *>(device.Get());
    // Grow independently from 32: resources reach 4096, samplers reach 256.
    // Earlier recorded draws pin every smaller page until command retirement.
    EXPECT_EQ(native->ResourceDescriptors().GetPageCounts().second, 8u);
    EXPECT_EQ(native->SamplerDescriptors().GetPageCounts().second, 4u);
    for (uint32_t size = 32; size <= 4096; size *= 2)
      EXPECT_EQ(native->ResourceDescriptors().GetPageCounts(size).second, 1u);
    for (uint32_t size = 32; size <= 256; size *= 2)
      EXPECT_EQ(native->SamplerDescriptors().GetPageCounts(size).second, 1u);
  }
#endif
  ASSERT_TRUE(commands->CmdEndRendering());
  sampling.Release();
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
#if defined(LONGMARCH_METAL_ENABLED)
  if (GetParam() == BACKEND_API_METAL) {
    auto native = static_cast<backend::metal::MetalDevice *>(device.Get());
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto event = NS::TransferPtr(native->NativeDevice()->newSharedEvent());
    ASSERT_TRUE(event);
    gate = std::shared_ptr<void>(native, [native, event](void *) {
      event->setSignaledValue(1);
      std::lock_guard<std::mutex> lock(native->QueueMutex());
      EXPECT_TRUE(native->WaitIdle());
    });
    std::lock_guard<std::mutex> lock(native->QueueMutex());
    auto blocker = native->NativeQueue()->commandBuffer();
    blocker->encodeWait(event.get(), 1);
    blocker->commit();
  }
#endif
  ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
  ASSERT_TRUE(done);
  EXPECT_EQ(done->GetStatus(), SUBMISSION_STATUS_PENDING);
  ASSERT_TRUE(selection->BindResource(3, BufferView{uniform.Get(), 0, 256}));
  selection.Release();
  uniform.Release();
  commands.Release();
#if defined(LONGMARCH_D3D12_ENABLED)
  if (GetParam() == BACKEND_API_D3D12) {
    auto native = static_cast<backend::d3d12::D3D12Device *>(device.Get());
    EXPECT_EQ(native->ResourceDescriptors().GetPageCounts().second, 8u);
    EXPECT_EQ(native->SamplerDescriptors().GetPageCounts().second, 4u);
  }
#endif
  gate.reset();
  ASSERT_TRUE(done->WaitHost());
#if defined(LONGMARCH_D3D12_ENABLED)
  if (GetParam() == BACKEND_API_D3D12) {
    auto native = static_cast<backend::d3d12::D3D12Device *>(device.Get());
    EXPECT_EQ(native->ResourceDescriptors().GetPageCounts().second, 0u);
    EXPECT_EQ(native->SamplerDescriptors().GetPageCounts().second, 0u);
  }
#endif
  std::array<uint8_t, 4> pixel{};
  ReadPixel(device.Get(), target.Get(), GetParam(), 16, 32, pixel);
  EXPECT_EQ(pixel, (std::array<uint8_t, 4>{255, 0, 0, 255}));
  ReadPixel(device.Get(), target.Get(), GetParam(), 48, 32, pixel);
  EXPECT_EQ(pixel, (std::array<uint8_t, 4>{0, 255, 0, 255}));
}

TEST_P(RhiImage, ExplicitStructuredStrideReadsSecondCompactElement) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
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
  desc.pipeline_layout.groups = {{{{0, BINDING_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT, 20},
                                   {1, BINDING_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT, 16}}}};
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ASSERT_TRUE(program);

  // Both backends use the same compact C++ element layout.
  struct Element {
    float color[4];
    float scale;
  };

  static_assert(sizeof(Element) == 20);
  const uint32_t stride = sizeof(Element);
  std::array<uint8_t, 256> data{};
  const float value[] = {1, 0, 0, 1, 1};
  std::memcpy(data.data() + stride, value, sizeof(value));
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ASSERT_TRUE(group);
  ASSERT_TRUE(group->BindResource(0, BufferView{buffer.Get(), 0, stride * 2}));
  EXPECT_FALSE(group->BindResource(1, BufferView{buffer.Get(), 0, 3}));
  ASSERT_TRUE(group->BindResource(1, buffer.Get()));
  EXPECT_FALSE(group->BindResource(0, BufferView{buffer.Get(), 4, stride * 2}));
  EXPECT_FALSE(group->BindResource(0, BufferView{buffer.Get(), 0, stride * 2 - 1}));
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

TEST_P(RhiImage, StructuredNestedArraysUseSameCppLayout) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
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
    struct Inner { float3 color; float weight; };
    struct Element { float scale; Inner samples[2]; float3 tail; };
    RHI_STRUCTURED_BUFFER(0,0,Element,values);
    RHI_STRUCTURED_BUFFER(0,1,float4,unusedValues);
    float4 VSMain(uint id:SV_VertexID):SV_Position {
      float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[id],0,1);
    }
    float4 PSMain():SV_Target0 { return float4(values[1].samples[1].color * values[1].scale * values[1].samples[1].weight, values[1].tail.z); }
  )";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.pipeline_layout.groups = {{{{0, BINDING_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT, 48},
                                   {1, BINDING_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT, 16}}}};
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ASSERT_TRUE(program);

  // Both backends use the same compact C++ element layout.
  struct Inner {
    float color[3];
    float weight;
  };

  struct Element {
    float scale;
    Inner samples[2];
    float tail[3];
  };

  static_assert(sizeof(Element) == 48);
  const uint32_t stride = sizeof(Element);
  std::array<Element, 2> data{};
  data[1].scale = 1;
  data[1].samples[1] = {{1, 0, 0}, 1};
  data[1].tail[2] = 1;
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ASSERT_TRUE(group);
  ASSERT_TRUE(group->BindResource(0, BufferView{buffer.Get(), 0, stride * 2}));
  EXPECT_FALSE(group->BindResource(1, BufferView{buffer.Get(), 0, 3}));
  ASSERT_TRUE(group->BindResource(1, buffer.Get()));
  EXPECT_FALSE(group->BindResource(0, BufferView{buffer.Get(), 4, stride * 2}));
  EXPECT_FALSE(group->BindResource(0, BufferView{buffer.Get(), 0, stride * 2 - 1}));
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands);
  ASSERT_TRUE(commands->CmdUploadBuffer(buffer.Get(), 0, data.data(), sizeof(data)));
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
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
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
                                   {1, BINDING_RW_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT, 16}}},
                                 {{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT},
                                   {1, BINDING_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT, 16}}}};
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

TEST_P(RhiImage, RawBufferWritesBecomeReadableAfterRendering) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
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
    RHI_RW_BYTE_ADDRESS_BUFFER(0,1,outputBuffer);
    #else
    RHI_TEXTURE2D(0,0,float4,inputImage);
    RHI_BYTE_ADDRESS_BUFFER(0,1,inputBuffer);
    #endif
    float4 VSMain(uint id:SV_VertexID):SV_Position {float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[id],0,1);}
    #ifdef WRITE_PASS
    float4 WritePS(float4 pos:SV_Position):SV_Target0 {uint2 xy=uint2(pos.xy);outputImage[xy]=float4(1,0,0,1);outputBuffer.Store<float4>((xy.y*64+xy.x)*16,float4(0,0,1,0));return 0;}
    #else
    float4 ReadPS(float4 pos:SV_Position):SV_Target0 {uint2 xy=uint2(pos.xy);return inputImage.Load(int3(xy,0))+inputBuffer.Load<float4>((xy.y*64+xy.x)*16);}
    #endif
  )";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> write = device->CreateShader({"#define WRITE_PASS\n" + source, "WritePS", SHADER_STAGE_FRAGMENT});
  ObjPtr<IShader> read = device->CreateShader({source, "ReadPS", SHADER_STAGE_FRAGMENT});
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = write.Get();
  desc.pipeline_layout.groups = {{{{0, BINDING_RW_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT},
                                   {1, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT}}},
                                 {{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT},
                                   {1, BINDING_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT}}}};
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

TEST_P(RhiImage, SamplerArraysPreserveFilteringAcrossCapturedVersions) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IImage> target = device->CreateImage({8, 8, IMAGE_FORMAT_RGBA8_UNORM});
  ObjPtr<IImage> texture = device->CreateImage({2, 1, IMAGE_FORMAT_RGBA8_UNORM});
  ObjPtr<ISampler> nearest = device->CreateSampler({SAMPLER_FILTER_NEAREST});
  ObjPtr<ISampler> linear = device->CreateSampler({SAMPLER_FILTER_LINEAR});
  ASSERT_TRUE(queue && target && texture && nearest && linear);
  const std::string source = R"(
    RHI_TEXTURE2D(0,0,float4,image);
    RHI_SAMPLER_ARRAY(0,3,samplers);
    float4 VSMain(uint id:SV_VertexID):SV_Position {
      float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
      return float4(p[id],0,1);
    }
    float4 PSMain():SV_Target0 {
      return image.Sample(samplers[1],float2(.375,.5));
    }
  )";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.pipeline_layout.groups = {
      {{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}, {3, BINDING_SAMPLER, 2, SHADER_VISIBILITY_FRAGMENT}}}};
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ASSERT_TRUE(group);
  ASSERT_TRUE(group->BindResource(0, texture.Get()));
  ASSERT_TRUE(group->BindResource(3, std::vector<ISampler *>{linear.Get(), nearest.Get()}));
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands);
  const std::array<uint8_t, 8> pixels{255, 0, 0, 255, 0, 0, 255, 255};
  ASSERT_TRUE(commands->CmdUploadImage(texture.Get(), pixels.data(), pixels.size()));
  ASSERT_TRUE(commands->CmdBindProgram(program.Get()));
  ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
  // The first draw must still use nearest; the second captures the partial update.
  ASSERT_TRUE(group->BindResource(3, std::vector<ISampler *>{linear.Get()}, 1));
  nearest.Release();
  linear.Release();
  ASSERT_TRUE(commands->CmdBeginRendering(
      {{{target.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}));
  ASSERT_TRUE(commands->CmdSetViewport({0, 0, 8, 8, 0, 1}));
  ASSERT_TRUE(commands->CmdSetScissor({0, 0, 4, 8}));
  ASSERT_TRUE(commands->CmdDraw(3));
  ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
  ASSERT_TRUE(commands->CmdSetScissor({4, 0, 4, 8}));
  ASSERT_TRUE(commands->CmdDraw(3));
  ASSERT_TRUE(commands->CmdEndRendering());
  group.Release();
  texture.Release();
  ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
  ASSERT_TRUE(done);
  commands.Release();
  ASSERT_TRUE(done->WaitHost());
  std::array<uint8_t, 4> pixel{};
  ReadPixel(device.Get(), target.Get(), GetParam(), 2, 4, pixel);
  EXPECT_EQ(pixel, (std::array<uint8_t, 4>{255, 0, 0, 255}));
  ReadPixel(device.Get(), target.Get(), GetParam(), 6, 4, pixel);
  EXPECT_NEAR(pixel[0], 191, 1);
  EXPECT_EQ(pixel[1], 0);
  EXPECT_NEAR(pixel[2], 64, 1);
  EXPECT_EQ(pixel[3], 255);
}

TEST_P(RhiImage, BindingArraysExposeCapturedBufferViewLengths) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IImage> target = device->CreateImage({8, 8, IMAGE_FORMAT_RGBA8_UNORM});
  ObjPtr<IBuffer> buffer = device->CreateBuffer({1024});
  ASSERT_TRUE(buffer && target && queue);
  const std::string source = R"(
    struct Color {float4 value;};
    RHI_UNIFORM_ARRAY(0,7,Color,colors);
    RHI_STRUCTURED_BUFFER_ARRAY(0,2,float4,structured);
    RHI_BYTE_ADDRESS_BUFFER_ARRAY(0,9,raw);
    float4 VSMain(uint id:SV_VertexID):SV_Position {
      float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
      return float4(p[id],colors[0].value.x,1);
    }
    float4 PSMain():SV_Target0 {
      uint n0,s0,n1,s1,b0,b1;
      structured[0].GetDimensions(n0,s0); structured[1].GetDimensions(n1,s1);
      raw[0].GetDimensions(b0);raw[1].GetDimensions(b1);
      return n0==1 && n1==2 && s0==16 && s1==16 && b0==4 && b1==8 ? colors[1].value : float4(0,1,0,1);
    }
  )";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.pipeline_layout.groups = {{{{7, BINDING_UNIFORM_BUFFER, 2},
                                   {2, BINDING_STRUCTURED_BUFFER, 2, SHADER_VISIBILITY_FRAGMENT, 16},
                                   {9, BINDING_BYTE_ADDRESS_BUFFER, 2, SHADER_VISIBILITY_FRAGMENT}}}};
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ASSERT_TRUE(program);
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ASSERT_TRUE(group);
  ASSERT_TRUE(group->BindResource(7, std::vector<BufferView>{{buffer.Get(), 0, 256}, {buffer.Get(), 256, 256}}));
  ASSERT_TRUE(group->BindResource(2, std::vector<BufferView>{{buffer.Get(), 512, 16}, {buffer.Get(), 528, 32}}));
  ASSERT_TRUE(group->BindResource(9, std::vector<BufferView>{{buffer.Get(), 768, 4}, {buffer.Get(), 784, 8}}));
  std::array<float, 256> data{};
  data[64] = data[67] = 1;
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands);
  ASSERT_TRUE(commands->CmdUploadBuffer(buffer.Get(), 0, data.data(), sizeof(data)));
  ASSERT_TRUE(commands->CmdBindProgram(program.Get()));
  ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
  // Both the size table and uniform array must remain those captured at BindGroup.
  ASSERT_TRUE(group->BindResource(2, std::vector<BufferView>{{buffer.Get(), 512, 32}, {buffer.Get(), 528, 16}}));
  ASSERT_TRUE(group->BindResource(9, std::vector<BufferView>{{buffer.Get(), 768, 8}, {buffer.Get(), 784, 4}}));
  ASSERT_TRUE(group->BindResource(7, std::vector<BufferView>{{buffer.Get(), 256, 256}, {buffer.Get(), 0, 256}}));
  group.Release();
  ASSERT_TRUE(commands->CmdBeginRendering(
      {{{target.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}));
  ASSERT_TRUE(commands->CmdSetViewport({0, 0, 8, 8, 0, 1}));
  ASSERT_TRUE(commands->CmdSetScissor({0, 0, 8, 8}));
  ASSERT_TRUE(commands->CmdDraw(3));
  ASSERT_TRUE(commands->CmdEndRendering());
  ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
  ASSERT_TRUE(done);
  ASSERT_TRUE(done->WaitHost());
  std::array<uint8_t, 4> pixel{};
  ReadPixel(device.Get(), target.Get(), GetParam(), 4, 4, pixel);
  EXPECT_EQ(pixel, (std::array<uint8_t, 4>{255, 0, 0, 255}));
}

TEST_P(RhiImage, StencilSurvivesDepthOnlyPassAndMasksColorPass) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
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
  desc.color_formats.clear();
  desc.depth_stencil.format = IMAGE_FORMAT_D24_UNORM_S8_UINT;
  desc.depth_stencil.stencil_test = true;
  desc.depth_stencil.stencil_reference = 5;
  desc.depth_stencil.front.pass_op = desc.depth_stencil.back.pass_op = STENCIL_REPLACE;
  ObjPtr<IProgram> writer = device->CreateProgram(desc);
  desc.color_formats = {IMAGE_FORMAT_RGBA8_UNORM};
  desc.fragment_shader = ps.Get();
  desc.depth_stencil.front.pass_op = desc.depth_stencil.back.pass_op = STENCIL_KEEP;
  desc.depth_stencil.front.compare = desc.depth_stencil.back.compare = COMPARE_EQUAL;
  ObjPtr<IProgram> reader = device->CreateProgram(desc);
  ASSERT_TRUE(writer && reader);
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
TEST_P(RhiImage, IndependentColorAndAlphaBlendingMatchesPixelResults) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(queue);
  std::array<ObjPtr<IImage>, 3> targets;
  for (auto &target : targets) {
    target = device->CreateImage({8, 8, IMAGE_FORMAT_RGBA8_UNORM});
    ASSERT_TRUE(target);
  }
  const std::string source = R"(
    float4 VSMain(uint id:SV_VertexID):SV_Position {
      float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
      return float4(p[id],0,1);
    }
    struct Output { float4 a:SV_Target0; float4 b:SV_Target1; float4 c:SV_Target2; };
    Output PSMain() { Output o; o.a=o.b=o.c=float4(.25,.5,.75,.5); return o; }
  )";
  ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});

  struct Case {
    BlendState state;
    std::array<float, 4> expected;
  };

  // Literal results distinguish independent RGB/alpha factors, operation order,
  // MIN/MAX ignoring factors, and disabled blending on adjacent MRT attachments.
  std::vector<Case> cases{
      {BlendState{}, {.25f, .5f, .75f, .5f}},
      {BlendState(true), {.375f, .375f, .4375f, .625f}},
      {{BLEND_FACTOR_ONE, BLEND_FACTOR_ONE, BLEND_OP_ADD, BLEND_FACTOR_ONE, BLEND_FACTOR_ONE, BLEND_OP_ADD},
       {.75f, .75f, .875f, .75f}},
      {{BLEND_FACTOR_ONE, BLEND_FACTOR_ONE, BLEND_OP_SUBTRACT, BLEND_FACTOR_ONE, BLEND_FACTOR_ONE, BLEND_OP_SUBTRACT},
       {0, .25f, .625f, .25f}},
      {{BLEND_FACTOR_ONE, BLEND_FACTOR_ONE, BLEND_OP_REVERSE_SUBTRACT, BLEND_FACTOR_ONE, BLEND_FACTOR_ONE,
        BLEND_OP_REVERSE_SUBTRACT},
       {.25f, 0, 0, 0}},
      {{BLEND_FACTOR_ZERO, BLEND_FACTOR_ZERO, BLEND_OP_MIN, BLEND_FACTOR_ZERO, BLEND_FACTOR_ZERO, BLEND_OP_MIN},
       {.25f, .25f, .125f, .25f}},
      {{BLEND_FACTOR_ZERO, BLEND_FACTOR_ZERO, BLEND_OP_MAX, BLEND_FACTOR_ZERO, BLEND_FACTOR_ZERO, BLEND_OP_MAX},
       {.5f, .5f, .75f, .5f}},
  };
  const std::array<float, 4> factor_results[] = {
      {0, 0, 0, 0},
      {.25f, .5f, .75f, .5f},
      {.0625f, .25f, .5625f, .25f},
      {.1875f, .25f, .1875f, .25f},
      {.125f, .125f, .09375f, .125f},
      {.125f, .375f, .65625f, .375f},
      {.125f, .25f, .375f, .25f},
      {.125f, .25f, .375f, .25f},
      {.0625f, .125f, .1875f, .125f},
      {.1875f, .375f, .5625f, .375f},
  };
  for (int f = BLEND_FACTOR_ZERO; f <= BLEND_FACTOR_ONE_MINUS_DST_ALPHA; ++f) {
    const auto factor = static_cast<BlendFactor>(f);
    cases.push_back(
        {{factor, BLEND_FACTOR_ZERO, BLEND_OP_ADD, factor, BLEND_FACTOR_ZERO, BLEND_OP_ADD}, factor_results[f]});
  }
  for (size_t first = 0; first < cases.size(); first += targets.size()) {
    ProgramDesc desc;
    desc.vertex_shader = vs.Get();
    desc.fragment_shader = ps.Get();
    desc.color_formats.assign(targets.size(), IMAGE_FORMAT_RGBA8_UNORM);
    for (size_t j = 0; j < targets.size(); ++j)
      desc.blend_states.push_back(cases[(std::min)(first + j, cases.size() - 1)].state);
    ObjPtr<IProgram> program = device->CreateProgram(desc);
    ASSERT_TRUE(program);
    ObjPtr<ICommandContext> commands = device->CreateCommandContext();
    ASSERT_TRUE(commands);
    ASSERT_TRUE(commands->CmdBindProgram(program.Get()));
    RenderingInfo rendering;
    for (auto &target : targets)
      rendering.color_attachments.push_back(
          {target.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {.5f, .25f, .125f, .25f}});
    ASSERT_TRUE(commands->CmdBeginRendering(rendering));
    ASSERT_TRUE(commands->CmdSetViewport({0, 0, 8, 8, 0, 1}));
    ASSERT_TRUE(commands->CmdSetScissor({0, 0, 8, 8}));
    ASSERT_TRUE(commands->CmdDraw(3));
    ASSERT_TRUE(commands->CmdEndRendering());
    ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
    ASSERT_TRUE(done);
    ASSERT_TRUE(done->WaitHost());
    for (size_t j = 0; j < targets.size(); ++j) {
      const auto index = (std::min)(first + j, cases.size() - 1);
      SCOPED_TRACE(index);
      std::array<uint8_t, 4> pixel{};
      ReadPixel(device.Get(), targets[j].Get(), GetParam(), 4, 4, pixel);
      for (size_t c = 0; c < pixel.size(); ++c)
        EXPECT_NEAR(pixel[c], cases[index].expected[c] * 255, 2);
    }
  }
}

TEST_P(RhiImage, ComputeGraphicsSwitchingPreservesImagesAndDescriptorTables) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  Track(device.Get());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(queue);
  ObjPtr<IImage> a = device->CreateImage({4, 4, IMAGE_FORMAT_RGBA8_UNORM});
  ASSERT_TRUE(a);
  ObjPtr<IImage> b = device->CreateImage({4, 4, IMAGE_FORMAT_RGBA8_UNORM});
  ASSERT_TRUE(b);
  ObjPtr<IShader> cs = device->CreateShader({R"(
    RHI_RW_TEXTURE2D(0, 0, float4, target);
    [numthreads(4, 4, 1)] void CSMain(uint3 id:SV_DispatchThreadID) {
      target[id.xy] += float4(1.0 / 255.0, 0, 0, 0);
    }
  )",
                                             "CSMain", SHADER_STAGE_COMPUTE});
  ASSERT_TRUE(cs);
  ComputeProgramDesc cd;
  cd.compute_shader = cs.Get();
  cd.pipeline_layout.groups = {{{{0, BINDING_RW_TEXTURE2D, 1, SHADER_VISIBILITY_COMPUTE}}}};
  ObjPtr<IComputeProgram> compute = device->CreateComputeProgram(cd);
  ASSERT_TRUE(compute);
  const std::string render_source = R"(
    RHI_TEXTURE2D(0, 0, float4, source_image);
    float4 VSMain(uint id:SV_VertexID):SV_Position {
      float2 p[3] = {float2(-1,-1),float2(3,-1),float2(-1,3)};
      return float4(p[id],0,1);
    }
    float4 PSMain(float4 p:SV_Position):SV_Target0 { return source_image.Load(int3(uint2(p.xy),0)); }
  )";
  ObjPtr<IShader> vs = device->CreateShader({render_source, "VSMain", SHADER_STAGE_VERTEX});
  ASSERT_TRUE(vs);
  ObjPtr<IShader> ps = device->CreateShader({render_source, "PSMain", SHADER_STAGE_FRAGMENT});
  ASSERT_TRUE(ps);
  ProgramDesc gd;
  gd.vertex_shader = vs.Get();
  gd.fragment_shader = ps.Get();
  gd.pipeline_layout.groups = {{{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}}}};
  ObjPtr<IProgram> graphics = device->CreateProgram(gd);
  ASSERT_TRUE(graphics);
  ObjPtr<IBindingGroup> ca = device->CreateBindingGroup(cd.pipeline_layout.groups[0]);
  ASSERT_TRUE(ca);
  ObjPtr<IBindingGroup> cb = device->CreateBindingGroup(cd.pipeline_layout.groups[0]);
  ASSERT_TRUE(cb);
  ObjPtr<IBindingGroup> ga = device->CreateBindingGroup(gd.pipeline_layout.groups[0]);
  ASSERT_TRUE(ga);
  ASSERT_TRUE(ca->BindResource(0, a.Get()));
  ASSERT_TRUE(cb->BindResource(0, b.Get()));
  ASSERT_TRUE(ga->BindResource(0, a.Get()));
  ObjPtr<ICommandContext> ctx = device->CreateCommandContext();
  ASSERT_TRUE(ctx);
  ASSERT_TRUE(ctx->CmdClearColor(a.Get(), {0, 0, 0, 1}));
  for (int round = 0; round < 40; ++round) {
    ASSERT_TRUE(ctx->CmdBindComputeProgram(compute.Get()));
    ASSERT_TRUE(ctx->CmdBindGroup(0, ca.Get()));
    ASSERT_TRUE(ctx->CmdDispatch(1));
    ASSERT_TRUE(ctx->CmdBindProgram(graphics.Get()));
    ASSERT_TRUE(ctx->CmdBindGroup(0, ga.Get()));
    ASSERT_TRUE(ctx->CmdBeginRendering({{{b.Get(), ATTACHMENT_LOAD_OP_DISCARD, ATTACHMENT_STORE_OP_STORE}}}));
    ASSERT_TRUE(ctx->CmdSetViewport({0, 0, 4, 4, 0, 1}));
    ASSERT_TRUE(ctx->CmdSetScissor({0, 0, 4, 4}));
    ASSERT_TRUE(ctx->CmdDraw(3));
    ASSERT_TRUE(ctx->CmdEndRendering());
    ASSERT_TRUE(ctx->CmdBindComputeProgram(compute.Get()));
    ASSERT_TRUE(ctx->CmdBindGroup(0, cb.Get()));
    ASSERT_TRUE(ctx->CmdDispatch(1));
  }
  compute.Release();
  graphics.Release();
  ca.Release();
  cb.Release();
  ga.Release();
  ObjPtr<ISubmissionToken> token = queue->Submit(ctx.Get());
  ASSERT_TRUE(token);
  ASSERT_TRUE(token->WaitHost());
  std::array<uint8_t, 4> pixel{};
  ReadPixel(device.Get(), a.Get(), GetParam(), 1, 1, pixel);
  EXPECT_EQ(pixel[0], 40);
  ReadPixel(device.Get(), b.Get(), GetParam(), 1, 1, pixel);
  EXPECT_EQ(pixel[0], 41);
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiImage, testing::ValuesIn(backends));
}  // namespace
}  // namespace grassland::rhi

#endif
