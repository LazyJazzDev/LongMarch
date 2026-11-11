#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend {
class BindingGroup;
}

namespace grassland::rhi::backend {
class Buffer;
}

namespace grassland::rhi::backend {
class CommandContext;
}

namespace grassland::rhi::backend {
class ComputeProgram;
}

namespace grassland::rhi::backend {
class Core;
}

namespace grassland::rhi::backend {
class Device;
}

namespace grassland::rhi::backend {
class DownloadToken;
}

namespace grassland::rhi::backend {
class Image;
}

namespace grassland::rhi::backend {
class Program;
}

namespace grassland::rhi::backend {
class Queue;
}

namespace grassland::rhi::backend {
class RayTracingProgram;
}

namespace grassland::rhi::backend {
class Sampler;
}

namespace grassland::rhi::backend {
class Shader;
}

namespace grassland::rhi::backend {
class ShaderBindingTable;
}

namespace grassland::rhi::backend {
class SubmissionToken;
}

namespace grassland::rhi::backend {
class Timeline;
}

namespace grassland::rhi::backend {
class Window;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12BottomLevelAccelerationStructure;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12TopLevelAccelerationStructure;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12AccelerationStructure;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12AllocatedBuffer;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12AllocatedImage;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12BindingGroup;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12Buffer;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12CommandContext;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12ComputeProgram;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12Core;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12Device;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12DownloadToken;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12Image;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12Program;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12ProgramBase;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12Queue;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12RayTracingProgram;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12Sampler;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12SwapChainImage;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12TemporalBufferLease;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12Timeline;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12Window;
}

namespace grassland::rhi::backend::metal {
class MetalBottomLevelAccelerationStructure;
}

namespace grassland::rhi::backend::metal {
class MetalTopLevelAccelerationStructure;
}

namespace grassland::rhi::backend::metal {
class MetalAccelerationStructure;
}

namespace grassland::rhi::backend::metal {
class MetalAllocatedBuffer;
}

namespace grassland::rhi::backend::metal {
class MetalAllocatedImage;
}

namespace grassland::rhi::backend::metal {
class MetalBindingGroup;
}

namespace grassland::rhi::backend::metal {
class MetalBuffer;
}

namespace grassland::rhi::backend::metal {
class MetalCommandContext;
}

namespace grassland::rhi::backend::metal {
class MetalComputeProgram;
}

namespace grassland::rhi::backend::metal {
class MetalCore;
}

namespace grassland::rhi::backend::metal {
class MetalDevice;
}

namespace grassland::rhi::backend::metal {
class MetalDownloadToken;
}

namespace grassland::rhi::backend::metal {
class MetalImage;
}

namespace grassland::rhi::backend::metal {
class MetalProgram;
}

namespace grassland::rhi::backend::metal {
class MetalProgramBase;
}

namespace grassland::rhi::backend::metal {
class MetalQueue;
}

namespace grassland::rhi::backend::metal {
class MetalSampler;
}

namespace grassland::rhi::backend::metal {
class MetalSwapChainImage;
}

namespace grassland::rhi::backend::metal {
class MetalTemporalBufferLease;
}

namespace grassland::rhi::backend::metal {
class MetalTimeline;
}

namespace grassland::rhi::backend::metal {
class MetalWindow;
}

namespace grassland::rhi::backend::vulkan {
class VulkanBottomLevelAccelerationStructure;
}

namespace grassland::rhi::backend::vulkan {
class VulkanTopLevelAccelerationStructure;
}

namespace grassland::rhi::backend::vulkan {
class VulkanAccelerationStructure;
}

namespace grassland::rhi::backend::vulkan {
class VulkanAllocatedBuffer;
}

namespace grassland::rhi::backend::vulkan {
class VulkanAllocatedImage;
}

namespace grassland::rhi::backend::vulkan {
class VulkanBindingGroup;
}

namespace grassland::rhi::backend::vulkan {
class VulkanBuffer;
}

namespace grassland::rhi::backend::vulkan {
class VulkanCommandContext;
}

namespace grassland::rhi::backend::vulkan {
class VulkanComputeProgram;
}

namespace grassland::rhi::backend::vulkan {
class VulkanCore;
}

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;
}

namespace grassland::rhi::backend::vulkan {
class VulkanDownloadToken;
}

namespace grassland::rhi::backend::vulkan {
class VulkanImage;
}

namespace grassland::rhi::backend::vulkan {
class VulkanProgram;
}

namespace grassland::rhi::backend::vulkan {
class VulkanProgramBase;
}

namespace grassland::rhi::backend::vulkan {
class VulkanQueue;
}

namespace grassland::rhi::backend::vulkan {
class VulkanRayTracingProgram;
}

namespace grassland::rhi::backend::vulkan {
class VulkanSampler;
}

namespace grassland::rhi::backend::vulkan {
class VulkanSwapChainImage;
}

namespace grassland::rhi::backend::vulkan {
class VulkanTemporalBufferLease;
}

namespace grassland::rhi::backend::vulkan {
class VulkanTimeline;
}

namespace grassland::rhi::backend::vulkan {
class VulkanWindow;
}

// UUIDv4 values generated once with Python uuid.uuid4(); keep published IDs stable.
LONGMARCH_RHI_UUID(grassland::rhi::backend::BindingGroup, "84d77c02-7724-4c23-ab73-333b6878b5aa")
LONGMARCH_RHI_UUID(grassland::rhi::backend::Buffer, "a86fae4e-f66b-43e7-89b4-3d5c431bf0d3")
LONGMARCH_RHI_UUID(grassland::rhi::backend::CommandContext, "4a13943e-6f88-4739-91c4-0eb6b6a04a0c")
LONGMARCH_RHI_UUID(grassland::rhi::backend::ComputeProgram, "f64ab67b-e062-4368-8605-dc1700c88e59")
LONGMARCH_RHI_UUID(grassland::rhi::backend::Core, "e4b44f28-4fd0-428a-927d-19a615be66fc")
LONGMARCH_RHI_UUID(grassland::rhi::backend::Device, "6d6cc71b-9a19-4b0d-ac68-84d86e402f5d")
LONGMARCH_RHI_UUID(grassland::rhi::backend::DownloadToken, "ed7d2ceb-b722-4881-ad53-9d19894b4bb2")
LONGMARCH_RHI_UUID(grassland::rhi::backend::Image, "1c62d59a-7c6e-4442-8855-d6b1d1b4804c")
LONGMARCH_RHI_UUID(grassland::rhi::backend::Program, "4a916e47-6d7c-4f6b-93bc-ed51bba0eaec")
LONGMARCH_RHI_UUID(grassland::rhi::backend::Queue, "5b1b277a-1b68-4947-bf02-14449939898d")
LONGMARCH_RHI_UUID(grassland::rhi::backend::RayTracingProgram, "e0b9733f-8a37-4702-800c-3022865d522c")
LONGMARCH_RHI_UUID(grassland::rhi::backend::Sampler, "34cc865a-50d9-49f8-b8db-591bc452621f")
LONGMARCH_RHI_UUID(grassland::rhi::backend::Shader, "74166d66-39c8-4b6f-b9bc-1721e275a3c2")
LONGMARCH_RHI_UUID(grassland::rhi::backend::ShaderBindingTable, "bf62c599-58f1-4e20-b3b8-b9b23eac18bc")
LONGMARCH_RHI_UUID(grassland::rhi::backend::SubmissionToken, "92c56d21-d1e3-4070-a232-2cc80606c437")
LONGMARCH_RHI_UUID(grassland::rhi::backend::Timeline, "1eefb6fe-e220-483b-a0da-92f046be5fe6")
LONGMARCH_RHI_UUID(grassland::rhi::backend::Window, "093b830a-4e7b-4d76-8df4-9a2693735c9d")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12BottomLevelAccelerationStructure,
                   "5531d910-c7c4-412a-b787-1d220ae37cdb")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12TopLevelAccelerationStructure,
                   "4755c2d8-b4b6-477e-809f-939fc3683688")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12AccelerationStructure, "c8f939f7-1266-44ee-ad3e-f0efb8e7007c")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12AllocatedBuffer, "f17438f1-4828-4dc2-943d-11e8afe77444")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12AllocatedImage, "3d5647bf-0a89-4f7f-bce8-dd9e99c879a1")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12BindingGroup, "b827b0e0-c57d-4316-b632-77a41d222ae7")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12Buffer, "29cbc4ee-c77c-48a4-8c31-de9ee75c704d")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12CommandContext, "5962a78a-e01d-4cc1-9f13-4544d7232c63")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12ComputeProgram, "f0ee363d-3b36-4a53-8e17-b8a0ea11bf0a")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12Core, "0ff7c9ac-b8cc-4d2c-b5ec-9995a69b861d")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12Device, "d258bd31-2295-4f22-9c98-bd0133662c47")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12DownloadToken, "c5315cf9-1ed2-46ac-b619-f12be68cb0ae")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12Image, "85eb3c59-b643-4c32-b41f-66ed7e262a3e")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12Program, "2919454c-7ff9-4ce3-b919-40901e038398")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12ProgramBase, "1a8ecd83-21dd-483c-bd0b-6efba7356e93")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12Queue, "46194ddb-7a1a-4d51-b878-0e51d212d642")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12RayTracingProgram, "5d1da804-a5b3-4469-a380-9d7a44cee916")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12Sampler, "d1e384fa-d35c-42e0-8d5f-110c64fe99d0")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12SwapChainImage, "f03d8d79-9e19-41ca-b585-8f6f5bd64143")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12TemporalBufferLease, "47d93f8f-2736-4b79-87e7-7cadabcf4f7a")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12Timeline, "32d5dccb-60c3-4ca2-af93-4b67d85e7b16")
LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12Window, "1babd07b-773d-4d2b-9392-ec5113e23fdf")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalBottomLevelAccelerationStructure,
                   "5efa0e0c-89c2-4fdf-8295-1b1e502f6363")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalTopLevelAccelerationStructure,
                   "240cc2a8-c334-4c86-b106-32c7fcc0f439")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalAccelerationStructure, "15b08793-ad90-4431-b925-abab83480476")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalAllocatedBuffer, "19d20694-8b38-47a4-a88f-56d5828c0e9d")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalAllocatedImage, "4ab5d857-2e2e-4232-bd99-aca3979a9c87")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalBindingGroup, "cc9ee713-7a82-44a9-9c5d-6b138c2b32bd")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalBuffer, "203e6dfb-bb02-4155-900a-b13f43af47bc")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalCommandContext, "cd04e62d-678c-4e07-aef6-f5fb982bba78")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalComputeProgram, "848b485d-2208-4d1d-86ae-cadee6182bfe")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalCore, "30ef4d7e-ae80-4882-b15d-023c44be58cf")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalDevice, "0edf854d-4ccc-4b02-8c0c-030ded3b67dc")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalDownloadToken, "cb7cec7d-cb18-4503-8444-82a8cde172f9")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalImage, "3c717669-dee3-407f-92be-d6485832e1c3")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalProgram, "26dfc710-560c-4dff-bfbd-2adc53cd3739")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalProgramBase, "48d78e4c-57e0-4ad1-9dc1-254d12006fc6")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalQueue, "b34deab1-e42d-43f3-b6c4-f47dc7638e30")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalSampler, "c609fe39-6f33-4a10-a4b1-db4fa86332b7")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalSwapChainImage, "e121e257-92bc-4748-bbe0-2eff7a5af162")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalTemporalBufferLease, "d8febe53-823a-4731-847d-f2f1766778c8")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalTimeline, "ca12a3b9-c2c2-4f0b-b396-324edaa7e96c")
LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalWindow, "e074a51b-c30d-43f4-8b6c-2d8fd1540424")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanBottomLevelAccelerationStructure,
                   "addcc51e-67e0-412d-befb-88c713400587")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanTopLevelAccelerationStructure,
                   "e2af066b-da31-4c27-93ea-c2e1e39f8499")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanAccelerationStructure, "8fd412fc-9636-4f9b-8c14-f6b47ac173db")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanAllocatedBuffer, "0e429c4d-302f-421e-90d8-b29a6593b7d3")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanAllocatedImage, "cc7d23e5-e7de-45ae-a4d6-2f954645a1ab")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanBindingGroup, "d572985f-e699-4c7c-b9da-2c57af2e528c")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanBuffer, "edd00360-77f8-46d3-9659-0c06d6616086")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanCommandContext, "91254acd-ec4a-490b-a6f9-ca359d27d0e5")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanComputeProgram, "3ce12e7b-3db2-461e-b7d0-9ec0f172ebd1")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanCore, "38b5aac6-db8d-4fe0-86ac-ed06f0b3c970")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanDevice, "96a1c412-20e1-40bc-b542-d515f9a41965")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanDownloadToken, "1f0619cd-eb9a-4ecc-8260-c1737da81a3b")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanImage, "bab85247-2604-44fe-b152-1ea15663177a")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanProgram, "d24a9c9f-6f6f-4489-a840-064c2a4309c1")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanProgramBase, "267525b2-441a-4f53-a2e2-826fd05cd6d2")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanQueue, "09666973-0b3a-4165-bb7f-0faf38e0f01a")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanRayTracingProgram, "dbbe212c-3234-418b-b6b0-6676da637f09")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanSampler, "cbdfdbcb-a546-480b-a5d0-a86e1d181b71")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanSwapChainImage, "f9decc34-658f-4a62-bf5b-e7a749c2c44e")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanTemporalBufferLease, "b8b91e0e-bc72-4ea2-b918-d6135648577f")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanTimeline, "30a1e28a-4bf1-4d83-a693-bc60f95ec95b")
LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::VulkanWindow, "2bd587e1-b088-4f5c-8138-3fd885407013")
