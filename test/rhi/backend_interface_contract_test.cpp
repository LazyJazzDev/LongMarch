#include <type_traits>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_buffer.h"
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_compute_program.h"
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/backend_program.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_ray_tracing_program.h"
#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/backend/backend_shader.h"
#include "grassland/rhi/backend/backend_shader_binding_table.h"
#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/backend_window.h"

namespace grassland::rhi::backend {
namespace {
// Apart from Object lifetime management, shared classes inherit RHI entry points unchanged. Comparing member
// pointer types also catches overrides that omit the override keyword.

static_assert(std::is_abstract_v<BindingGroup>);
static_assert(std::is_same_v<decltype(&BindingGroup::GetLayout), decltype(&IBindingGroup::GetLayout)>);

static_assert(std::is_abstract_v<Buffer>);
static_assert(std::is_same_v<decltype(&Buffer::GetDesc), decltype(&IBuffer::GetDesc)>);

static_assert(std::is_abstract_v<CommandContext>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdQueryCompactedSize),
                             decltype(&ICommandContext::CmdQueryCompactedSize)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdCompactAccelerationStructure),
                             decltype(&ICommandContext::CmdCompactAccelerationStructure)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdBuildBottomLevelAccelerationStructure),
                             decltype(&ICommandContext::CmdBuildBottomLevelAccelerationStructure)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdBuildTopLevelAccelerationStructure),
                             decltype(&ICommandContext::CmdBuildTopLevelAccelerationStructure)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdBindRayTracingProgram),
                             decltype(&ICommandContext::CmdBindRayTracingProgram)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdDispatchRays), decltype(&ICommandContext::CmdDispatchRays)>);
static_assert(
    std::is_same_v<decltype(&CommandContext::CmdDownloadBuffer), decltype(&ICommandContext::CmdDownloadBuffer)>);
static_assert(
    std::is_same_v<decltype(&CommandContext::CmdDownloadImage), decltype(&ICommandContext::CmdDownloadImage)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdUploadBuffer), decltype(&ICommandContext::CmdUploadBuffer)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdCopyBuffer), decltype(&ICommandContext::CmdCopyBuffer)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdBindGroup), decltype(&ICommandContext::CmdBindGroup)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdUploadImage), decltype(&ICommandContext::CmdUploadImage)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdBindComputeProgram),
                             decltype(&ICommandContext::CmdBindComputeProgram)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdDispatch), decltype(&ICommandContext::CmdDispatch)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdBindProgram), decltype(&ICommandContext::CmdBindProgram)>);
static_assert(
    std::is_same_v<decltype(&CommandContext::CmdBeginRendering), decltype(&ICommandContext::CmdBeginRendering)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdEndRendering), decltype(&ICommandContext::CmdEndRendering)>);
static_assert(
    std::is_same_v<decltype(&CommandContext::CmdBindVertexBuffer), decltype(&ICommandContext::CmdBindVertexBuffer)>);
static_assert(
    std::is_same_v<decltype(&CommandContext::CmdBindIndexBuffer), decltype(&ICommandContext::CmdBindIndexBuffer)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdSetViewport), decltype(&ICommandContext::CmdSetViewport)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdSetScissor), decltype(&ICommandContext::CmdSetScissor)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdDraw), decltype(&ICommandContext::CmdDraw)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdDrawIndexed), decltype(&ICommandContext::CmdDrawIndexed)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdClearImage), decltype(&ICommandContext::CmdClearImage)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdDrawImGui), decltype(&ICommandContext::CmdDrawImGui)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdPresentImage), decltype(&ICommandContext::CmdPresentImage)>);
static_assert(std::is_same_v<decltype(&CommandContext::CmdPresent), decltype(&ICommandContext::CmdPresent)>);

static_assert(std::is_abstract_v<ComputeProgram>);
static_assert(std::is_same_v<decltype(&ComputeProgram::GetDesc), decltype(&IComputeProgram::GetDesc)>);

static_assert(std::is_abstract_v<Core>);
static_assert(std::is_same_v<decltype(&Core::GetPhysicalDevices), decltype(&ICore::GetPhysicalDevices)>);
static_assert(std::is_same_v<decltype(&Core::CreateDevice), decltype(&ICore::CreateDevice)>);
static_assert(
    std::is_same_v<decltype(&Core::CreateDeviceByAutoSelection), decltype(&ICore::CreateDeviceByAutoSelection)>);

static_assert(std::is_abstract_v<Device>);
static_assert(std::is_same_v<decltype(&Device::CreateBottomLevelAccelerationStructure),
                             decltype(&IDevice::CreateBottomLevelAccelerationStructure)>);
static_assert(std::is_same_v<decltype(&Device::CreateTopLevelAccelerationStructure),
                             decltype(&IDevice::CreateTopLevelAccelerationStructure)>);
static_assert(std::is_same_v<decltype(&Device::CreateRayTracingProgram), decltype(&IDevice::CreateRayTracingProgram)>);
static_assert(std::is_same_v<decltype(&Device::Name), decltype(&IDevice::Name)>);
static_assert(
    std::is_same_v<decltype(&Device::SupportRayTracingPipeline), decltype(&IDevice::SupportRayTracingPipeline)>);
static_assert(std::is_same_v<decltype(&Device::SupportRayQuery), decltype(&IDevice::SupportRayQuery)>);
static_assert(std::is_same_v<decltype(&Device::SupportGeometryShader), decltype(&IDevice::SupportGeometryShader)>);
static_assert(std::is_same_v<decltype(&Device::CreateImage), decltype(&IDevice::CreateImage)>);
static_assert(std::is_same_v<decltype(&Device::CreateSampler), decltype(&IDevice::CreateSampler)>);
static_assert(std::is_same_v<decltype(&Device::CreateBindingGroup), decltype(&IDevice::CreateBindingGroup)>);
static_assert(std::is_same_v<decltype(&Device::CreateBuffer), decltype(&IDevice::CreateBuffer)>);
static_assert(std::is_same_v<decltype(&Device::CreateComputeProgram), decltype(&IDevice::CreateComputeProgram)>);
static_assert(std::is_same_v<decltype(&Device::CreateProgram), decltype(&IDevice::CreateProgram)>);

template <typename Owner>
constexpr bool IsShaderFactoryInterface(IShader *(Owner::*)(const ShaderDesc &)) {
  return std::is_same_v<Owner, IDevice>;
}

static_assert(IsShaderFactoryInterface(&Device::CreateShader));
static_assert(std::is_same_v<decltype(&Device::CreateGLFWWindow), decltype(&IDevice::CreateGLFWWindow)>);
static_assert(std::is_same_v<decltype(&Device::GetQueue), decltype(&IDevice::GetQueue)>);
static_assert(std::is_same_v<decltype(&Device::CreateCommandContext), decltype(&IDevice::CreateCommandContext)>);

static_assert(std::is_abstract_v<DownloadToken>);
static_assert(std::is_same_v<decltype(&DownloadToken::GetStatus), decltype(&IDownloadToken::GetStatus)>);
static_assert(std::is_same_v<decltype(&DownloadToken::WaitHost), decltype(&IDownloadToken::WaitHost)>);
static_assert(std::is_same_v<decltype(&DownloadToken::GetSize), decltype(&IDownloadToken::GetSize)>);
static_assert(std::is_same_v<decltype(&DownloadToken::CopyTo), decltype(&IDownloadToken::CopyTo)>);

static_assert(std::is_abstract_v<Image>);
static_assert(std::is_same_v<decltype(&Image::GetDesc), decltype(&IImage::GetDesc)>);
static_assert(std::is_same_v<decltype(&Image::IsValid), decltype(&IImage::IsValid)>);

static_assert(std::is_abstract_v<Program>);
static_assert(std::is_same_v<decltype(&Program::GetDesc), decltype(&IProgram::GetDesc)>);

static_assert(std::is_abstract_v<Queue>);
static_assert(std::is_same_v<decltype(&Queue::Submit), decltype(&IQueue::Submit)>);
static_assert(std::is_same_v<decltype(&Queue::GetType), decltype(&IQueue::GetType)>);
static_assert(std::is_same_v<decltype(&Queue::SharesPhysicalQueue), decltype(&IQueue::SharesPhysicalQueue)>);

static_assert(std::is_abstract_v<RayTracingProgram>);
static_assert(std::is_same_v<decltype(&RayTracingProgram::CreateShaderBindingTable),
                             decltype(&IRayTracingProgram::CreateShaderBindingTable)>);
static_assert(std::is_same_v<decltype(&RayTracingProgram::GetDesc), decltype(&IRayTracingProgram::GetDesc)>);

static_assert(std::is_abstract_v<Sampler>);
static_assert(std::is_same_v<decltype(&Sampler::GetDesc), decltype(&ISampler::GetDesc)>);

static_assert(std::is_abstract_v<Shader>);
static_assert(std::is_same_v<decltype(&Shader::GetStage), decltype(&IShader::GetStage)>);
static_assert(std::is_same_v<decltype(&Shader::GetEntry), decltype(&IShader::GetEntry)>);
static_assert(std::is_same_v<decltype(&Shader::GetPreprocessedCode), decltype(&IShader::GetPreprocessedCode)>);

static_assert(std::is_abstract_v<ShaderBindingTable>);
static_assert(std::is_same_v<decltype(&ShaderBindingTable::GetDesc), decltype(&IShaderBindingTable::GetDesc)>);
static_assert(std::is_same_v<decltype(&ShaderBindingTable::GetProgram), decltype(&IShaderBindingTable::GetProgram)>);

static_assert(std::is_abstract_v<SubmissionToken>);
static_assert(std::is_same_v<decltype(&SubmissionToken::GetStatus), decltype(&ISubmissionToken::GetStatus)>);
static_assert(std::is_same_v<decltype(&SubmissionToken::WaitHost), decltype(&ISubmissionToken::WaitHost)>);

static_assert(std::is_abstract_v<Window>);
static_assert(std::is_same_v<decltype(&Window::ResizeEvent), decltype(&IWindow::ResizeEvent)>);
static_assert(std::is_same_v<decltype(&Window::FramebufferResizeEvent), decltype(&IWindow::FramebufferResizeEvent)>);
static_assert(std::is_same_v<decltype(&Window::CursorEnterEvent), decltype(&IWindow::CursorEnterEvent)>);
static_assert(std::is_same_v<decltype(&Window::FocusEvent), decltype(&IWindow::FocusEvent)>);
static_assert(std::is_same_v<decltype(&Window::MouseMoveEvent), decltype(&IWindow::MouseMoveEvent)>);
static_assert(std::is_same_v<decltype(&Window::MouseButtonEvent), decltype(&IWindow::MouseButtonEvent)>);
static_assert(std::is_same_v<decltype(&Window::ScrollEvent), decltype(&IWindow::ScrollEvent)>);
static_assert(std::is_same_v<decltype(&Window::KeyEvent), decltype(&IWindow::KeyEvent)>);
static_assert(std::is_same_v<decltype(&Window::CharEvent), decltype(&IWindow::CharEvent)>);
static_assert(std::is_same_v<decltype(&Window::DropEvent), decltype(&IWindow::DropEvent)>);
static_assert(std::is_same_v<decltype(&Window::EnableImGui), decltype(&IWindow::EnableImGui)>);
static_assert(std::is_same_v<decltype(&Window::UpdateImGui), decltype(&IWindow::UpdateImGui)>);
static_assert(std::is_same_v<decltype(&Window::GLFWWindow), decltype(&IWindow::GLFWWindow)>);
static_assert(std::is_same_v<decltype(&Window::ShouldClose), decltype(&IWindow::ShouldClose)>);
static_assert(std::is_same_v<decltype(&Window::IsKeyDown), decltype(&IWindow::IsKeyDown)>);
static_assert(std::is_same_v<decltype(&Window::SetTitle), decltype(&IWindow::SetTitle)>);
static_assert(std::is_same_v<decltype(&Window::GetNextImage), decltype(&IWindow::GetNextImage)>);
static_assert(std::is_same_v<decltype(&Window::IsSwapChainInvalid), decltype(&IWindow::IsSwapChainInvalid)>);
static_assert(std::is_same_v<decltype(&Window::RebuildSwapChain), decltype(&IWindow::RebuildSwapChain)>);
static_assert(std::is_same_v<decltype(&Window::SetPresentMode), decltype(&IWindow::SetPresentMode)>);
static_assert(std::is_same_v<decltype(&Window::GetPresentMode), decltype(&IWindow::GetPresentMode)>);
static_assert(std::is_same_v<decltype(&Window::SetColorSpace), decltype(&IWindow::SetColorSpace)>);
static_assert(std::is_same_v<decltype(&Window::GetColorSpace), decltype(&IWindow::GetColorSpace)>);
static_assert(
    std::is_same_v<decltype(&Window::GetHDRReferenceWhiteNits), decltype(&IWindow::GetHDRReferenceWhiteNits)>);
static_assert(
    std::is_same_v<decltype(&Window::GetHDRReferenceWhiteScale), decltype(&IWindow::GetHDRReferenceWhiteScale)>);
static_assert(std::is_same_v<decltype(&Window::GetHDRHeadroom), decltype(&IWindow::GetHDRHeadroom)>);
static_assert(std::is_same_v<decltype(&Window::GetSupportedPresentationOptions),
                             decltype(&IWindow::GetSupportedPresentationOptions)>);

template <typename Resource, typename Owner>
constexpr bool IsBindingInterface(bool (Owner::*)(uint32_t, const std::vector<Resource> &, uint32_t)) {
  return std::is_same_v<Owner, IBindingGroup>;
}

static_assert(IsBindingInterface<BufferView>(&BindingGroup::BindResource));
static_assert(IsBindingInterface<IImage *>(&BindingGroup::BindResource));
static_assert(IsBindingInterface<ISampler *>(&BindingGroup::BindResource));
static_assert(IsBindingInterface<ITopLevelAccelerationStructure *>(&BindingGroup::BindResource));
// Object is the shared lifetime-management exception.
static_assert(!std::is_abstract_v<Object<IObject>>);
}  // namespace
}  // namespace grassland::rhi::backend
