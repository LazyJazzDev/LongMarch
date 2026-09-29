#include "grassland/graphics/backend/webgpu/webgpu_core.h"

#include <memory>
#include <stdexcept>

#include "grassland/graphics/backend/webgpu/webgpu_buffer.h"
#include "grassland/graphics/backend/webgpu/webgpu_command_context.h"
#include "grassland/graphics/backend/webgpu/webgpu_image.h"
#include "grassland/graphics/backend/webgpu/webgpu_program.h"
#include "grassland/graphics/backend/webgpu/webgpu_sampler.h"
#include "grassland/graphics/backend/webgpu/webgpu_shader.h"

namespace grassland::graphics::backend {

namespace {
wgpu::Device &HostDevice() {
  static wgpu::Device device;
  return device;
}
}  // namespace

WebGPUCore::WebGPUCore(const Settings &settings) : Core(settings) {
}

void WebGPUCore::SetDevice(wgpu::Device device) {
  HostDevice() = std::move(device);
}

int WebGPUCore::GetPhysicalDeviceProperties(PhysicalDeviceProperties *properties) {
  if (!HostDevice())
    return 0;
  if (properties)
    properties[0] = {"WebGPU", 1000ull, false, false};
  return 1;
}

int WebGPUCore::InitializeLogicalDevice(int index) {
  if (index != 0 || !HostDevice())
    return -1;
  device_ = HostDevice();
  queue_ = device_.GetQueue();
  device_name_ = "WebGPU";
  ray_tracing_support_ = false;
  return 0;
}

int WebGPUCore::SubmitCommandContext(CommandContext *context) {
  auto webgpu = dynamic_cast<WebGPUCommandContext *>(context);
  if (!webgpu || webgpu->GetCore() != this || webgpu->submitted)
    return -1;
  auto commands = webgpu->Finish();
  queue_.Submit(1, &commands);
  webgpu->submitted = true;
  auto callbacks = std::make_shared<std::vector<std::function<void()>>>(webgpu->GetPostExecutionCallbacks());
  if (!callbacks->empty())
    queue_.OnSubmittedWorkDone(wgpu::CallbackMode::AllowSpontaneous,
                               [callbacks](wgpu::QueueWorkDoneStatus, wgpu::StringView) {
                                 for (auto &callback : *callbacks)
                                   callback();
                               });
  frame_ = (frame_ + 1) % std::max(1, FramesInFlight());
  return 0;
}

int WebGPUCore::CreateBuffer(size_t size, BufferType type, double_ptr<Buffer> pp_buffer) {
  pp_buffer.construct<WebGPUBuffer>(this, size, type);
  return 0;
}

int WebGPUCore::CreateImage(int width, int height, ImageFormat format, double_ptr<Image> pp_image) {
  pp_image.construct<WebGPUImage>(this, width, height, format);
  return 0;
}

int WebGPUCore::CreateSampler(const SamplerInfo &info, double_ptr<Sampler> pp_sampler) {
  pp_sampler.construct<WebGPUSampler>(this, info);
  return 0;
}

int WebGPUCore::CreateWindowObject(int, int, const std::string &, bool, bool, double_ptr<Window>) {
  // Browser pages host their canvas; applications use hosted windows.
  return -1;
}

int WebGPUCore::CreateShader(const std::string &source_code,
                             const std::string &entry_point,
                             const std::string &target,
                             double_ptr<Shader> pp_shader) {
  VirtualFileSystem vfs;
  vfs.WriteFile("shader.slang", source_code);
  return CreateShader(vfs, "shader.slang", entry_point, target, pp_shader);
}

int WebGPUCore::CreateShader(const VirtualFileSystem &vfs,
                             const std::string &source_file,
                             const std::string &entry_point,
                             const std::string &target,
                             double_ptr<Shader> pp_shader) {
  return CreateShader(vfs, source_file, entry_point, target, {}, pp_shader);
}

int WebGPUCore::CreateShader(const VirtualFileSystem &vfs,
                             const std::string &source_file,
                             const std::string &entry_point,
                             const std::string &target,
                             const std::vector<std::string> &args,
                             double_ptr<Shader> pp_shader) {
  // Slang emits @group(space) @binding(register) from the same declarations.
  std::vector<std::string> compile_args = {"-target", "wgsl"};
  compile_args.insert(compile_args.end(), args.begin(), args.end());
  auto blob = CompileShader(vfs, source_file, entry_point, target, compile_args);
  if (blob.data.empty())
    return -1;
  pp_shader.construct<WebGPUShader>(this, blob);
  return 0;
}

int WebGPUCore::CreateProgram(const std::vector<ImageFormat> &color_formats,
                              ImageFormat depth_format,
                              double_ptr<Program> pp_program) {
  pp_program.construct<WebGPUProgram>(this, color_formats, depth_format);
  return 0;
}

int WebGPUCore::CreateComputeProgram(Shader *compute_shader, double_ptr<ComputeProgram> pp_program) {
  pp_program.construct<WebGPUComputeProgram>(this, dynamic_cast<WebGPUShader *>(compute_shader));
  return 0;
}

int WebGPUCore::CreateCommandContext(double_ptr<CommandContext> pp_command_context) {
  pp_command_context.construct<WebGPUCommandContext>(this);
  return 0;
}

int WebGPUCore::CreateBottomLevelAccelerationStructure(BufferRange,
                                                       uint32_t,
                                                       uint32_t,
                                                       RayTracingGeometryFlag,
                                                       double_ptr<AccelerationStructure>) {
  return -1;
}

int WebGPUCore::CreateBottomLevelAccelerationStructure(BufferRange,
                                                       BufferRange,
                                                       uint32_t,
                                                       uint32_t,
                                                       uint32_t,
                                                       RayTracingGeometryFlag,
                                                       double_ptr<AccelerationStructure>) {
  return -1;
}

int WebGPUCore::CreateBottomLevelAccelerationStructure(Buffer *,
                                                       Buffer *,
                                                       uint32_t,
                                                       double_ptr<AccelerationStructure>) {
  return -1;
}

int WebGPUCore::CreateTopLevelAccelerationStructure(const std::vector<RayTracingInstance> &,
                                                    double_ptr<AccelerationStructure>) {
  return -1;
}

int WebGPUCore::CreateRayTracingProgram(double_ptr<RayTracingProgram>) {
  return -1;
}

}  // namespace grassland::graphics::backend
