#include "grassland/graphics/backend/metal/metal_core.h"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>

#include "grassland/graphics/backend/metal/metal_acceleration_structure.h"
#include "grassland/graphics/backend/metal/metal_buffer.h"
#include "grassland/graphics/backend/metal/metal_command_context.h"
#include "grassland/graphics/backend/metal/metal_image.h"
#include "grassland/graphics/backend/metal/metal_program.h"
#include "grassland/graphics/backend/metal/metal_sampler.h"
#include "grassland/graphics/backend/metal/metal_shader.h"
#include "grassland/graphics/backend/metal/metal_window.h"

namespace grassland::graphics::backend {

MetalCore::MetalCore(const Settings &settings) : Core(settings) {
}

MetalCore::~MetalCore() {
  try {
    WaitGPU();
  } catch (const std::exception &e) {
    LogWarning("{}", e.what());
  }
}

int MetalCore::GetPhysicalDeviceProperties(PhysicalDeviceProperties *properties) {
  MetalPool pool;
  auto devices = NS::TransferPtr(MTL::CopyAllDevices());
  if (properties)
    for (NS::UInteger i = 0; i < devices->count(); ++i) {
      auto device = devices->object<MTL::Device>(i);
      properties[i] = {device->name()->utf8String(), device->hasUnifiedMemory() ? 1000ull : 100ull, false, false};
    }
  return static_cast<int>(devices->count());
}

int MetalCore::InitializeLogicalDevice(int index) {
  MetalPool pool;
  auto devices = NS::TransferPtr(MTL::CopyAllDevices());
  if (index < 0 || index >= devices->count())
    return -1;
  device_ = NS::RetainPtr(devices->object<MTL::Device>(index));
  if (device_->argumentBuffersSupport() != MTL::ArgumentBuffersTier2 || !device_->hasUnifiedMemory())
    throw std::runtime_error("Metal backend requires Apple Silicon with tier 2 argument buffers");
  queue_ = NS::TransferPtr(device_->newCommandQueue());
  MetalCheck(queue_.get(), nullptr, "newCommandQueue");
  device_name_ = device_->name()->utf8String();
  ray_tracing_support_ = false;  // Sparkium uses its compute BVH and traversal backend.
  return 0;
}

uint32_t MetalCore::WaveSize() const {
  return 32;
}

void MetalCore::Reap(bool wait) {
  while (!pending_.empty()) {
    auto &submission = pending_.front();
    auto buffer = submission.buffer;
    if (wait)
      buffer->waitUntilCompleted();
    if (buffer->status() < MTL::CommandBufferStatusCompleted)
      break;
    auto callbacks = std::move(submission.callbacks);
    pending_.pop_front();
    if (buffer->status() == MTL::CommandBufferStatusError)
      MetalCheck(nullptr, buffer->error(), "Metal command submission");
    for (auto &callback : callbacks)
      callback();
  }
}

void MetalCore::WaitGPU() {
  Reap(true);
}

void MetalCore::Commit(MTL::CommandBuffer *buffer, std::vector<std::function<void()>> callbacks) {
  Reap(false);
  if (pending_.size() >= std::max(1, FramesInFlight())) {
    pending_.front().buffer->waitUntilCompleted();
    Reap(false);
  }
  pending_.push_back({NS::RetainPtr(buffer), std::move(callbacks)});
  buffer->commit();
  frame_ = (frame_ + 1) % std::max(1, FramesInFlight());
}

int MetalCore::SubmitCommandContext(CommandContext *context) {
  auto metal = dynamic_cast<MetalCommandContext *>(context);
  if (!metal || metal->GetCore() != this || metal->submitted)
    return -1;
  metal->EndEncoder();
  Commit(metal->Handle(), metal->GetPostExecutionCallbacks());
  metal->submitted = true;
  return 0;
}

int MetalCore::CreateBuffer(size_t size, BufferType type, double_ptr<Buffer> pp_buffer) {
  pp_buffer.construct<MetalBuffer>(this, size, type);
  return 0;
}

int MetalCore::CreateImage(int width, int height, ImageFormat format, double_ptr<Image> pp_image) {
  pp_image.construct<MetalImage>(this, width, height, format);
  return 0;
}

int MetalCore::CreateSampler(const SamplerInfo &info, double_ptr<Sampler> pp_sampler) {
  pp_sampler.construct<MetalSampler>(this, info);
  return 0;
}

int MetalCore::CreateWindowObject(int width,
                                  int height,
                                  const std::string &title,
                                  bool fullscreen,
                                  bool resizable,
                                  double_ptr<Window> pp_window) {
  pp_window.construct<MetalWindow>(this, width, height, title, fullscreen, resizable);
  return 0;
}

int MetalCore::CreateShader(const std::string &source_code,
                            const std::string &entry_point,
                            const std::string &target,
                            double_ptr<Shader> pp_shader) {
  VirtualFileSystem vfs;
  vfs.WriteFile("shader.hlsl", source_code);
  return CreateShader(vfs, "shader.hlsl", entry_point, target, pp_shader);
}

int MetalCore::CreateShader(const VirtualFileSystem &vfs,
                            const std::string &source_file,
                            const std::string &entry_point,
                            const std::string &target,
                            double_ptr<Shader> pp_shader) {
  return CreateShader(vfs, source_file, entry_point, target, {}, pp_shader);
}

int MetalCore::CreateShader(const VirtualFileSystem &vfs,
                            const std::string &source_file,
                            const std::string &entry_point,
                            const std::string &target,
                            const std::vector<std::string> &args,
                            double_ptr<Shader> pp_shader) {
  std::vector<std::string> compile_args = {"-spirv", "-fspv-target-env=vulkan1.2", "-fvk-use-dx-layout"};
  compile_args.insert(compile_args.end(), args.begin(), args.end());
  if (const char *directory = std::getenv("LONGMARCH_METAL_SHADER_DUMP"))
    vfs.SaveToDirectory(std::filesystem::path(directory) / "hlsl");
  auto blob = CompileShader(vfs, source_file, entry_point, target, compile_args);
  if (blob.data.empty())
    return -1;
  pp_shader.construct<MetalShader>(blob);
  return 0;
}

int MetalCore::CreateProgram(const std::vector<ImageFormat> &color_formats,
                             ImageFormat depth_format,
                             double_ptr<Program> pp_program) {
  pp_program.construct<MetalProgram>(this, color_formats, depth_format);
  return 0;
}

int MetalCore::CreateComputeProgram(Shader *compute_shader, double_ptr<ComputeProgram> pp_program) {
  pp_program.construct<MetalComputeProgram>(this, compute_shader);
  return 0;
}

int MetalCore::CreateCommandContext(double_ptr<CommandContext> pp_command_context) {
  pp_command_context.construct<MetalCommandContext>(this);
  return 0;
}

int MetalCore::CreateBottomLevelAccelerationStructure(BufferRange aabbs,
                                                      uint32_t stride,
                                                      uint32_t count,
                                                      RayTracingGeometryFlag flags,
                                                      double_ptr<AccelerationStructure> result) {
  result.construct<MetalAccelerationStructure>(this, aabbs, stride, count, flags);
  return 0;
}

int MetalCore::CreateBottomLevelAccelerationStructure(BufferRange vertices,
                                                      BufferRange indices,
                                                      uint32_t vertex_count,
                                                      uint32_t stride,
                                                      uint32_t triangle_count,
                                                      RayTracingGeometryFlag flags,
                                                      double_ptr<AccelerationStructure> result) {
  result.construct<MetalAccelerationStructure>(this, vertices, indices, vertex_count, stride, triangle_count, flags);
  return 0;
}

int MetalCore::CreateBottomLevelAccelerationStructure(Buffer *vertices,
                                                      Buffer *indices,
                                                      uint32_t stride,
                                                      double_ptr<AccelerationStructure> result) {
  if (!vertices || !indices || stride < 12 || vertices->Size() / stride > UINT32_MAX ||
      indices->Size() / 12 > UINT32_MAX)
    throw std::invalid_argument("invalid Metal mesh buffers");
  return CreateBottomLevelAccelerationStructure(vertices->Range(), indices->Range(), vertices->Size() / stride, stride,
                                                indices->Size() / 12, RAYTRACING_GEOMETRY_FLAG_NONE, result);
}

int MetalCore::CreateTopLevelAccelerationStructure(const std::vector<RayTracingInstance> &instances,
                                                   double_ptr<AccelerationStructure> result) {
  result.construct<MetalAccelerationStructure>(this, instances);
  return 0;
}

int MetalCore::CreateRayTracingProgram(double_ptr<RayTracingProgram>) {
  return -1;
}

}  // namespace grassland::graphics::backend
