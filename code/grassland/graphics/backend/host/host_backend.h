#pragma once
// Host (CPU memory) graphics backend.
//
// This backend runs with no graphics device at all. Buffers and images are
// backed by ordinary host memory, while every GPU specific object (shader,
// program, command context, acceleration structure, window) is an inert stub so
// that code written against the Core interface keeps compiling and running
// without touching a device. It is meant for CPU only consumers such as a CPU
// path tracing backend.
//
// Unlike the Vulkan/Metal/D3D12 backends, this backend is kept in a single
// host_backend.h + host_backend.cpp pair: apart from Buffer and Image, every
// class below is a handful of no-op lines, and one file pair per class would
// only add noise.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "grassland/graphics/interface.h"
#include "grassland/util/log.h"

namespace grassland::graphics::backend {

class HostCore;

// A buffer backed by a plain std::vector<uint8_t>.
class HostBuffer : public Buffer {
 public:
  HostBuffer(size_t size, BufferType type);

  BufferType Type() const override {
    return type_;
  }

  size_t Size() const override {
    return data_.size();
  }

  void Resize(size_t new_size) override;

  void UploadData(const void *data, size_t size, size_t offset = 0) override;

  void DownloadData(void *data, size_t size, size_t offset = 0) override;

  // Direct access to the backing storage, for hosts that render on the CPU.
  uint8_t *Data() {
    return data_.data();
  }
  const uint8_t *Data() const {
    return data_.data();
  }

 private:
  BufferType type_;
  std::vector<uint8_t> data_;
};

// An image backed by a plain std::vector<uint8_t> of tightly packed pixels.
class HostImage : public Image {
 public:
  HostImage(int width, int height, ImageFormat format);

  Extent2D Extent() const override {
    return extent_;
  }

  ImageFormat Format() const override {
    return format_;
  }

  void UploadData(const void *data) const override {
    UploadData(data, {0, 0}, extent_);
  }

  void DownloadData(void *data) const override {
    DownloadData(data, {0, 0}, extent_);
  }

  // Partial transfers use tightly packed rows of `extent` pixels, matching the
  // Vulkan and Metal backends.
  void UploadData(const void *data, const Offset2D &offset, const Extent2D &extent) const override;

  void DownloadData(void *data, const Offset2D &offset, const Extent2D &extent) const override;

  // Direct access to the backing storage, for hosts that render on the CPU.
  uint8_t *Data() {
    return data_.data();
  }
  const uint8_t *Data() const {
    return data_.data();
  }

 private:
  // Pixel writes go through the const Image interface, so the backing storage
  // itself has to be mutable.
  Extent2D extent_;
  ImageFormat format_;
  size_t bytes_per_pixel_ = 0;
  mutable std::vector<uint8_t> data_;
};

// Samplers carry no state on the host, they exist to keep call sites valid.
class HostSampler : public Sampler {
 public:
  HostSampler(const SamplerInfo &) {
  }
};

// Shaders are never compiled on the host; only the entry point is remembered.
class HostShader : public Shader {
 public:
  HostShader(std::string source_code, std::string entry_point)
      : source_code_(std::move(source_code)), entry_point_(std::move(entry_point)) {
  }

  std::string EntryPoint() const override {
    return entry_point_;
  }

  const std::string &SourceCode() const {
    return source_code_;
  }

 private:
  std::string source_code_;
  std::string entry_point_;
};

// Programs record nothing: there is no pipeline to build on the host.
class HostProgram : public Program {
 public:
  HostProgram(const std::vector<ImageFormat> &, ImageFormat) {
  }

  void AddInputBinding(uint32_t, bool = false) override {
  }
  void AddInputAttribute(uint32_t, InputType, uint32_t) override {
  }
  void AddResourceBinding(ResourceType, int) override {
  }
  void SetCullMode(CullMode) override {
  }
  void SetBlendState(int, const BlendState &) override {
  }
  void BindShader(Shader *, ShaderType) override {
  }
  void Finalize() override {
  }
};

class HostComputeProgram : public ComputeProgram {
 public:
  explicit HostComputeProgram(Shader *) {
  }

  void AddResourceBinding(ResourceType, int) override {
  }
  void Finalize() override {
  }
};

class HostRayTracingProgram : public RayTracingProgram {
 public:
  HostRayTracingProgram() = default;

  void AddResourceBinding(ResourceType, int) override {
  }
  void AddRayGenShader(Shader *) override {
  }
  void AddMissShader(Shader *) override {
  }
  void AddHitGroup(HitGroup) override {
  }
  void AddCallableShader(Shader *) override {
  }
  void Finalize(const std::vector<int32_t> &, const std::vector<int32_t> &, const std::vector<int32_t> &) override {
  }
  void Finalize() override {
  }
};

// Every command is a no-op: there is no GPU queue to record into.
class HostCommandContext : public CommandContext {
 public:
  explicit HostCommandContext(HostCore *core) : core_(core) {
  }

  Core *GetCore() const override;

  void CmdBindProgram(Program *) override {
  }
  void CmdBindRayTracingProgram(RayTracingProgram *) override {
  }
  void CmdBindComputeProgram(ComputeProgram *) override {
  }

  void CmdBindVertexBuffers(uint32_t, const std::vector<Buffer *> &, const std::vector<uint64_t> &) override {
  }
  void CmdBindIndexBuffer(Buffer *, uint64_t) override {
  }
  void CmdBindResources(int, const std::vector<BufferRange> &, BindPoint = BIND_POINT_GRAPHICS) override {
  }
  using CommandContext::CmdBindResources;
  void CmdBindResources(int, const std::vector<Image *> &, BindPoint = BIND_POINT_GRAPHICS) override {
  }
  void CmdBindResources(int, const std::vector<Sampler *> &, BindPoint = BIND_POINT_GRAPHICS) override {
  }
  void CmdBindResources(int, AccelerationStructure *, BindPoint = BIND_POINT_RAYTRACING) override {
  }

  void CmdBeginRendering(const std::vector<Image *> &, Image *) override {
  }
  void CmdEndRendering() override {
  }

  void CmdSetViewport(const Viewport &) override {
  }
  void CmdSetScissor(const Scissor &) override {
  }
  void CmdSetPrimitiveTopology(PrimitiveTopology) override {
  }
  void CmdDraw(uint32_t, uint32_t, int32_t, uint32_t) override {
  }
  void CmdDrawIndexed(uint32_t, uint32_t, uint32_t, int32_t, uint32_t) override {
  }
  void CmdClearImage(Image *, const ClearValue &) override {
  }
  void CmdPresent(Window *, Image *) override {
  }

  void CmdDispatchRays(uint32_t, uint32_t, uint32_t) override {
  }
  void CmdDispatch(uint32_t, uint32_t, uint32_t) override {
  }
  void CmdCopyBuffer(Buffer *, Buffer *, uint64_t, uint64_t = 0, uint64_t = 0) override {
  }

  // The host has no queue to defer work to, so HostCore::SubmitCommandContext
  // runs the queued callbacks right away and then drops them.
  void RunPostExecutionCallbacks();

 private:
  HostCore *core_;
};

// Acceleration structures are unsupported on the host; this class only exists so
// that the pure virtuals of AccelerationStructure can be implemented.
class HostAccelerationStructure : public AccelerationStructure {
 public:
  HostAccelerationStructure() = default;

  int UpdateInstances(const std::vector<RayTracingInstance> &instances) override;
};

class HostCore : public Core {
 public:
  explicit HostCore(const Settings &settings);

  BackendAPI API() const override {
    return BACKEND_API_HOST;
  }

  int CreateBuffer(size_t size, BufferType type, double_ptr<Buffer> pp_buffer) override;

  int CreateImage(int width, int height, ImageFormat format, double_ptr<Image> pp_image) override;

  int CreateSampler(const SamplerInfo &info, double_ptr<Sampler> pp_sampler) override;

  int CreateWindowObject(int width,
                         int height,
                         const std::string &title,
                         bool fullscreen,
                         bool resizable,
                         double_ptr<Window> pp_window) override;

  int CreateShader(const std::string &source_code,
                   const std::string &entry_point,
                   const std::string &target,
                   double_ptr<Shader> pp_shader) override;

  int CreateShader(const VirtualFileSystem &vfs,
                   const std::string &source_file,
                   const std::string &entry_point,
                   const std::string &target,
                   double_ptr<Shader> pp_shader) override;

  int CreateShader(const VirtualFileSystem &vfs,
                   const std::string &source_file,
                   const std::string &entry_point,
                   const std::string &target,
                   const std::vector<std::string> &args,
                   double_ptr<Shader> pp_shader) override;

  int CreateProgram(const std::vector<ImageFormat> &color_formats,
                    ImageFormat depth_format,
                    double_ptr<Program> pp_program) override;

  int CreateComputeProgram(Shader *compute_shader, double_ptr<ComputeProgram> pp_program) override;

  int CreateCommandContext(double_ptr<CommandContext> pp_command_context) override;

  int CreateBottomLevelAccelerationStructure(BufferRange aabb_buffer,
                                             uint32_t stride,
                                             uint32_t num_aabb,
                                             RayTracingGeometryFlag flags,
                                             double_ptr<AccelerationStructure> pp_blas) override;

  int CreateBottomLevelAccelerationStructure(BufferRange vertex_buffer,
                                             BufferRange index_buffer,
                                             uint32_t num_vertex,
                                             uint32_t stride,
                                             uint32_t num_primitive,
                                             RayTracingGeometryFlag flags,
                                             double_ptr<AccelerationStructure> pp_blas) override;

  int CreateBottomLevelAccelerationStructure(Buffer *vertex_buffer,
                                             Buffer *index_buffer,
                                             uint32_t stride,
                                             double_ptr<AccelerationStructure> pp_blas) override;

  int CreateTopLevelAccelerationStructure(const std::vector<RayTracingInstance> &instances,
                                          double_ptr<AccelerationStructure> pp_tlas) override;

  int CreateRayTracingProgram(double_ptr<RayTracingProgram> pp_program) override;

  int SubmitCommandContext(CommandContext *p_command_context) override;

  int GetPhysicalDeviceProperties(PhysicalDeviceProperties *p_physical_device_properties = nullptr) override;

  int InitializeLogicalDevice(int device_index) override;

  void WaitGPU() override {
  }

  // The host executes one thread, which is the smallest possible wave.
  uint32_t WaveSize() const override {
    return 1;
  }

  uint32_t CurrentFrame() const override {
    return 0;
  }

  bool DeviceRayQuerySupport() const override {
    return false;
  }

#if defined(LONGMARCH_CUDA_RUNTIME)
  int CreateCUDABuffer(size_t size, double_ptr<CUDABuffer> pp_buffer) override;
  void CUDABeginExecutionBarrier(cudaStream_t = 0) override {
  }
  void CUDAEndExecutionBarrier(cudaStream_t = 0) override {
  }
#endif

  // Name reported for the single synthetic device of this backend.
  static constexpr const char *kDeviceName = "LongMarch host device";
};

}  // namespace grassland::graphics::backend
