#include "grassland/graphics/backend/host/host_backend.h"

#include <cstring>

namespace grassland::graphics::backend {

namespace {

// Partial image transfers use tightly packed rows of the requested extent,
// matching the Vulkan and Metal backends, so the caller's pitch is always
// `extent.width * bytes_per_pixel`.
bool CheckImageRegion(Extent2D full, Offset2D offset, Extent2D extent) {
  if (offset.x < 0 || offset.y < 0 || static_cast<uint64_t>(offset.x) + extent.width > full.width ||
      static_cast<uint64_t>(offset.y) + extent.height > full.height) {
    LogError("[graphics] Host image region ({}, {}) {}x{} is out of bounds of a {}x{} image", offset.x, offset.y,
             extent.width, extent.height, full.width, full.height);
    return false;
  }
  return true;
}

}  // namespace

HostBuffer::HostBuffer(size_t size, BufferType type) : type_(type), data_(size) {
}

void HostBuffer::Resize(size_t new_size) {
  data_.resize(new_size);
}

void HostBuffer::UploadData(const void *data, size_t size, size_t offset) {
  if (offset > data_.size() || size > data_.size() - offset) {
    LogError("[graphics] Host buffer upload out of range: {} bytes at offset {} of a {} byte buffer", size, offset,
             data_.size());
    return;
  }
  if (size == 0) {
    return;
  }
  std::memcpy(data_.data() + offset, data, size);
}

void HostBuffer::DownloadData(void *data, size_t size, size_t offset) {
  if (offset > data_.size() || size > data_.size() - offset) {
    LogError("[graphics] Host buffer download out of range: {} bytes at offset {} of a {} byte buffer", size, offset,
             data_.size());
    return;
  }
  if (size == 0) {
    return;
  }
  std::memcpy(data, data_.data() + offset, size);
}

HostImage::HostImage(int width, int height, ImageFormat format)
    : extent_{static_cast<uint32_t>(width), static_cast<uint32_t>(height)},
      format_(format),
      bytes_per_pixel_(PixelSize(format)),
      data_(static_cast<size_t>(width) * static_cast<size_t>(height) * PixelSize(format)) {
}

void HostImage::UploadData(const void *data, const Offset2D &offset, const Extent2D &extent) const {
  if (bytes_per_pixel_ == 0) {
    LogError("[graphics] Unsupported host image format {}", static_cast<int>(format_));
    return;
  }
  if (!CheckImageRegion(extent_, offset, extent) || extent.width == 0 || extent.height == 0) {
    return;
  }
  const size_t row_bytes = static_cast<size_t>(extent.width) * bytes_per_pixel_;
  const auto *source = static_cast<const uint8_t *>(data);
  for (uint32_t y = 0; y < extent.height; ++y) {
    const size_t destination_row = static_cast<size_t>(offset.y) + y;
    const size_t destination_column = static_cast<size_t>(offset.x);
    std::memcpy(data_.data() + (destination_row * extent_.width + destination_column) * bytes_per_pixel_,
                source + y * row_bytes, row_bytes);
  }
}

void HostImage::DownloadData(void *data, const Offset2D &offset, const Extent2D &extent) const {
  if (bytes_per_pixel_ == 0) {
    LogError("[graphics] Unsupported host image format {}", static_cast<int>(format_));
    return;
  }
  if (!CheckImageRegion(extent_, offset, extent) || extent.width == 0 || extent.height == 0) {
    return;
  }
  const size_t row_bytes = static_cast<size_t>(extent.width) * bytes_per_pixel_;
  auto *destination = static_cast<uint8_t *>(data);
  for (uint32_t y = 0; y < extent.height; ++y) {
    const size_t source_row = static_cast<size_t>(offset.y) + y;
    const size_t source_column = static_cast<size_t>(offset.x);
    std::memcpy(destination + y * row_bytes,
                data_.data() + (source_row * extent_.width + source_column) * bytes_per_pixel_, row_bytes);
  }
}

Core *HostCommandContext::GetCore() const {
  return core_;
}

void HostCommandContext::RunPostExecutionCallbacks() {
  auto callbacks = std::move(post_execution_callbacks_);
  post_execution_callbacks_.clear();
  for (auto &callback : callbacks) {
    if (callback) {
      callback();
    }
  }
}

int HostAccelerationStructure::UpdateInstances(const std::vector<RayTracingInstance> &) {
  LogError("[graphics] Acceleration structures are not supported by the host backend");
  return -1;
}

HostCore::HostCore(const Settings &settings) : Core(settings) {
}

int HostCore::CreateBuffer(size_t size, BufferType type, double_ptr<Buffer> pp_buffer) {
  pp_buffer.construct<HostBuffer>(size, type);
  return 0;
}

int HostCore::CreateImage(int width, int height, ImageFormat format, double_ptr<Image> pp_image) {
  if (width <= 0 || height <= 0) {
    LogError("[graphics] Invalid host image extent {}x{}", width, height);
    return -1;
  }
  if (PixelSize(format) == 0) {
    LogError("[graphics] Unsupported host image format {}", static_cast<int>(format));
    return -1;
  }
  pp_image.construct<HostImage>(width, height, format);
  return 0;
}

int HostCore::CreateSampler(const SamplerInfo &info, double_ptr<Sampler> pp_sampler) {
  pp_sampler.construct<HostSampler>(info);
  return 0;
}

int HostCore::CreateWindowObject(int,
                                 int,
                                 const std::string &,
                                 bool,
                                 bool,
                                 double_ptr<Window>) {
  LogError("[graphics] The host backend has no window support");
  return -1;
}

int HostCore::CreateShader(const std::string &source_code,
                           const std::string &entry_point,
                           const std::string &,
                           double_ptr<Shader> pp_shader) {
  pp_shader.construct<HostShader>(source_code, entry_point);
  return 0;
}

int HostCore::CreateShader(const VirtualFileSystem &vfs,
                           const std::string &source_file,
                           const std::string &entry_point,
                           const std::string &target,
                           double_ptr<Shader> pp_shader) {
  return CreateShader(vfs, source_file, entry_point, target, {}, pp_shader);
}

int HostCore::CreateShader(const VirtualFileSystem &vfs,
                           const std::string &source_file,
                           const std::string &entry_point,
                           const std::string &,
                           const std::vector<std::string> &,
                           double_ptr<Shader> pp_shader) {
  // Shaders are never compiled on the host. The source is kept when the virtual
  // file system happens to contain it, which costs nothing and helps debugging.
  std::vector<uint8_t> source;
  vfs.ReadFile(source_file, source);
  pp_shader.construct<HostShader>(std::string(source.begin(), source.end()), entry_point);
  return 0;
}

int HostCore::CreateProgram(const std::vector<ImageFormat> &color_formats,
                            ImageFormat depth_format,
                            double_ptr<Program> pp_program) {
  pp_program.construct<HostProgram>(color_formats, depth_format);
  return 0;
}

int HostCore::CreateComputeProgram(Shader *compute_shader, double_ptr<ComputeProgram> pp_program) {
  pp_program.construct<HostComputeProgram>(compute_shader);
  return 0;
}

int HostCore::CreateCommandContext(double_ptr<CommandContext> pp_command_context) {
  pp_command_context.construct<HostCommandContext>(this);
  return 0;
}

int HostCore::CreateBottomLevelAccelerationStructure(BufferRange,
                                                     uint32_t,
                                                     uint32_t,
                                                     RayTracingGeometryFlag,
                                                     double_ptr<AccelerationStructure>) {
  LogError("[graphics] Acceleration structures are not supported by the host backend");
  return -1;
}

int HostCore::CreateBottomLevelAccelerationStructure(BufferRange,
                                                     BufferRange,
                                                     uint32_t,
                                                     uint32_t,
                                                     uint32_t,
                                                     RayTracingGeometryFlag,
                                                     double_ptr<AccelerationStructure>) {
  LogError("[graphics] Acceleration structures are not supported by the host backend");
  return -1;
}

int HostCore::CreateBottomLevelAccelerationStructure(Buffer *,
                                                     Buffer *,
                                                     uint32_t,
                                                     double_ptr<AccelerationStructure>) {
  LogError("[graphics] Acceleration structures are not supported by the host backend");
  return -1;
}

int HostCore::CreateTopLevelAccelerationStructure(const std::vector<RayTracingInstance> &,
                                                  double_ptr<AccelerationStructure>) {
  LogError("[graphics] Acceleration structures are not supported by the host backend");
  return -1;
}

int HostCore::CreateRayTracingProgram(double_ptr<RayTracingProgram>) {
  LogError("[graphics] Ray tracing programs are not supported by the host backend");
  return -1;
}

int HostCore::SubmitCommandContext(CommandContext *p_command_context) {
  auto *host_context = dynamic_cast<HostCommandContext *>(p_command_context);
  if (!host_context || host_context->GetCore() != this) {
    LogError("[graphics] Command context does not belong to this host core");
    return -1;
  }
  // There is no GPU work to wait for, so post-execution callbacks run right away.
  host_context->RunPostExecutionCallbacks();
  return 0;
}

int HostCore::GetPhysicalDeviceProperties(PhysicalDeviceProperties *p_physical_device_properties) {
  // The host backend always exposes exactly one synthetic device.
  if (p_physical_device_properties) {
    PhysicalDeviceProperties properties{};
    properties.name = kDeviceName;
    properties.score = 1;
    properties.ray_tracing_support = false;
    properties.geometry_shader_support = false;
#if defined(LONGMARCH_CUDA_RUNTIME)
    properties.cuda_device_index = -1;
#endif
    p_physical_device_properties[0] = properties;
  }
  return 1;
}

int HostCore::InitializeLogicalDevice(int device_index) {
  if (device_index != 0) {
    LogError("[graphics] Host backend has a single device, requested index {}", device_index);
    return -1;
  }
  device_name_ = kDeviceName;
  ray_tracing_support_ = false;
  return 0;
}

#if defined(LONGMARCH_CUDA_RUNTIME)
int HostCore::CreateCUDABuffer(size_t, double_ptr<CUDABuffer>) {
  LogError("[graphics] CUDA buffers are not supported by the host backend");
  return -1;
}
#endif

}  // namespace grassland::graphics::backend
