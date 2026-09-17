#include <algorithm>
#include <cmath>
#include <limits>

#include "grassland/graphics/command_context.h"
#include "native_internal.h"

namespace grassland::graphics::backend {
NativeMemory::NativeMemory(bool cuda, size_t size) : cuda_(cuda), size_(size) {
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
  if (cuda_) {
    CUdeviceptr ptr;
    CheckCUDA(cuMemAlloc(&ptr, std::max(size, size_t(16))));
    data_ = reinterpret_cast<void *>(ptr);
    CheckCUDA(cuMemsetD8(ptr, 0, std::max(size, size_t(16))));
    return;
  }
#endif
  if (cuda_)
    NativeUnsupported();
  data_ = ::operator new(std::max(size, size_t(16)));
  std::memset(data_, 0, std::max(size, size_t(16)));
}
NativeMemory::~NativeMemory() {
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
  if (cuda_) {
    cuMemFree(reinterpret_cast<CUdeviceptr>(data_));
    return;
  }
#endif
  ::operator delete(data_);
}
void NativeMemory::Upload(const void *p, size_t size, size_t offset) const {
  if (offset > size_ || size > size_ - offset)
    throw std::out_of_range("native buffer upload");
  if (!size)
    return;
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
  if (cuda_) {
    CheckCUDA(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(data_) + offset, p, size));
    return;
  }
#endif
  std::memcpy(static_cast<char *>(data_) + offset, p, size);
}
void NativeMemory::Download(void *p, size_t size, size_t offset) const {
  if (offset > size_ || size > size_ - offset)
    throw std::out_of_range("native buffer download");
  if (!size)
    return;
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
  if (cuda_) {
    CheckCUDA(cuMemcpyDtoH(p, reinterpret_cast<CUdeviceptr>(data_) + offset, size));
    return;
  }
#endif
  std::memcpy(p, static_cast<char *>(data_) + offset, size);
}
NativeBuffer::NativeBuffer(bool cuda, size_t size, BufferType type) : cuda_(cuda), type_(type) {
  memory = std::make_unique<NativeMemory>(cuda, size);
}
void NativeBuffer::Resize(size_t size) {
  auto next = std::make_unique<NativeMemory>(cuda_, size);
  std::vector<uint8_t> data(std::min(size, Size()));
  memory->Download(data.data(), data.size());
  next->Upload(data.data(), data.size());
  memory = std::move(next);
}
NativeImage::NativeImage(bool cuda, int width, int height, ImageFormat format)
    : extent_{uint32_t(width), uint32_t(height)}, format_(format), unorm_(false) {
  if (width <= 0 || height <= 0)
    throw std::invalid_argument("native image dimensions must be positive");
  switch (format) {
    case IMAGE_FORMAT_R8G8B8A8_UNORM:
    case IMAGE_FORMAT_B8G8R8A8_UNORM:
      channels_ = 4;
      unorm_ = true;
      break;
    case IMAGE_FORMAT_R32G32B32A32_SFLOAT:
      channels_ = 4;
      break;
    case IMAGE_FORMAT_R32G32B32_SFLOAT:
      channels_ = 3;
      break;
    case IMAGE_FORMAT_R32G32_SFLOAT:
      channels_ = 2;
      break;
    case IMAGE_FORMAT_R32_SFLOAT:
    case IMAGE_FORMAT_D32_SFLOAT:
    case IMAGE_FORMAT_R32_UINT:
    case IMAGE_FORMAT_R32_SINT:
      channels_ = 1;
      break;
    default:
      throw std::invalid_argument("unsupported native image format");
  }
  external_bytes_ = channels_ * (unorm_ ? 1 : 4);
  if (uint64_t(width) * height > std::numeric_limits<size_t>::max() / (channels_ * 4))
    throw std::overflow_error("native image size overflow");
  // Keep SDR texels packed. Expanding all imported textures to float4 wastes
  // four times their memory and prevents large pinned scenes fitting on CPU.
  memory = std::make_unique<NativeMemory>(cuda, size_t(width) * height * external_bytes_);
}
namespace {
void CheckRegion(Extent2D full, Offset2D o, Extent2D e) {
  if (o.x < 0 || o.y < 0 || uint64_t(o.x) + e.width > full.width || uint64_t(o.y) + e.height > full.height)
    throw std::out_of_range("native image region");
}
}  // namespace
void NativeImage::UploadData(const void *data, const Offset2D &o, const Extent2D &e) const {
  CheckRegion(extent_, o, e);
  std::vector<uint8_t> row(size_t(e.width) * external_bytes_);
  for (uint32_t y = 0; y < e.height; ++y) {
    const auto *src = static_cast<const uint8_t *>(data) + size_t(y) * e.width * external_bytes_;
    std::memcpy(row.data(), src, row.size());
    if (format_ == IMAGE_FORMAT_B8G8R8A8_UNORM)
      for (uint32_t x = 0; x < e.width; ++x)
        std::swap(row[x * 4], row[x * 4 + 2]);
    memory->Upload(row.data(), row.size(), (size_t(y + o.y) * extent_.width + o.x) * external_bytes_);
  }
}
void NativeImage::DownloadData(void *data, const Offset2D &o, const Extent2D &e) const {
  CheckRegion(extent_, o, e);
  std::vector<uint8_t> row(size_t(e.width) * external_bytes_);
  for (uint32_t y = 0; y < e.height; ++y) {
    memory->Download(row.data(), row.size(), (size_t(y + o.y) * extent_.width + o.x) * external_bytes_);
    auto *dst = static_cast<uint8_t *>(data) + size_t(y) * e.width * external_bytes_;
    if (format_ == IMAGE_FORMAT_B8G8R8A8_UNORM)
      for (uint32_t x = 0; x < e.width; ++x)
        std::swap(row[x * 4], row[x * 4 + 2]);
    std::memcpy(dst, row.data(), row.size());
  }
}
void NativeImage::Clear(const ClearValue &value) {
  if (unorm_) {
    const float c[]{value.color.r, value.color.g, value.color.b, value.color.a};
    std::vector<uint8_t> pixels(size_t(extent_.width) * extent_.height * 4);
    for (size_t i = 0; i < pixels.size(); ++i)
      pixels[i] = uint8_t(std::nearbyint(std::clamp(c[i % 4], 0.0f, 1.0f) * 255.0f));
    memory->Upload(pixels.data(), pixels.size());
    return;
  }
  std::vector<float> pixels(size_t(extent_.width) * extent_.height * channels_);
  const float c[4] = {value.color.r, value.color.g, value.color.b, value.color.a};
  for (size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = c[i % channels_];
  memory->Upload(pixels.data(), pixels.size() * 4);
}
class NativeCommands final : public CommandContext {
 public:
  explicit NativeCommands(NativeCore *core) : core_(core) {
  }
  Core *GetCore() const override {
    return core_;
  }
  // Like the graphics backends, retain compatible resource bindings across
  // program switches. The shared hierarchical prefix scan relies on this.
  void CmdBindComputeProgram(ComputeProgram *p) override {
    program_ = dynamic_cast<NativeProgram *>(p);
    if (!program_)
      throw std::runtime_error("foreign native compute program");
  }
  void CmdBindResources(int s, const std::vector<BufferRange> &v, BindPoint b) override {
    Check(b);
    ClearSlot(s);
    bindings_.buffers[s] = v;
  }
  void CmdBindResources(int s, const std::vector<Image *> &v, BindPoint b) override {
    Check(b);
    ClearSlot(s);
    bindings_.images[s] = v;
  }
  void CmdBindResources(int s, const std::vector<Sampler *> &v, BindPoint b) override {
    Check(b);
    ClearSlot(s);
    bindings_.samplers[s] = v;
  }
  void CmdDispatch(uint32_t x, uint32_t y, uint32_t z) override {
    if (!program_)
      throw std::runtime_error("native dispatch without program");
    auto bindings = bindings_;
    auto *shader = program_->shader;
    commands.emplace_back([=] { shader->Dispatch(bindings, x, y, z); });
  }
  void CmdClearImage(Image *image, const ClearValue &value) override {
    auto *native = dynamic_cast<NativeImage *>(image);
    if (!native)
      throw std::runtime_error("foreign native image");
    commands.emplace_back([=] { native->Clear(value); });
  }
  void CmdCopyBuffer(Buffer *d, Buffer *s, uint64_t n, uint64_t od, uint64_t os) override {
    commands.emplace_back([=] {
      std::vector<uint8_t> bytes(n);
      s->DownloadData(bytes.data(), n, os);
      d->UploadData(bytes.data(), n, od);
    });
  }
  void CmdBindProgram(Program *) override {
    NativeUnsupported();
  }
  void CmdBindRayTracingProgram(RayTracingProgram *) override {
    NativeUnsupported();
  }
  void CmdBindVertexBuffers(uint32_t, const std::vector<Buffer *> &, const std::vector<uint64_t> &) override {
    NativeUnsupported();
  }
  void CmdBindIndexBuffer(Buffer *, uint64_t) override {
    NativeUnsupported();
  }
  void CmdBindResources(int, AccelerationStructure *, BindPoint) override {
    NativeUnsupported();
  }
  void CmdBeginRendering(const std::vector<Image *> &, Image *) override {
    NativeUnsupported();
  }
  void CmdEndRendering() override {
    NativeUnsupported();
  }
  void CmdSetViewport(const Viewport &) override {
    NativeUnsupported();
  }
  void CmdSetScissor(const Scissor &) override {
    NativeUnsupported();
  }
  void CmdSetPrimitiveTopology(PrimitiveTopology) override {
    NativeUnsupported();
  }
  void CmdDraw(uint32_t, uint32_t, int32_t, uint32_t) override {
    NativeUnsupported();
  }
  void CmdDrawIndexed(uint32_t, uint32_t, uint32_t, int32_t, uint32_t) override {
    NativeUnsupported();
  }
  void CmdPresent(Window *, Image *) override {
    NativeUnsupported();
  }
  void CmdDispatchRays(uint32_t, uint32_t, uint32_t) override {
    NativeUnsupported();
  }
  std::vector<std::function<void()>> commands;

 private:
  void Check(BindPoint b) {
    if (b != BIND_POINT_COMPUTE)
      NativeUnsupported();
  }
  void ClearSlot(int s) {
    bindings_.buffers.erase(s);
    bindings_.images.erase(s);
    bindings_.samplers.erase(s);
  }
  NativeCore *core_;
  NativeProgram *program_{};
  NativeBindings bindings_;
};
NativeCore::NativeCore(BackendAPI api, const Settings &settings) : Core(settings), api_(api) {
}
NativeCore::~NativeCore() {
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
  if (cuda_context_) {
    cuCtxSynchronize();
    cuDevicePrimaryCtxRelease(device_index_);
  }
#endif
}
int NativeCore::GetPhysicalDeviceProperties(PhysicalDeviceProperties *p) {
  if (api_ == BACKEND_API_CPU) {
    if (p) {
      p[0].name = "Native CPU (Slang C++)";
      p[0].score = 1;
      p[0].ray_tracing_support = false;
      p[0].geometry_shader_support = false;
    }
    return 1;
  }
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
  CheckCUDA(cuInit(0));
  int count = 0;
  CheckCUDA(cuDeviceGetCount(&count));
  if (p)
    for (int i = 0; i < count; ++i) {
      char name[256];
      CheckCUDA(cuDeviceGetName(name, sizeof(name), i));
      p[i].name = name;
      p[i].score = 1;
      p[i].ray_tracing_support = false;
      p[i].geometry_shader_support = false;
      p[i].cuda_device_index = i;
    }
  return count;
#else
  return 0;
#endif
}
int NativeCore::InitializeLogicalDevice(int index) {
  int count = GetPhysicalDeviceProperties();
  if (index < 0 || index >= count)
    return -1;
  std::vector<PhysicalDeviceProperties> properties(count);
  GetPhysicalDeviceProperties(properties.data());
  device_name_ = properties[index].name;
  device_index_ = index;
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
  if (api_ == BACKEND_API_CUDA) {
    CUcontext context;
    CheckCUDA(cuDevicePrimaryCtxRetain(&context, index));
    cuda_context_ = context;
    cuda_device_ = index;
    CheckCUDA(cuCtxSetCurrent(context));
  }
#endif
  return 0;
}
void NativeCore::WaitGPU() {
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
  if (api_ == BACKEND_API_CUDA)
    CheckCUDA(cuCtxSynchronize());
#endif
}
int NativeCore::CreateBuffer(size_t s, BufferType t, double_ptr<Buffer> p) {
  p.construct<NativeBuffer>(api_ == BACKEND_API_CUDA, s, t);
  return 0;
}
int NativeCore::CreateImage(int w, int h, ImageFormat f, double_ptr<Image> p) {
  p.construct<NativeImage>(api_ == BACKEND_API_CUDA, w, h, f);
  return 0;
}
int NativeCore::CreateSampler(const SamplerInfo &i, double_ptr<Sampler> p) {
  p.construct<NativeSampler>(i);
  return 0;
}
int NativeCore::CreateShader(const std::string &s, const std::string &e, const std::string &t, double_ptr<Shader> p) {
  VirtualFileSystem v;
  v.WriteFile("input.hlsl", s);
  return CreateShader(v, "input.hlsl", e, t, {}, p);
}
int NativeCore::CreateShader(const VirtualFileSystem &v,
                             const std::string &s,
                             const std::string &e,
                             const std::string &t,
                             double_ptr<Shader> p) {
  return CreateShader(v, s, e, t, {}, p);
}
int NativeCore::CreateShader(const VirtualFileSystem &v,
                             const std::string &s,
                             const std::string &e,
                             const std::string &t,
                             const std::vector<std::string> &a,
                             double_ptr<Shader> p) {
  if (t.rfind("cs_", 0) != 0)
    NativeUnsupported();
  p.construct<NativeShader>(api_ == BACKEND_API_CUDA, v, s, e, a);
  return 0;
}
int NativeCore::CreateComputeProgram(Shader *s, double_ptr<ComputeProgram> p) {
  auto *n = dynamic_cast<NativeShader *>(s);
  if (!n)
    throw std::runtime_error("foreign native shader");
  p.construct<NativeProgram>(n);
  return 0;
}
int NativeCore::CreateCommandContext(double_ptr<CommandContext> p) {
  p.construct<NativeCommands>(this);
  return 0;
}
int NativeCore::SubmitCommandContext(CommandContext *p) {
  auto *n = dynamic_cast<NativeCommands *>(p);
  if (!n)
    throw std::runtime_error("foreign native command context");
  for (auto &f : n->commands)
    f();
  WaitGPU();
  for (auto &f : n->GetPostExecutionCallbacks())
    f();
  return 0;
}
int NativeCore::CreateWindowObject(int, int, const std::string &, bool, bool, double_ptr<Window>) {
  NativeUnsupported();
}
int NativeCore::CreateProgram(const std::vector<ImageFormat> &, ImageFormat, double_ptr<Program>) {
  NativeUnsupported();
}
int NativeCore::CreateBottomLevelAccelerationStructure(BufferRange,
                                                       uint32_t,
                                                       uint32_t,
                                                       RayTracingGeometryFlag,
                                                       double_ptr<AccelerationStructure>) {
  NativeUnsupported();
}
int NativeCore::CreateBottomLevelAccelerationStructure(BufferRange,
                                                       BufferRange,
                                                       uint32_t,
                                                       uint32_t,
                                                       uint32_t,
                                                       RayTracingGeometryFlag,
                                                       double_ptr<AccelerationStructure>) {
  NativeUnsupported();
}
int NativeCore::CreateBottomLevelAccelerationStructure(Buffer *,
                                                       Buffer *,
                                                       uint32_t,
                                                       double_ptr<AccelerationStructure>) {
  NativeUnsupported();
}
int NativeCore::CreateTopLevelAccelerationStructure(const std::vector<RayTracingInstance> &,
                                                    double_ptr<AccelerationStructure>) {
  NativeUnsupported();
}
int NativeCore::CreateRayTracingProgram(double_ptr<RayTracingProgram>) {
  NativeUnsupported();
}
#if defined(LONGMARCH_CUDA_RUNTIME)
int NativeCore::CreateCUDABuffer(size_t, double_ptr<CUDABuffer>) {
  NativeUnsupported();
}
void NativeCore::CUDABeginExecutionBarrier(cudaStream_t) {
  NativeUnsupported();
}
void NativeCore::CUDAEndExecutionBarrier(cudaStream_t) {
  NativeUnsupported();
}
#endif
}  // namespace grassland::graphics::backend
