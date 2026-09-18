// CUDA backend for the portable Sparkium renderer. The transpiled kernel
// source (identical to the CPU build) is compiled with nvcc at first use,
// loaded through the CUDA driver, and executed with one thread per pixel.

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

// hlsl_compat.h must come before the CUDA headers: it defines
// sparkium_portable::float2/3/4, which collide with the global vector types
// from cuda_runtime.h wherever unqualified names and `using namespace
// sparkium_portable` mix. All host-side references to the compat types below
// are fully qualified (sparkium_portable::...) for the same reason.
#include "sparkium/pipelines/portable/hlsl_compat.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "sparkium/pipelines/portable/frame_resources.h"
#include "sparkium/pipelines/portable/kernel_runtime.h"

namespace sparkium::portable {
namespace {

#define SPARKIUM_CU_CHECK(expr)                                                                                  \
  do {                                                                                                           \
    CUresult result_ = (expr);                                                                                   \
    if (result_ != CUDA_SUCCESS) {                                                                               \
      const char *message_ = nullptr;                                                                            \
      cuGetErrorString(result_, &message_);                                                                      \
      throw std::runtime_error(std::string("CUDA driver error: ") + (message_ ? message_ : "?") + " at " #expr); \
    }                                                                                                            \
  } while (0)

#define SPARKIUM_CUDA_CHECK(expr)                                                                            \
  do {                                                                                                       \
    cudaError_t result_ = (expr);                                                                            \
    if (result_ != cudaSuccess)                                                                              \
      throw std::runtime_error(std::string("CUDA runtime error: ") + cudaGetErrorString(result_) + " at " #expr); \
  } while (0)

std::string ShellQuote(const std::string &path) {
  std::string out = "'";
  for (char c : path) {
    if (c == '\'')
      out += "'\\''";
    else
      out += c;
  }
  return out + "'";
}

struct CudaModule {
  CUmodule module{nullptr};
  CUfunction function{nullptr};
  ~CudaModule() {
    if (module)
      cuModuleUnload(module);
  }
};

std::map<std::string, std::weak_ptr<CudaModule>> g_module_cache;
std::mutex g_module_mutex;

std::shared_ptr<CudaModule> CompileModule(const std::string &source, const std::string &cache_directory) {
  const std::string key = HashString(source + "\n;abi=cuda-v1");
  std::lock_guard<std::mutex> lock(g_module_mutex);
  if (auto cached = g_module_cache[key].lock())
    return cached;

  std::filesystem::path dir = cache_directory.empty() ? std::filesystem::temp_directory_path() / "sparkium_portable"
                                                      : std::filesystem::path(cache_directory);
  std::filesystem::create_directories(dir);
  const std::filesystem::path cu = dir / ("kernel_" + key + ".cu");
  const std::filesystem::path cubin = dir / ("kernel_" + key + ".cubin");

  if (!std::filesystem::exists(cubin)) {
    {
      std::ofstream stream(cu);
      if (!stream)
        throw std::runtime_error("portable kernel: cannot write " + cu.string());
      stream << source;
    }
    int major = 8, minor = 6;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
      major = prop.major;
      minor = prop.minor;
    }
  std::ostringstream command;
    const char *nvcc_env = std::getenv("SPARKIUM_PORTABLE_NVCC");
    const std::string nvcc = nvcc_env && *nvcc_env ? nvcc_env : "nvcc";
    const char *include_env = std::getenv("SPARKIUM_PORTABLE_INCLUDE");
    const std::string include_dir =
        include_env && *include_env ? include_env : std::string(LONGMARCH_PORTABLE_INCLUDE_DIR);
    command << nvcc << " -std=c++17 -O3 -cubin -arch=sm_" << major << minor << " -I " << ShellQuote(include_dir)
            << " -x cu " << ShellQuote(cu.string()) << " -o " << ShellQuote(cubin.string()) << " 2>&1";
    std::array<char, 4096> buffer{};
    std::ostringstream output;
    FILE *pipe = popen(command.str().c_str(), "r");
    if (!pipe)
      throw std::runtime_error("portable kernel: failed to invoke nvcc");
    while (fgets(buffer.data(), buffer.size(), pipe))
      output << buffer.data();
    if (pclose(pipe) != 0)
      throw std::runtime_error("portable CUDA kernel compilation failed:\n" + output.str());
  }
  auto result = std::make_shared<CudaModule>();
  SPARKIUM_CU_CHECK(cuModuleLoad(&result->module, cubin.string().c_str()));
  SPARKIUM_CU_CHECK(cuModuleGetFunction(&result->function, result->module, "PortableRenderKernel"));
  g_module_cache[key] = result;
  return result;
}

template <typename T>
struct DeviceBuffer {
  T *ptr{nullptr};
  size_t count{0};
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() {
    if (ptr)
      cudaFree(ptr);
  }
  void Upload(const std::vector<T> &data) {
    Allocate(std::max<size_t>(1, data.size()));
    if (!data.empty())
      SPARKIUM_CUDA_CHECK(cudaMemcpy(ptr, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  void UploadBytes(const std::vector<uint8_t> &data) {
    Allocate(std::max<size_t>(1, data.size()));
    if (!data.empty())
      SPARKIUM_CUDA_CHECK(cudaMemcpy(ptr, data.data(), data.size(), cudaMemcpyHostToDevice));
  }
  void Allocate(size_t n) {
    if (n > count) {
      if (ptr)
        cudaFree(ptr);
      SPARKIUM_CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
      count = n;
    }
  }
};

struct DeviceScene {
  DeviceBuffer<uint8_t> settings;
  DeviceBuffer<uint8_t> camera;
  DeviceBuffer<uint32_t> sobol;
  DeviceBuffer<uint8_t> instance_metadatas;
  DeviceBuffer<uint8_t> light_metadatas;
  DeviceBuffer<uint8_t> light_selector;
  DeviceBuffer<uint8_t> instances;
  DeviceBuffer<uint8_t> nodes;
  DeviceBuffer<sparkium_portable::Texture2D> sdr_textures;
  DeviceBuffer<sparkium_portable::Texture2D> hdr_textures;
  DeviceBuffer<uint8_t> data_blob;               // all data buffers concatenated
  DeviceBuffer<const uint8_t *> data_ptr_table;  // N device pointers then N sizes
  std::vector<std::unique_ptr<DeviceBuffer<uint8_t>>> texture_pixels;
  DeviceBuffer<sparkium_portable::float4> color;
  DeviceBuffer<float> samples;
};

}  // namespace

bool CudaDeviceReady() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    return false;
  static bool initialized = false;
  if (!initialized) {
    if (cuInit(0) != CUDA_SUCCESS)
      return false;
    if (cudaFree(nullptr) != cudaSuccess)
      return false;
    initialized = true;
  }
  return true;
}

void RenderPixelsCuda(const BakeResult &bake,
                      const std::vector<uint32_t> &sobol_table,
                      const std::string &kernel_source,
                      sparkium_portable::float4 *accumulated_color,
                      float *accumulated_samples,
                      uint32_t width,
                      uint32_t height,
                      uint32_t accumulated_sample_base) {
  if (!CudaDeviceReady())
    throw std::runtime_error("no CUDA device available");

  // The kernel source already contains the CUDA entry point
  // (PortableRenderKernel, guarded by __CUDACC__ in kernel_gen.cpp).
  const char *cache_env = std::getenv("SPARKIUM_PORTABLE_CACHE");
  auto module = CompileModule(kernel_source, cache_env ? cache_env : "");

  FrameResources res;
  PopulateContext(bake, sobol_table, nullptr, nullptr, width, height, res);
  auto *settings = const_cast<uint8_t *>(res.ctx.render_settings);
  const int32_t base = int32_t(accumulated_sample_base);
  std::memcpy(settings + 28, &base, 4);

  static DeviceScene device_state;
  DeviceScene &device = device_state;
  device.settings.UploadBytes(bake.render_settings);
  device.camera.UploadBytes(bake.camera_data);
  device.sobol.Upload(res.sobol_bits);
  device.instance_metadatas.UploadBytes(res.instance_metadatas);
  device.light_metadatas.UploadBytes(res.light_metadatas);
  device.light_selector.UploadBytes(res.light_selector);
  device.instances.UploadBytes(res.instances);
  device.nodes.UploadBytes(
      std::vector<uint8_t>(reinterpret_cast<const uint8_t *>(bake.nodes.data()),
                           reinterpret_cast<const uint8_t *>(bake.nodes.data()) +
                               bake.nodes.size() * sizeof(Node)));
  device.sdr_textures.Upload(res.sdr_textures);
  device.hdr_textures.Upload(res.hdr_textures);

  // Data buffers: concatenate on device and fix up the pointer table.
  const size_t buffer_count = res.ctx.data_buffer_count;
  std::vector<size_t> offsets(buffer_count + 1, 0);
  const size_t *sizes_host = reinterpret_cast<const size_t *>(res.table.data() + buffer_count);
  for (size_t i = 0; i < buffer_count; ++i)
    offsets[i + 1] = offsets[i] + ((sizes_host[i] + 255) / 256) * 256;
  device.data_blob.Allocate(std::max<size_t>(256, offsets.back()));
  std::vector<const uint8_t *> device_ptrs(buffer_count);
  for (size_t i = 0; i < buffer_count; ++i) {
    device_ptrs[i] = device.data_blob.ptr + offsets[i];
    if (sizes_host[i])
      SPARKIUM_CUDA_CHECK(
          cudaMemcpy(device.data_blob.ptr + offsets[i], res.table[i], sizes_host[i], cudaMemcpyHostToDevice));
  }
  device.data_ptr_table.Allocate(std::max<size_t>(1, buffer_count) * 2);
  if (buffer_count) {
    SPARKIUM_CUDA_CHECK(cudaMemcpy(device.data_ptr_table.ptr, device_ptrs.data(),
                                   buffer_count * sizeof(const uint8_t *), cudaMemcpyHostToDevice));
    SPARKIUM_CUDA_CHECK(cudaMemcpy(device.data_ptr_table.ptr + buffer_count, sizes_host,
                                   buffer_count * sizeof(size_t), cudaMemcpyHostToDevice));
  }

  // Textures: upload pixel data and patch the device Texture2D structs.
  auto upload_textures = [&device](const std::vector<TextureData> &textures,
                                   DeviceBuffer<sparkium_portable::Texture2D> &table) {
    if (textures.empty())
      return;
    std::vector<sparkium_portable::Texture2D> host(textures.size());
    for (size_t i = 0; i < textures.size(); ++i) {
      auto pixels = std::make_unique<DeviceBuffer<uint8_t>>();
      pixels->UploadBytes(textures[i].pixels);
      host[i] = textures[i].texture;
      host[i].data = pixels->ptr;
      device.texture_pixels.push_back(std::move(pixels));
    }
    table.Upload(host);
  };
  device.texture_pixels.clear();
  upload_textures(bake.sdr_textures, device.sdr_textures);
  upload_textures(bake.hdr_textures, device.hdr_textures);

  const size_t pixel_count = size_t(width) * height;
  device.color.Allocate(pixel_count);
  device.samples.Allocate(pixel_count);
  SPARKIUM_CUDA_CHECK(
      cudaMemcpy(device.color.ptr, accumulated_color, pixel_count * sizeof(sparkium_portable::float4), cudaMemcpyHostToDevice));
  SPARKIUM_CUDA_CHECK(
      cudaMemcpy(device.samples.ptr, accumulated_samples, pixel_count * sizeof(float), cudaMemcpyHostToDevice));

  // Fill the device context (mirrors FrameResources with device pointers).
  sparkium_portable::KernelContext ctx{};
  ctx.accumulated_color = device.color.ptr;
  ctx.accumulated_samples = device.samples.ptr;
  ctx.image_width = width;
  ctx.image_height = height;
  ctx.render_settings = device.settings.ptr;
  ctx.sobol_table = device.sobol.ptr;
  ctx.camera_data = device.camera.ptr;
  ctx.data_buffers =
      buffer_count ? reinterpret_cast<const uint8_t *const *>(device.data_ptr_table.ptr) : nullptr;
  ctx.data_buffer_count = static_cast<uint32_t>(buffer_count);
  ctx.instance_metadatas = device.instance_metadatas.ptr;
  ctx.light_selector_data = device.light_selector.ptr;
  ctx.light_metadatas = device.light_metadatas.ptr;
  ctx.software_instances = device.instances.ptr;
  ctx.software_nodes = device.nodes.ptr;
  ctx.sdr_textures = device.sdr_textures.ptr;
  ctx.sdr_texture_count = static_cast<uint32_t>(res.sdr_textures.size());
  ctx.hdr_textures = device.hdr_textures.ptr;
  ctx.hdr_texture_count = static_cast<uint32_t>(res.hdr_textures.size());

  CUdeviceptr constant_ptr = 0;
  size_t constant_size = 0;
  SPARKIUM_CU_CHECK(
      cuModuleGetGlobal(&constant_ptr, &constant_size, module->module, "_ZN17sparkium_portable5g_ctxE"));
  SPARKIUM_CU_CHECK(cuMemcpyHtoD(constant_ptr, &ctx, sizeof(ctx)));

  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  void *args[] = {};
  SPARKIUM_CU_CHECK(
      cuLaunchKernel(module->function, grid.x, grid.y, 1, block.x, block.y, 1, 0, nullptr, args, nullptr));
  SPARKIUM_CU_CHECK(cuCtxSynchronize());

  SPARKIUM_CUDA_CHECK(
      cudaMemcpy(accumulated_color, device.color.ptr, pixel_count * sizeof(sparkium_portable::float4), cudaMemcpyDeviceToHost));
  SPARKIUM_CUDA_CHECK(
      cudaMemcpy(accumulated_samples, device.samples.ptr, pixel_count * sizeof(float), cudaMemcpyDeviceToHost));
}

}  // namespace sparkium::portable
