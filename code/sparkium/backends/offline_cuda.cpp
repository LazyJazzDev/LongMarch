#include "sparkium/backends/offline_cuda.h"

#if defined(LONGMARCH_CUDA_ENABLED)

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "sparkium/backends/offline_cuda_kernels.h"

namespace sparkium::backends {

namespace {

void CheckCuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(status));
}

// A device allocation that is grown on demand: arrays whose contents change
// every frame (the film) and arrays uploaded once per scene share the same
// helper.
struct DeviceArray {
  void *pointer{nullptr};
  size_t bytes{0};

  void Ensure(size_t required) {
    if (bytes >= required && pointer)
      return;
    Release();
    if (required == 0)
      return;
    // Keep a small slack so growth of a few hundred bytes does not reallocate.
    size_t rounded = required + (required / 8) + 4096;
    CheckCuda(cudaMalloc(&pointer, rounded), "cudaMalloc");
    bytes = rounded;
  }

  void Release() {
    if (pointer)
      CheckCuda(cudaFree(pointer), "cudaFree");
    pointer = nullptr;
    bytes = 0;
  }

  template <typename T>
  void Upload(const std::vector<T> &data, const char *what) {
    if (data.empty()) {
      Release();
      return;
    }
    Ensure(data.size() * sizeof(T));
    CheckCuda(cudaMemcpy(pointer, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice), what);
  }

  template <typename T>
  T *As() {
    return static_cast<T *>(pointer);
  }

  template <typename T>
  const T *As() const {
    return static_cast<const T *>(pointer);
  }
};

}  // namespace

struct CudaOfflineBackend::Impl {
  DeviceArray mesh_data, meshes, mesh_nodes, instance_nodes, instances, materials, lights, light_power_cdf,
      primitive_power_cdf, textures, texture_pixels, sobol_table;
  DeviceArray device_scene;
  DeviceArray color, samples;
  // Tone mapping runs on the device too; the buffers are reused between
  // Develop calls and are only touched after path tracing is done.
  mutable DeviceArray resolve_color, resolve_samples, resolve_rgba8;
  mutable std::vector<uint8_t> host_rgba8;
  size_t pixel_count{0};
  bool scene_uploaded{false};
  const OfflineScene *uploaded_scene{nullptr};
  const uint32_t *uploaded_sobol{nullptr};
  std::string device_description;

  void ReleaseAll() {
    mesh_data.Release();
    meshes.Release();
    mesh_nodes.Release();
    instance_nodes.Release();
    instances.Release();
    materials.Release();
    lights.Release();
    light_power_cdf.Release();
    primitive_power_cdf.Release();
    textures.Release();
    texture_pixels.Release();
    sobol_table.Release();
    device_scene.Release();
    color.Release();
    samples.Release();
    resolve_color.Release();
    resolve_samples.Release();
    resolve_rgba8.Release();
    pixel_count = 0;
    scene_uploaded = false;
    uploaded_scene = nullptr;
    uploaded_sobol = nullptr;
  }
};

CudaOfflineBackend::CudaOfflineBackend() : impl_(std::make_unique<Impl>()) {
  int device = 0;
  CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp properties{};
  CheckCuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
  impl_->device_description = std::string(properties.name) + " (sm_" + std::to_string(properties.major) +
                               std::to_string(properties.minor) + ", " +
                               std::to_string(properties.multiProcessorCount) + " SM)";
}

CudaOfflineBackend::~CudaOfflineBackend() {
  try {
    impl_->ReleaseAll();
  } catch (...) {
    // Destructors must not throw; a failing free leaves the context teardown to
    // the driver.
  }
}

void CudaOfflineBackend::Sync(const OfflineScene &scene) {
  Impl &impl = *impl_;
  const DeviceScene &device = scene.Device();
  if (!impl.scene_uploaded || impl.uploaded_scene != &scene) {
    impl.mesh_data.Upload(scene.MeshData(), "upload mesh_data");
    impl.meshes.Upload(scene.Meshes(), "upload meshes");
    impl.mesh_nodes.Upload(scene.MeshNodes(), "upload mesh_nodes");
    impl.instance_nodes.Upload(scene.InstanceNodes(), "upload instance_nodes");
    impl.instances.Upload(scene.Instances(), "upload instances");
    impl.materials.Upload(scene.Materials(), "upload materials");
    impl.lights.Upload(scene.Lights(), "upload lights");
    impl.light_power_cdf.Upload(scene.LightPowerCdf(), "upload light_power_cdf");
    impl.primitive_power_cdf.Upload(scene.PrimitivePowerCdf(), "upload primitive_power_cdf");
    impl.textures.Upload(scene.Textures(), "upload textures");
    impl.texture_pixels.Upload(scene.TexturePixels(), "upload texture_pixels");
    impl.scene_uploaded = true;
    impl.uploaded_scene = &scene;
    impl.uploaded_sobol = nullptr;
  }
  if (impl.uploaded_sobol != device.sobol_table) {
    impl.sobol_table.Upload(scene.SobolTable(), "upload sobol_table");
    impl.uploaded_sobol = device.sobol_table;
  }

  DeviceScene mirror = device;
  mirror.mesh_data = impl.mesh_data.As<const uint8_t>();
  mirror.meshes = impl.meshes.As<const MeshRange>();
  mirror.mesh_nodes = impl.mesh_nodes.As<const SoftwareNode>();
  mirror.instance_nodes = impl.instance_nodes.As<const SoftwareNode>();
  mirror.instances = impl.instances.As<const InstanceData>();
  mirror.materials = impl.materials.As<const MaterialData>();
  mirror.lights = impl.lights.As<const LightData>();
  mirror.light_power_cdf = impl.light_power_cdf.As<const float>();
  mirror.primitive_power_cdf = impl.primitive_power_cdf.As<const float>();
  mirror.textures = impl.textures.As<const TextureData>();
  mirror.texture_pixels = impl.texture_pixels.As<const float>();
  mirror.sobol_table = impl.sobol_table.As<const uint32_t>();
  impl.device_scene.Ensure(sizeof(DeviceScene));
  CheckCuda(cudaMemcpy(impl.device_scene.pointer, &mirror, sizeof(DeviceScene), cudaMemcpyHostToDevice),
            "upload device_scene");
}

void CudaOfflineBackend::Dispatch(const OfflineScene &scene, const RenderSettings &settings, OfflineFilm &film) {
  Impl &impl = *impl_;
  if (!impl.scene_uploaded || impl.uploaded_scene != &scene ||
      impl.uploaded_sobol != scene.Device().sobol_table)
    Sync(scene);

  const size_t pixels = film.PixelCount();
  if (pixels == 0)
    return;
  if (impl.pixel_count != pixels) {
    impl.color.Ensure(pixels * sizeof(float4));
    impl.samples.Ensure(pixels * sizeof(float));
    impl.pixel_count = pixels;
  }
  CheckCuda(cudaMemcpy(impl.color.pointer, film.Color(), pixels * sizeof(float4), cudaMemcpyHostToDevice),
            "upload film color");
  CheckCuda(cudaMemcpy(impl.samples.pointer, film.Samples(), pixels * sizeof(float), cudaMemcpyHostToDevice),
            "upload film samples");

  SparkiumOfflineCudaPathTrace(impl.device_scene.pointer, &settings, impl.color.pointer, impl.samples.pointer,
                               film.Width(), film.Height());
  // Dispatch is synchronous from the caller's point of view: the accumulated
  // film must be readable right after it returns.
  SparkiumOfflineCudaSynchronize();
  CheckCuda(cudaMemcpy(film.Color(), impl.color.pointer, pixels * sizeof(float4), cudaMemcpyDeviceToHost),
            "download film color");
  CheckCuda(cudaMemcpy(film.Samples(), impl.samples.pointer, pixels * sizeof(float), cudaMemcpyDeviceToHost),
            "download film samples");
}

std::vector<uint8_t> CudaOfflineBackend::Develop(const OfflineFilm &film,
                                                const ToneMappingSettings &settings) const {
  Impl &impl = *impl_;
  const size_t pixels = film.PixelCount();
  if (pixels == 0)
    return {};
  impl.resolve_color.Ensure(pixels * sizeof(float4));
  impl.resolve_samples.Ensure(pixels * sizeof(float));
  impl.resolve_rgba8.Ensure(pixels * 4);
  CheckCuda(cudaMemcpy(impl.resolve_color.pointer, film.Color(), pixels * sizeof(float4), cudaMemcpyHostToDevice),
            "upload resolve color");
  CheckCuda(
      cudaMemcpy(impl.resolve_samples.pointer, film.Samples(), pixels * sizeof(float), cudaMemcpyHostToDevice),
      "upload resolve samples");
  SparkiumOfflineCudaToneMap(impl.resolve_color.pointer, impl.resolve_samples.pointer, impl.resolve_rgba8.pointer,
                            &settings, film.Width(), film.Height());
  SparkiumOfflineCudaSynchronize();
  impl.host_rgba8.resize(pixels * 4);
  CheckCuda(cudaMemcpy(impl.host_rgba8.data(), impl.resolve_rgba8.pointer, pixels * 4, cudaMemcpyDeviceToHost),
            "download rgba8");
  return impl.host_rgba8;
}

std::string CudaOfflineBackend::DeviceDescription() const {
  return impl_->device_description;
}

size_t CudaOfflineBackend::DeviceMemoryBytes() const {
  const Impl &impl = *impl_;
  auto sum = [](const DeviceArray &array) { return array.pointer ? array.bytes : 0; };
  return sum(impl.mesh_data) + sum(impl.meshes) + sum(impl.mesh_nodes) + sum(impl.instance_nodes) +
         sum(impl.instances) + sum(impl.materials) + sum(impl.lights) + sum(impl.light_power_cdf) +
         sum(impl.primitive_power_cdf) + sum(impl.textures) + sum(impl.texture_pixels) + sum(impl.sobol_table) +
         sum(impl.device_scene) + sum(impl.color) + sum(impl.samples) + sum(impl.resolve_color) +
         sum(impl.resolve_samples) + sum(impl.resolve_rgba8);
}

std::unique_ptr<OfflineBackend> CreateCudaBackend() {
  return std::make_unique<CudaOfflineBackend>();
}

}  // namespace sparkium::backends

#endif  // LONGMARCH_CUDA_ENABLED
