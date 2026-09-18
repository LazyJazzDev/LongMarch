#include "sparkium/pipelines/native/cuda/cuda_renderer.h"

#include <cuda_runtime.h>

#include <map>
#include <stdexcept>
#include <string>

namespace sparkium::native {

namespace {

void CheckCuda(cudaError_t error, const char *what) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("CUDA ") + what + " failed: " + cudaGetErrorString(error));
}

// A device allocation that only grows, so a steady-state animation reuses it.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() {
    if (data_)
      cudaFree(data_);
  }

  void Reserve(size_t bytes) {
    if (bytes <= capacity_)
      return;
    if (data_)
      CheckCuda(cudaFree(data_), "cudaFree");
    CheckCuda(cudaMalloc(&data_, bytes), "cudaMalloc");
    capacity_ = bytes;
  }

  template <typename T>
  void Upload(const T *host, size_t count) {
    size_ = count * sizeof(T);
    Reserve(size_ ? size_ : 1);
    if (size_)
      CheckCuda(cudaMemcpy(data_, host, size_, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
  }

  template <typename T>
  void Download(T *host, size_t count) const {
    if (count)
      CheckCuda(cudaMemcpy(host, data_, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  }

  template <typename T>
  T *As() const {
    return static_cast<T *>(data_);
  }

  uint32_t Size() const {
    return static_cast<uint32_t>(size_);
  }

 private:
  void *data_{nullptr};
  size_t capacity_{0};
  size_t size_{0};
};

using DeviceBufferPtr = std::unique_ptr<DeviceBuffer>;

DeviceBufferPtr MakeDeviceBuffer() {
  return std::make_unique<DeviceBuffer>();
}

}  // namespace

struct CudaRenderer::Impl {
  // Blob mirrors keyed by the host object the bytes came from, so an unchanged
  // geometry or material is uploaded once.
  struct CachedBlob {
    uint64_t revision{0};
    DeviceBufferPtr buffer;
  };
  std::map<const void *, CachedBlob> blobs;
  std::map<const void *, CachedBlob> textures;

  DeviceBuffer nodes;
  DeviceBuffer instances;
  DeviceBuffer data_buffer_table;
  DeviceBuffer sobol;
  DeviceBuffer camera;
  DeviceBuffer instance_metadatas;
  DeviceBuffer light_selector;
  DeviceBuffer light_metadatas;
  DeviceBuffer sdr_table;
  DeviceBuffer hdr_table;
  DeviceBuffer materials;
  DeviceBuffer graph_table;
  std::vector<DeviceBufferPtr> graph_payloads;

  DeviceBuffer accumulated_color;
  DeviceBuffer accumulated_samples;

  uint64_t uploaded_geometry_revision{0};
  bool geometry_valid{false};
};

// One thread per pixel, running the very same `RenderPixel` the CPU backend
// calls.
__global__ void RenderKernel(SceneView view, uint2 extent, float4 *accumulated_color, float *accumulated_samples) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= extent.x || y >= extent.y)
    return;
  RenderPixel(view, uint2{x, y}, extent, accumulated_color, accumulated_samples);
}

bool CudaRenderer::Available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

CudaRenderer::CudaRenderer() : impl_(std::make_unique<Impl>()) {
  if (!Available())
    throw std::runtime_error("no CUDA device is available");
}

CudaRenderer::~CudaRenderer() = default;

void CudaRenderer::Render(const SceneData &scene_data,
                          uint2 extent,
                          std::vector<float4> &accumulated_color,
                          std::vector<float> &accumulated_samples) {
  auto &impl = *impl_;
  const SceneView &host_view = scene_data.View();
  SceneView view = host_view;

  auto upload_buffer = [](DeviceBuffer &target, const ByteBuffer &source) {
    target.Upload(source.data, source.size / sizeof(uint32_t));
    ByteBuffer device;
    device.data = target.As<uint32_t>();
    device.size = source.size;
    return device;
  };

  // Geometry, materials and light CDFs: re-upload only what changed.
  std::vector<ByteBuffer> data_buffers;
  data_buffers.reserve(scene_data.DataBuffers().size());
  for (const auto &entry : scene_data.DataBuffers()) {
    auto &cached = impl.blobs[entry.key];
    if (!cached.buffer || cached.revision != entry.revision) {
      if (!cached.buffer)
        cached.buffer = MakeDeviceBuffer();
      cached.buffer->Upload(entry.data->data(), entry.data->size());
      cached.revision = entry.revision;
    }
    ByteBuffer device;
    device.data = cached.buffer->As<uint32_t>();
    device.size = static_cast<uint32_t>(entry.data->size() * sizeof(uint32_t));
    data_buffers.push_back(device);
  }
  impl.data_buffer_table.Upload(data_buffers.data(), data_buffers.size());
  view.data_buffers = impl.data_buffer_table.As<ByteBuffer>();

  auto upload_textures = [&](const std::vector<const HostTexture *> &host_textures, DeviceBuffer &table) {
    std::vector<DeviceTexture> device_textures;
    device_textures.reserve(host_textures.size());
    for (const auto *texture : host_textures) {
      auto &cached = impl.textures[texture];
      if (!cached.buffer || cached.revision != texture->revision) {
        if (!cached.buffer)
          cached.buffer = MakeDeviceBuffer();
        if (texture->sdr.empty())
          cached.buffer->Upload(texture->hdr.data(), texture->hdr.size());
        else
          cached.buffer->Upload(texture->sdr.data(), texture->sdr.size());
        cached.revision = texture->revision;
      }
      DeviceTexture device{};
      if (texture->sdr.empty())
        device.hdr_data = cached.buffer->As<float>();
      else
        device.sdr_data = cached.buffer->As<uint32_t>();
      device.width = texture->width;
      device.height = texture->height;
      device_textures.push_back(device);
    }
    table.Upload(device_textures.data(), device_textures.size());
    return table.As<DeviceTexture>();
  };
  view.sdr_textures = upload_textures(scene_data.SdrTextures(), impl.sdr_table);
  view.hdr_textures = upload_textures(scene_data.HdrTextures(), impl.hdr_table);

  if (!impl.geometry_valid || impl.uploaded_geometry_revision != scene_data.GeometryRevision()) {
    view.software_nodes = upload_buffer(impl.nodes, host_view.software_nodes);
    impl.uploaded_geometry_revision = scene_data.GeometryRevision();
    impl.geometry_valid = true;
  } else {
    view.software_nodes.data = impl.nodes.As<uint32_t>();
  }
  view.software_instances = upload_buffer(impl.instances, host_view.software_instances);
  view.sobol_table = upload_buffer(impl.sobol, host_view.sobol_table);
  view.camera_data = upload_buffer(impl.camera, host_view.camera_data);
  view.instance_metadatas = upload_buffer(impl.instance_metadatas, host_view.instance_metadatas);
  view.light_selector_data = upload_buffer(impl.light_selector, host_view.light_selector_data);
  view.light_metadatas = upload_buffer(impl.light_metadatas, host_view.light_metadatas);

  impl.materials.Upload(scene_data.Materials().data(), scene_data.Materials().size());
  view.materials = impl.materials.As<NativeMaterial>();

  // Shader graphs: the instruction list plus its constant and data pools.
  const auto &graph_programs = scene_data.GraphPrograms();
  impl.graph_payloads.clear();
  std::vector<GraphProgram> device_programs;
  device_programs.reserve(graph_programs.size());
  for (const auto &program : graph_programs) {
    GraphProgram device{};
    impl.graph_payloads.push_back(MakeDeviceBuffer());
    impl.graph_payloads.back()->Upload(program.instructions.data(), program.instructions.size());
    device.instructions = impl.graph_payloads.back()->As<GraphInstruction>();
    device.instruction_count = static_cast<uint32_t>(program.instructions.size());
    impl.graph_payloads.push_back(MakeDeviceBuffer());
    impl.graph_payloads.back()->Upload(program.constants.data(), program.constants.size());
    device.constants = impl.graph_payloads.back()->As<float4>();
    impl.graph_payloads.push_back(MakeDeviceBuffer());
    impl.graph_payloads.back()->Upload(program.data.data(), program.data.size());
    device.data = impl.graph_payloads.back()->As<float>();
    for (int i = 0; i < GRAPH_SURFACE_OUTPUT_COUNT; ++i)
      device.outputs[i] = program.outputs[i];
    device_programs.push_back(device);
  }
  impl.graph_table.Upload(device_programs.data(), device_programs.size());
  view.graph_programs = impl.graph_table.As<GraphProgram>();

  impl.accumulated_color.Upload(accumulated_color.data(), accumulated_color.size());
  impl.accumulated_samples.Upload(accumulated_samples.data(), accumulated_samples.size());

  const dim3 block(8, 8, 1);
  const dim3 grid((extent.x + block.x - 1) / block.x, (extent.y + block.y - 1) / block.y, 1);
  RenderKernel<<<grid, block>>>(view, extent, impl.accumulated_color.As<float4>(),
                                impl.accumulated_samples.As<float>());
  CheckCuda(cudaGetLastError(), "kernel launch");
  CheckCuda(cudaDeviceSynchronize(), "kernel execution");

  impl.accumulated_color.Download(accumulated_color.data(), accumulated_color.size());
  impl.accumulated_samples.Download(accumulated_samples.data(), accumulated_samples.size());
}

}  // namespace sparkium::native
