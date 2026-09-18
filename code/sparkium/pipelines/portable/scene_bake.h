#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "sparkium/core/core_util.h"
#include "sparkium/pipelines/portable/hlsl_compat.h"
#include "sparkium/scene_io/json_scene.h"

namespace sparkium::portable {

// Host-side mirror of shaders/software/layout.hlsli SoftwareNode.
struct Node {
  float lo[3];
  uint32_t first;
  float hi[3];
  uint32_t second;
};
static_assert(sizeof(Node) == 32, "software node layout changed");

// Host-side mirror of SoftwareInstance.
struct Instance {
  float object_to_world[12];  // float3x4: column-major, four columns of three floats
  float world_to_object[12];
  uint32_t root, geometry, material, primitive_count;
};
static_assert(sizeof(Instance) == 112, "software instance layout changed");

struct GpuLightMetadata {
  int32_t sampler_shader_index;
  int32_t sampler_data_index;
  int32_t custom_index;
  uint32_t power_offset;
};
static_assert(sizeof(GpuLightMetadata) == 16, "light metadata layout changed");

struct GpuInstanceMetadata {
  int32_t geometry_data_index;
  int32_t material_data_index;
  int32_t custom_index;
};
static_assert(sizeof(GpuInstanceMetadata) == 12, "instance metadata layout changed");

struct GeometryData {
  std::vector<uint8_t> buffer;
  uint32_t root{0};
  uint32_t leaves{0};
  uint32_t count{0};
  uint32_t buffer_index{0};
};

struct MaterialData {
  bool shader_graph{false};
  bool shadow_any_hit{false};     // SAMPLE_SHADOW_ANY_HIT in the sampler
  bool shadow_no_hitrecord{false};  // SAMPLE_SHADOW_NO_HITRECORD
  std::string source;  // transpiled C++ sampler body
  std::vector<uint8_t> buffer;
};

// One entry per material object: the serialized GPU buffer contents plus the
// index of the deduplicated shader variant in BakeResult::materials.
struct MaterialBufferData {
  uint32_t shader_index{0};
  std::vector<uint8_t> buffer;
};

struct LightData {
  GpuLightMetadata metadata{};
  std::vector<uint8_t> buffer;  // sampler data (point light params or mesh CDF)
};

struct TextureData {
  sparkium_portable::Texture2D texture{};
  std::vector<uint8_t> pixels;
};

// Fully baked scene consumed by the portable kernels. All indices match the
// semantics of the GPU compute renderer (data buffer index tables,
// 0x10000xx light sampler IDs, software node layout).
struct BakeResult {
  std::vector<Node> nodes;
  std::vector<Instance> instances;
  std::vector<GeometryData> geometries;
  std::vector<MaterialData> materials;
  std::vector<MaterialBufferData> material_buffers;
  std::vector<LightData> lights;
  std::vector<GpuInstanceMetadata> instance_metadatas;
  std::vector<float> light_power_cdf;
  std::vector<TextureData> sdr_textures;
  std::vector<TextureData> hdr_textures;
  std::vector<uint8_t> camera_data;     // HLSL CameraData layout (160 bytes)
  std::vector<uint8_t> render_settings; // HLSL RenderSettings layout (64 bytes)
};

// Builds a BakeResult from a loaded sparkium scene. Shared by the CPU and
// CUDA backends; the BVH construction mirrors shaders/software/build.hlsl.
BakeResult BakeScene(sparkium::Scene *scene,
                     sparkium::Camera *camera,
                     sparkium::Film *film,
                     const std::map<graphics::Image *, const HostImageData *> &host_images,
                     uint32_t *seed);

using RenderPixelFn = void (*)(uint32_t, uint32_t);

// Renders baked pixels with a compiled kernel entry point (host backend).
void RenderPixelsHostPrepared(const BakeResult &bake,
                              const std::vector<uint32_t> &sobol_table,
                              RenderPixelFn entry,
                              void *context_slot,
                              sparkium_portable::float4 *accumulated_color,
                              float *accumulated_samples,
                              uint32_t width,
                              uint32_t height,
                              uint32_t accumulated_sample_base,
                              uint32_t threads);

#ifdef SPARKIUM_PORTABLE_CUDA
bool CudaDeviceReady();
void RenderPixelsCuda(const BakeResult &bake,
                      const std::vector<uint32_t> &sobol_table,
                      const std::string &kernel_source,
                      sparkium_portable::float4 *accumulated_color,
                      float *accumulated_samples,
                      uint32_t width,
                      uint32_t height,
                      uint32_t accumulated_sample_base);
#endif

// Applies the film tone mapping (shaders/tone_mapping.hlsl semantics) and
// writes 8-bit RGBA.
void ToneMapPixels(const sparkium_portable::float4 *averaged_color,
                   int view_transform,
                   float exposure,
                   float gamma,
                   float contrast,
                   uint32_t width,
                   uint32_t height,
                   uint8_t *rgba);

}  // namespace sparkium::portable
