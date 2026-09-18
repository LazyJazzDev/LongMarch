// Host interface to the CPU path tracer's compiled shaders.
//
// Everything the shaders need is passed in as plain pointers, so this header is
// the only place that has to stay free of the HLSL compatibility layer. The
// pipeline that fills it in never includes a shader.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sparkium::raytracing::cpu {

// A byte addressed host buffer, matching what ByteAddressBuffer sees.
struct BufferView {
  const uint8_t *data{nullptr};
  size_t size{0};
};

// A host image in the layout SampleTexture expects. Pixels are tightly packed
// RGBA floats regardless of the source format, which is what the pipeline
// converts every registered image to.
struct TextureView {
  float *pixels{nullptr};
  uint32_t width{0};
  uint32_t height{0};
  uint32_t channels{4};
};

// Mirrors RenderSettings in common.hlsli; the shaders read it as a constant
// buffer. The fields are spelled out rather than reusing the shader's struct so
// that this interface stays independent of the compatibility layer.
struct RenderSettingsView {
  int32_t samples_per_dispatch{1};
  int32_t max_bounces{4};
  int32_t alpha_shadow{0};
  float background_color[3]{0.0f, 0.0f, 0.0f};
  int32_t accumulated_samples{0};
  float persistence{1.0f};
  float clamping{100.0f};
  float max_exposure{1.0f};
  int32_t view_transform{0};
  float exposure{0.0f};
  float gamma{1.0f};
  float contrast{1.0f};
};

// One frame's worth of bindings. The data buffer vector must hold the scene's
// registered buffers followed by the six fixed slots the shader macros index
// past software_data_buffer_count (sobol table, camera, instance metadata,
// light selector, light metadata, software instances).
struct FrameBindings {
  BufferView nodes;
  std::vector<BufferView> data_buffers;
  std::vector<TextureView> sdr_textures;
  std::vector<TextureView> hdr_textures;

  // Film accumulation. accumulated_color holds RGBA and accumulated_samples a
  // single channel per pixel; `width`/`height` describe both.
  TextureView accumulated_color;
  TextureView accumulated_samples;
  uint32_t width{0};
  uint32_t height{0};

  RenderSettingsView settings;

  // Maps a compacted material index to MaterialKernel. Filled in by the
  // pipeline after compacting materials by sampler source, exactly as
  // SoftwarePipeline does.
  std::vector<uint32_t> material_kernels;
};

// Binds `bindings` for the following calls. The shaders read these as globals;
// there is no descriptor set to record into.
void BindFrame(const FrameBindings &bindings);

// Traces every pixel of the frame `bindings` was bound with, over
// `thread_count` worker threads. Each call dispatches one sample per pixel and
// folds it into the film, matching RenderPixel's accumulation rules.
void RenderFrame(const FrameBindings &bindings, uint32_t thread_count);

// Runs one of the BVH build passes over the currently bound nodes/keys/instances
// buffers. `pass` selects the kernel and `first`/`count`/`stage`/`stride` are
// the same parameters software/build.hlsl takes.
enum class BuildPass { InitLeaves, ReduceNodes, MortonKeys, BitonicSort, SortLeaves };

// Which engine evaluates a shader graph's generated source. Both are provided
// by the cpu sublibrary and produce identical surfaces; see graph_program.h.
enum class GraphEngine {
  Interpreter,
  Jit,
};

// Registers the generated source of one shader-graph material. `source` is what
// ShaderGraphCompiler produced, exactly as the GPU backend writes it into
// software_materials.hlsli. Registration only records it; FinalizeGraphMaterials
// compiles the whole set, which lets the JIT share one module between them.
void RegisterGraphMaterial(uint32_t material_data_index, const std::string &source);

// Compiles every registered graph with `engine` and installs the textures its
// SampleTexture reads. Returns false if any material could not be compiled, in
// which case the interpreter is tried instead.
bool FinalizeGraphMaterials(GraphEngine engine,
                            const std::vector<TextureView> &sdr_textures,
                            const std::vector<TextureView> &hdr_textures);

// Drops every registered graph; called when the material set changes.
void ClearGraphMaterials();

// Mirrors ToneMappingSettings in tone_mapping.hlsl.
struct ToneMappingSettingsView {
  int32_t view_transform{0};
  float exposure{0.0f};
  float gamma{1.0f};
  float contrast{1.0f};
};

// Develops an HDR image into an 8-bit target, i.e. tone_mapping.hlsl without a
// device. `raw` holds RGBA floats, the target is written as UNORM8 pixels.
void ToneMap(const TextureView &raw, uint8_t *target_pixels, const ToneMappingSettingsView &settings);

// Per-primitive emitted power, used to build a mesh light's sampling CDF. This
// is the host equivalent of gather_primitive_power.hlsl and reuses the same
// GEOMETRY_SAMPLER / MaterialEvaluator shader code.
float PrimitivePower(uint32_t material_kernel,
                     BufferView geometry,
                     BufferView material,
                     const float transform[12],
                     uint32_t primitive_index);

// Number of material kernels the shader knows about; also the range of valid
// values for the material_kernels table.
enum MaterialKernel : uint32_t {
  kMaterialKernelLambertian = 0,
  kMaterialKernelLight = 1,
  kMaterialKernelPrincipled = 2,
  kMaterialKernelSpecular = 3,
  kMaterialKernelShaderGraph = 4,
};

}  // namespace sparkium::raytracing::cpu
