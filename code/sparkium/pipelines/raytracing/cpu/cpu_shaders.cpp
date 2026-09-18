// The CPU backend's only translation unit that knows about the HLSL sources.
//
// Including the shaders here compiles the path tracer, its traversal, the BSDFs
// and the material samplers into native code. Everything outside this file
// talks to them through cpu_shaders.h, so the compatibility layer's names never
// leak into the rest of the renderer.
#include "sparkium/pipelines/raytracing/cpu/cpu_shaders.h"

#include <algorithm>
#include <cstring>
#include <thread>

#include "grassland/util/log.h"
#include "sparkium/pipelines/raytracing/cpu/graph_program.h"
#include "sparkium/pipelines/raytracing/cpu/shaders/hlsl_cpu_render.h"

namespace sparkium::raytracing::cpu {

using grassland::LogError;

// The registry the compiled shaders call into. It lives here rather than in the
// pipeline so that the shader glue and the registration entry points cannot
// disagree about which one is in use. The shader namespace below reaches it by
// qualified name, which is why it is not in an anonymous namespace.
GraphProgramRegistry g_graph_programs;

// Recorded by RegisterGraphMaterial and compiled by FinalizeGraphMaterials.
std::vector<std::pair<uint32_t, std::string>> pending_graphs_;

void RegisterGraphMaterial(uint32_t material_data_index, const std::string &source) {
  // Recorded now, compiled by FinalizeGraphMaterials: the JIT needs the whole
  // material set to build a single module.
  pending_graphs_.emplace_back(material_data_index, source);
}

bool FinalizeGraphMaterials(GraphEngine engine,
                            const std::vector<TextureView> &sdr_textures,
                            const std::vector<TextureView> &hdr_textures) {
  if (pending_graphs_.empty())
    return true;

  std::vector<std::string> sources;
  sources.reserve(pending_graphs_.size());
  for (const auto &[index, source] : pending_graphs_)
    sources.push_back(source);
  (void)sources;

  if (engine == GraphEngine::Jit) {
    if (!JitGraphProgramsAvailable()) {
      LogError("[sparkium] this build has no CPU shader-graph JIT; falling back to the interpreter");
    } else {
      std::vector<std::string> ordered;
      ordered.reserve(pending_graphs_.size());
      for (const auto &[index, source] : pending_graphs_)
        ordered.push_back(source);
      std::vector<std::unique_ptr<GraphProgram>> programs =
          MakeJitGraphPrograms(ordered, sdr_textures.data(), static_cast<int>(sdr_textures.size()),
                               hdr_textures.data(), static_cast<int>(hdr_textures.size()));
      if (programs.size() == pending_graphs_.size()) {
        for (size_t i = 0; i < programs.size(); ++i)
          g_graph_programs.Register(pending_graphs_[i].first, std::move(programs[i]));
        pending_graphs_.clear();
        return true;
      }
      LogError("[sparkium] the CPU shader-graph JIT failed; using the interpreter instead");
    }
  }

  bool ok = true;
  for (const auto &[index, source] : pending_graphs_) {
    std::unique_ptr<GraphProgram> program = MakeInterpreterGraphProgram(source);
    if (!program) {
      ok = false;
      continue;
    }
    g_graph_programs.Register(index, std::move(program));
  }
  pending_graphs_.clear();
  return ok;
}

void ClearGraphMaterials() {
  pending_graphs_.clear();
  g_graph_programs.Clear();
}

}  // namespace sparkium::raytracing::cpu

// The shader-graph sampler calls this with the same signature the generated
// per-material function has on the GPU. The hit identifies the material, so the
// registry can pick the right compiled graph.
namespace sparkium_cpu_shaders {
namespace {

// SampleTexture, in the shape the graph engines expect.
void SampleTextureForGraph(void *, int32_t texture_index, float u, float v, float out[4]) {
  const Float4 sampled = SampleTexture(texture_index, Float2(u, v));
  for (int i = 0; i < 4; ++i)
    out[i] = sampled[i];
}

void Store(const Float3 &value, float *out) {
  out[0] = value.x;
  out[1] = value.y;
  out[2] = value.z;
}

}  // namespace

GraphSurface EvaluateShaderGraph(HitRecord hit_record,
                                 Float3 view_direction,
                                 int bounce,
                                 int ray_type,
                                 bool is_shadow_ray,
                                 ByteAddressBuffer material_data) {
  GraphSurface surface{};
  InstanceMetadata metadata =
      instance_metadatas.Load<InstanceMetadata>(sizeof(InstanceMetadata) * hit_record.object_index);

  sparkium::raytracing::cpu::GraphEvalInput input;
  Store(hit_record.position, input.position);
  Store(hit_record.object_position, input.object_position);
  Store(hit_record.object_origin, input.object_origin);
  Store(hit_record.normal, input.normal);
  Store(hit_record.geom_normal, input.geom_normal);
  Store(hit_record.tangent, input.tangent);
  Store(hit_record.color, input.color);
  input.tex_coord[0] = hit_record.tex_coord.x;
  input.tex_coord[1] = hit_record.tex_coord.y;
  input.t = hit_record.t;
  input.signal = hit_record.signal;
  input.pdf = hit_record.pdf;
  input.primitive_index = hit_record.primitive_index;
  input.object_index = hit_record.object_index;
  input.front_facing = hit_record.front_facing;
  input.view_direction[0] = view_direction.x;
  input.view_direction[1] = view_direction.y;
  input.view_direction[2] = view_direction.z;
  input.bounce = bounce;
  input.ray_type = ray_type;
  input.is_shadow_ray = is_shadow_ray;
  input.material_data = material_data.Data();
  input.material_data_size = material_data.Size();
  input.sample_texture = &SampleTextureForGraph;

  sparkium::raytracing::cpu::GraphEvalOutput output;
  sparkium::raytracing::cpu::g_graph_programs.Evaluate(metadata.material_data_index, input, output);

  surface.base_color = Float3(output.base_color[0], output.base_color[1], output.base_color[2]);
  surface.metallic = output.metallic;
  surface.specular = output.specular;
  surface.roughness = output.roughness;
  surface.anisotropic = output.anisotropic;
  surface.anisotropic_rotation = output.anisotropic_rotation;
  surface.sheen = output.sheen;
  surface.clearcoat = output.clearcoat;
  surface.clearcoat_roughness = output.clearcoat_roughness;
  surface.ior = output.ior;
  surface.transmission = output.transmission;
  surface.transmission_roughness = output.transmission_roughness;
  surface.emission = Float3(output.emission[0], output.emission[1], output.emission[2]);
  surface.normal = Float3(output.normal[0], output.normal[1], output.normal[2]);
  surface.opacity = output.opacity;
  surface.shadow_opacity = output.shadow_opacity;
  surface.thin_walled = output.thin_walled;
  surface.subsurface = output.subsurface;
  surface.subsurface_scale = output.subsurface_scale;
  surface.subsurface_radius =
      Float3(output.subsurface_radius[0], output.subsurface_radius[1], output.subsurface_radius[2]);
  surface.subsurface_method = output.subsurface_method;
  return surface;
}
}  // namespace sparkium_cpu_shaders

// tone_mapping.hlsl names the same slots the path tracer does, so it is
// compiled in a namespace of its own where its own bindings live.
namespace sparkium_cpu_shaders::ToneMapping {

// The shader declares accumulated_color, output and settings itself; the host
// only binds its own storage to them.
#include "tone_mapping.hlsl"

}  // namespace sparkium_cpu_shaders::ToneMapping

namespace sparkium::raytracing::cpu {
namespace {

using namespace sparkium_cpu_shaders;
using grassland::LogError;

// The per-primitive power of a mesh light comes from the material's evaluator,
// exactly as gather_primitive_power.hlsl computes it. Every evaluator defines a
// class called MaterialEvaluator, so each lives in its own namespace.
// GeometrySampler is shared and guarded by #pragma once, so it is included once
// here rather than inside each of them.
#include "geometry/mesh/geometry_sampler.hlsli"

namespace EvaluateLambertian {
#include "material/lambertian/evaluator.hlsli"
}  // namespace EvaluateLambertian

namespace EvaluateLight {
#include "material/light/evaluator.hlsli"
}  // namespace EvaluateLight

namespace EvaluatePrincipled {
#include "material/principled/evaluator.hlsli"
}  // namespace EvaluatePrincipled

namespace EvaluateSpecular {
#include "material/specular/evaluator.hlsli"
}  // namespace EvaluateSpecular

namespace EvaluateGraph {
#include "material/shader_graph/evaluator.hlsli"
}  // namespace EvaluateGraph

// The shaders treat resources as bound globals; the host side needs somewhere
// for those globals to point. Binding happens once per frame, before the render
// threads start, so one set of storage is enough. The texture vectors are sized
// before any pointer into them is taken, so no view ever dangles.
std::vector<TextureData> g_sdr_texture_data;
std::vector<TextureData> g_hdr_texture_data;
TextureData g_color_target;
TextureData g_sample_target;

// Builds the host view a Texture2D expects from the shim-free description.
TextureData MakeTextureData(const TextureView &view) {
  TextureData data;
  data.pixels = view.pixels;
  data.width = view.width;
  data.height = view.height;
  data.channels = view.channels;
  return data;
}

// Renders a contiguous run of rows. Threads own disjoint pixels, and the only
// shared state the shader touches is the read-only resource set.
void RenderRange(const FrameBindings &bindings, uint32_t first_row, uint32_t row_count) {
  const Uint2 extent(bindings.width, bindings.height);
  for (uint32_t row = first_row; row < first_row + row_count; ++row)
    for (uint32_t column = 0; column < bindings.width; ++column)
      RenderDispatch(Uint2(column, row), extent);
}

}  // namespace

void BindFrame(const FrameBindings &bindings) {
  software_nodes = ByteAddressBuffer(bindings.nodes.data, bindings.nodes.size);

  // The last six entries are the fixed slots the shader's index macros reach
  // past software_data_buffer_count.
  software_data_buffer_count = static_cast<uint32_t>(bindings.data_buffers.size()) - 6;
  data_buffers.resize(bindings.data_buffers.size());
  for (size_t i = 0; i < bindings.data_buffers.size(); ++i)
    data_buffers[i] = ByteAddressBuffer(bindings.data_buffers[i].data, bindings.data_buffers[i].size);

  g_sdr_texture_data.resize(bindings.sdr_textures.size());
  sdr_textures.resize(bindings.sdr_textures.size());
  for (size_t i = 0; i < bindings.sdr_textures.size(); ++i) {
    g_sdr_texture_data[i] = MakeTextureData(bindings.sdr_textures[i]);
    sdr_textures[i] = Texture2D<Float4>(&g_sdr_texture_data[i]);
  }

  g_hdr_texture_data.resize(bindings.hdr_textures.size());
  hdr_textures.resize(bindings.hdr_textures.size());
  for (size_t i = 0; i < bindings.hdr_textures.size(); ++i) {
    g_hdr_texture_data[i] = MakeTextureData(bindings.hdr_textures[i]);
    hdr_textures[i] = Texture2D<Float4>(&g_hdr_texture_data[i]);
  }

  samplers.assign(2, SamplerState{});

  g_color_target = MakeTextureData(bindings.accumulated_color);
  g_sample_target = MakeTextureData(bindings.accumulated_samples);
  accumulated_color = RWTexture2D<Float4>(&g_color_target);
  accumulated_samples = RWTexture2D<float>(&g_sample_target);

  const RenderSettingsView &source = bindings.settings;
  render_settings.samples_per_dispatch = source.samples_per_dispatch;
  render_settings.max_bounces = source.max_bounces;
  render_settings.alpha_shadow = source.alpha_shadow != 0;
  render_settings._settings_padding = 0;
  render_settings.background_color =
      Float3(source.background_color[0], source.background_color[1], source.background_color[2]);
  render_settings.accumulated_samples = source.accumulated_samples;
  render_settings.persistence = source.persistence;
  render_settings.clamping = source.clamping;
  render_settings.max_exposure = source.max_exposure;
  render_settings.view_transform = source.view_transform;
  render_settings.exposure = source.exposure;
  render_settings.gamma = source.gamma;
  render_settings.contrast = source.contrast;

  material_kernels = bindings.material_kernels;
}

void RenderFrame(const FrameBindings &bindings, uint32_t thread_count) {
  thread_count = std::max<uint32_t>(1, std::min<uint32_t>(thread_count, std::max<uint32_t>(1, bindings.height)));
  if (thread_count == 1) {
    RenderRange(bindings, 0, bindings.height);
    return;
  }

  // Rows are split evenly; each worker writes only the pixels it owns, so no
  // synchronisation is needed beyond joining.
  std::vector<std::thread> workers;
  workers.reserve(thread_count);
  const uint32_t rows_per_thread = (bindings.height + thread_count - 1) / thread_count;
  for (uint32_t i = 0; i < thread_count; ++i) {
    const uint32_t first = i * rows_per_thread;
    if (first >= bindings.height)
      break;
    const uint32_t count = std::min(rows_per_thread, bindings.height - first);
    workers.emplace_back([&bindings, first, count]() { RenderRange(bindings, first, count); });
  }
  for (auto &worker : workers)
    worker.join();
}

void ToneMap(const TextureView &raw, uint8_t *target_pixels, const ToneMappingSettingsView &view) {
  TextureData source = MakeTextureData(raw);
  // The shader writes float4, but the target is an 8-bit image, so the mapping
  // runs into a float buffer that is then converted the way UNORM8 uploads are.
  std::vector<float> mapped(static_cast<size_t>(raw.width) * raw.height * 4, 0.0f);
  TextureData destination;
  destination.pixels = mapped.data();
  destination.width = raw.width;
  destination.height = raw.height;
  destination.channels = 4;

  ToneMapping::accumulated_color = Texture2D<Float4>(&source);
  ToneMapping::output = RWTexture2D<Float4>(&destination);
  ToneMapping::settings.view_transform = view.view_transform;
  ToneMapping::settings.exposure = view.exposure;
  ToneMapping::settings.gamma = view.gamma;
  ToneMapping::settings.contrast = view.contrast;


  const Uint2 extent(raw.width, raw.height);
  for (uint32_t y = 0; y < raw.height; ++y)
    for (uint32_t x = 0; x < raw.width; ++x)
      ToneMapping::ToneMapPixel(Uint2(x, y), extent);

  for (size_t i = 0; i < mapped.size(); ++i) {
    const float value = std::min(1.0f, std::max(0.0f, mapped[i]));
    target_pixels[i] = static_cast<uint8_t>(value * 255.0f + 0.5f);
  }
}

float PrimitivePower(uint32_t material_kernel,
                     BufferView geometry,
                     BufferView material,
                     const float transform[12],
                     uint32_t primitive_index) {
  const ByteAddressBuffer geometry_buffer(geometry.data, geometry.size);
  const ByteAddressBuffer material_buffer(material.data, material.size);

  // The transform arrives as the same twelve floats a SoftwareInstance stores,
  // which gather_primitive_power.hlsl reads back through LoadFloat3x4. Going
  // through the same helper keeps the two in step.
  const ByteAddressBuffer transform_buffer(transform, sizeof(float) * 12);
  const Float3x4 object_to_world = LoadFloat3x4(transform_buffer, 0);

  switch (material_kernel) {
    case kMaterialKernelLambertian: {
      GeometrySampler<ByteAddressBuffer> sampler;
      sampler.geometry_data = geometry_buffer;
      sampler.transform = object_to_world;
      EvaluateLambertian::MaterialEvaluator<ByteAddressBuffer> evaluator;
      evaluator.material_data = material_buffer;
      return evaluator.PrimitivePower(sampler, primitive_index);
    }
    case kMaterialKernelLight: {
      GeometrySampler<ByteAddressBuffer> sampler;
      sampler.geometry_data = geometry_buffer;
      sampler.transform = object_to_world;
      EvaluateLight::MaterialEvaluator<ByteAddressBuffer> evaluator;
      evaluator.material_data = material_buffer;
      return evaluator.PrimitivePower(sampler, primitive_index);
    }
    case kMaterialKernelPrincipled: {
      GeometrySampler<ByteAddressBuffer> sampler;
      sampler.geometry_data = geometry_buffer;
      sampler.transform = object_to_world;
      EvaluatePrincipled::MaterialEvaluator<ByteAddressBuffer> evaluator;
      evaluator.material_data = material_buffer;
      return evaluator.PrimitivePower(sampler, primitive_index);
    }
    case kMaterialKernelSpecular: {
      GeometrySampler<ByteAddressBuffer> sampler;
      sampler.geometry_data = geometry_buffer;
      sampler.transform = object_to_world;
      EvaluateSpecular::MaterialEvaluator<ByteAddressBuffer> evaluator;
      evaluator.material_data = material_buffer;
      return evaluator.PrimitivePower(sampler, primitive_index);
    }
    case kMaterialKernelShaderGraph: {
      GeometrySampler<ByteAddressBuffer> sampler;
      sampler.geometry_data = geometry_buffer;
      sampler.transform = object_to_world;
      EvaluateGraph::MaterialEvaluator<ByteAddressBuffer> evaluator;
      evaluator.material_data = material_buffer;
      return evaluator.PrimitivePower(sampler, primitive_index);
    }
    default:
      return 0.0f;
  }
}

}  // namespace sparkium::raytracing::cpu
