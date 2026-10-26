#include "sparkium/pipelines/raytracing/core/software_pipeline.h"

#include <algorithm>
#include <array>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "sparkium/core/frame_profile.h"
#include "sparkium/pipelines/raytracing/core/camera.h"
#include "sparkium/pipelines/raytracing/core/core.h"
#include "sparkium/pipelines/raytracing/core/geometry.h"
#include "sparkium/pipelines/raytracing/core/material.h"

namespace sparkium::raytracing {
namespace {
uint32_t LeafCount(size_t count) {
  // 64 threads per group, at most 65535 groups on baseline Vulkan/D3D12.
  // Rounding to a power of two limits each tree to 2^21 actual leaves.
  if (count > (uint32_t(1) << 21))
    throw std::runtime_error("software BVH exceeds 2097152 leaves per tree");
  uint32_t result = 1;
  while (result < count)
    result *= 2;
  return result;
}

std::string MaterialSource(const CodeLines &source) {
  std::istringstream input(static_cast<std::string>(source));
  std::string line, result;
  while (std::getline(input, line)) {
    auto first = line.find_first_not_of(" \t");
    if (first != std::string::npos &&
        (line.compare(first, 8, "#include") == 0 || line.compare(first, 7, "#pragma") == 0))
      continue;
    result += line + '\n';
  }
  return result;
}

struct GPUInstance {
  glm::mat4x3 object_to_world;
  glm::mat4x3 world_to_object;
  uint32_t root, geometry, material, primitive_count;
};

static_assert(sizeof(GPUInstance) == 112, "Slang software instance layout changed");
}  // namespace

SoftwarePipeline::SoftwarePipeline(Core *core, bool ray_query) : core_(core), ray_query_(ray_query) {
  if (ray_query_ && !core_->GetDevice()->SupportRayQuery())
    throw std::runtime_error("native ray queries are unavailable on the selected backend");
  static_assert(sizeof(BuildParameters) == 256, "uniform buffer alignment");
}

void SoftwarePipeline::ClearInstances() {
  instances_.clear();
}

void SoftwarePipeline::AddInstance(Geometry *geometry,
                                   Material *material,
                                   const glm::mat4x3 &transform,
                                   uint32_t geometry_index) {
  instances_.push_back({geometry, material, transform, geometry_index});
}

void SoftwarePipeline::CompileBuilders(uint32_t buffer_count) {
  CpuProfileScope compile_profile("compile_builders");
  auto device = core_->GetDevice();
  builders_.clear();
  if (builder_shaders_.empty()) {
    for (const char *entry : {"InitLeaves", "ReduceNodes", "MortonKeys", "BitonicSort", "SortLeaves"}) {
      rhi::ObjPtr<rhi::IShader> shader;
      shader = device->CreateShader("software/build.slang", entry, rhi::SHADER_STAGE_COMPUTE, &core_->GetShadersVFS(),
                                    {"."}, {});
      if (!shader)
        throw std::runtime_error(std::string("failed to compile software BVH kernel: ") + entry);
      builder_shaders_.push_back(std::move(shader));
    }
  }
  for (auto &shader : builder_shaders_) {
    rhi::ObjPtr<rhi::IComputeProgram> program;
    {
      rhi::ComputeProgramDesc desc;
      desc.compute_shader = shader.Get();
      desc.pipeline_layout.groups = {{{{0, rhi::BINDING_RW_BYTE_ADDRESS_BUFFER, 1},
                                       {1, rhi::BINDING_BYTE_ADDRESS_BUFFER, static_cast<uint32_t>(buffer_count)},
                                       {2, rhi::BINDING_UNIFORM_BUFFER, 1},
                                       {3, rhi::BINDING_RW_BYTE_ADDRESS_BUFFER, 1},
                                       {4, rhi::BINDING_BYTE_ADDRESS_BUFFER, 1}}}};
      program = device->CreateComputeProgram(desc);
    }
    builders_.push_back(std::move(program));
  }
  build_group_ = core_->GetDevice()->CreateBindingGroup(builders_.front()->GetDesc().pipeline_layout.groups[0]);
  builder_buffer_count_ = buffer_count;
}

void SoftwarePipeline::CompileRenderer(const std::vector<MaterialCode> &materials,
                                       uint32_t buffers,
                                       uint32_t sdr_count,
                                       uint32_t hdr_count,
                                       const std::string &camera_file,
                                       const std::string &camera_entry) {
  CpuProfileScope compile_profile("compile_renderer");
  const bool has_graph = std::any_of(materials.begin(), materials.end(),
                                     [](const MaterialCode &material) { return material.shader_graph; });
  std::ostringstream source;
  for (size_t i = 0; i < materials.size(); ++i) {
    source << "namespace SoftwareMaterial" << i << " {\n" << materials[i].source;
    if (materials[i].shader_graph) {
      source << "}\n";
      continue;
    }
    source << R"(
float Transmission(HitRecord hit, float3 direction) {
#ifdef SAMPLE_SHADOW_ANY_HIT
  return 1.0f - saturate(SampleShadowOpacity(hit, direction));
#else
  ShadowRayPayload payload;
  payload.shadow = 1.0f;
#ifdef SAMPLE_SHADOW_NO_HITRECORD
  SampleShadow(payload);
#else
  SampleShadow(payload, hit);
#endif
  return payload.shadow;
#endif
}
}
#ifdef SAMPLE_SHADOW_ANY_HIT
#undef SAMPLE_SHADOW_ANY_HIT
#endif
#ifdef SAMPLE_SHADOW_NO_HITRECORD
#undef SAMPLE_SHADOW_NO_HITRECORD
#endif
)";
  }
  if (has_graph) {
    source << R"(
  ByteAddressBuffer SoftwareMaterialData(HitRecord hit) {
    InstanceMetadata metadata = instance_metadatas.Load<InstanceMetadata>(sizeof(InstanceMetadata) * hit.object_index);
    return data_buffers[NonUniformResourceIndex(metadata.material_data_index)];
  }
  void SoftwareSampleMaterial(uint material, inout RenderContext context, HitRecord hit) {
    GraphSurface graph;
    switch (material) {
  )";
    for (size_t i = 0; i < materials.size(); ++i) {
      source << "case " << i << ": ";
      if (materials[i].shader_graph)
        source << "graph = SoftwareMaterial" << i
               << "::EvaluateShaderGraph(hit, -context.direction, context.bounce, context.ray_type, false, "
                  "SoftwareMaterialData(hit)); break;\n";
      else
        source << "SoftwareMaterial" << i << "::SampleMaterial(context, hit); return;\n";
    }
    source << R"(
      default: context.throughput = float3(0, 0, 0); return;
    }
    SampleGraphSurface(context, hit, graph);
  }
  float SoftwareShadowTransmission(uint material, HitRecord hit, float3 direction) {
    switch (material) {
  )";
    for (size_t i = 0; i < materials.size(); ++i) {
      source << "case " << i << ": return ";
      if (materials[i].shader_graph)
        source
            << "1.0f - saturate(GraphShadowOpacity(SoftwareMaterial" << i
            << "::EvaluateShaderGraph(hit, -direction, 1, RAY_TYPE_REFLECTION, true, SoftwareMaterialData(hit))));\n";
      else
        source << "SoftwareMaterial" << i << "::Transmission(hit, direction);\n";
    }
    source << "default: return 0.0f;\n}}\n";
  } else {
    // Preserve the compact dispatch for scenes without material graphs.
    source << "void SoftwareSampleMaterial(uint material, inout RenderContext context, HitRecord hit) {\n"
              "switch (material) {\n";
    for (size_t i = 0; i < materials.size(); ++i)
      source << "case " << i << ": SoftwareMaterial" << i << "::SampleMaterial(context, hit); return;\n";
    source
        << "default: context.throughput = float3(0, 0, 0); break;\n}}\n"
           "float SoftwareShadowTransmission(uint material, HitRecord hit, float3 direction) {\nswitch (material) {\n";
    for (size_t i = 0; i < materials.size(); ++i)
      source << "case " << i << ": return SoftwareMaterial" << i << "::Transmission(hit, direction);\n";
    source << "default: return 0.0f;\n}}\n";
  }

  auto vfs = core_->GetShadersVFS();
  vfs.WriteFile("software_materials.slang", source.str());
  vfs.WriteFile("software_camera.slang", "#include \"" + camera_file + "\"\n");
  render_program_.Release();
  std::vector<std::pair<std::string, std::string>> defines{{"SOFTWARE_DATA_BUFFER_COUNT", std::to_string(buffers)}};
  defines.emplace_back("SPARKIUM_CAMERA_ENTRY", camera_entry);
  if (ray_query_)
    defines.emplace_back("SPARKIUM_RAY_QUERY", "1");
  if (has_graph)
    defines.emplace_back("SPARKIUM_SHADER_GRAPHS", "1");
  render_shader_ = core_->GetDevice()->CreateShader("software/render.slang", "Main", rhi::SHADER_STAGE_COMPUTE, &vfs,
                                                    {"."}, defines);
  if (!render_shader_)
    throw std::runtime_error("failed to compile compute ray tracing shader");
  rhi::ComputeProgramDesc desc;
  desc.compute_shader = render_shader_.Get();
  desc.pipeline_layout.groups = {
      {{{0, rhi::BINDING_RW_TEXTURE2D},
        {1, rhi::BINDING_RW_TEXTURE2D},
        {2, ray_query_ ? rhi::BINDING_ACCELERATION_STRUCTURE : rhi::BINDING_BYTE_ADDRESS_BUFFER},
        {3, rhi::BINDING_UNIFORM_BUFFER},
        {4, rhi::BINDING_BYTE_ADDRESS_BUFFER, buffers + 6},
        {5, rhi::BINDING_TEXTURE2D, sdr_count},
        {6, rhi::BINDING_TEXTURE2D, hdr_count},
        {7, rhi::BINDING_SAMPLER, 2}}}};
  render_program_ = core_->GetDevice()->CreateComputeProgram(desc);
  if (!render_program_)
    throw std::runtime_error("failed to create compute path tracing pipeline");
  render_group_ = core_->GetDevice()->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  camera_file_ = camera_file;
  camera_entry_ = camera_entry;
  material_sources_ = materials;
  buffer_count_ = buffers;
  sdr_count_ = sdr_count;
  hdr_count_ = hdr_count;
}

void SoftwarePipeline::AppendBuild(std::vector<BuildPass> &passes, BuildParameters parameters) {
  passes.push_back({0, parameters, parameters.leaves});
  auto reduce = [&]() {
    for (uint32_t count = parameters.leaves / 2; count; count /= 2) {
      parameters.first = count - 1;
      parameters.level_count = count;
      passes.push_back({1, parameters, count});
    }
  };

  reduce();
  if (parameters.leaves > 1) {
    passes.push_back({2, parameters, parameters.leaves});
    for (uint32_t stage = 2; stage <= parameters.leaves; stage *= 2) {
      for (uint32_t stride = stage / 2; stride; stride /= 2) {
        parameters.stage = stage;
        parameters.stride = stride;
        passes.push_back({3, parameters, parameters.leaves});
      }
    }
    passes.push_back({4, parameters, parameters.leaves});
    reduce();
  }
}

void SoftwarePipeline::Update(rhi::ICommandContext *commands,
                              const std::vector<rhi::IBuffer *> &buffers,
                              uint32_t sdr_count,
                              uint32_t hdr_count,
                              Camera *camera) {
  auto device = core_->GetDevice();
  CpuProfileScope prepare_profile("software_prepare");
  const std::string camera_file = camera->ShaderFile();
  const std::string camera_entry = camera->EntryPoint();
  std::vector<GeometryLayout> geometries;
  std::vector<GPUInstance> gpu_instances;
  std::vector<rhi::RayTracingInstance> native_instances;
  std::vector<MaterialCode> materials;
  if (16ull + instances_.size() * sizeof(GPUInstance) > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("software instance byte address overflow");
  const uint32_t tlas_leaves = ray_query_ ? 1 : LeafCount(instances_.size());
  uint64_t node_count = uint64_t(tlas_leaves) * 2 - 1;
  uint32_t max_leaves = tlas_leaves;
  for (const auto &instance : instances_) {
    auto geometry = std::find_if(geometries.begin(), geometries.end(),
                                 [&](const GeometryLayout &g) { return g.geometry == instance.geometry; });
    if (geometry == geometries.end()) {
      const uint32_t count = instance.geometry->PrimitiveCount(), leaves = ray_query_ ? 1 : LeafCount(count);
      if (node_count + uint64_t(leaves) * 2 - 1 > std::numeric_limits<uint32_t>::max() / 32u)
        throw std::runtime_error("software BVH node address overflow");
      geometries.push_back(
          {instance.geometry, static_cast<uint32_t>(node_count), leaves, count, instance.geometry_index});
      geometry = geometries.end() - 1;
      node_count += uint64_t(leaves) * 2 - 1;
      max_leaves = std::max(max_leaves, leaves);
    }
    const auto *graph = instance.material->GraphImpl();
    MaterialCode source{graph != nullptr, MaterialSource(graph ? *graph : instance.material->SamplerImpl())};
    auto material = std::find(materials.begin(), materials.end(), source);
    uint32_t material_index = std::distance(materials.begin(), material);
    if (material == materials.end())
      materials.push_back(std::move(source));
    glm::mat4 object_to_world(instance.transform);
    if (std::abs(glm::determinant(object_to_world)) < 1.0e-20f)
      throw std::runtime_error("software ray tracing requires invertible instance transforms");
    gpu_instances.push_back({instance.transform, glm::mat4x3(glm::inverse(object_to_world)), geometry->root,
                             instance.geometry_index, material_index, geometry->count});
    if (ray_query_) {
      auto blas = instance.geometry->BLAS();
      if (!blas)
        throw std::runtime_error("failed to build native triangle BLAS");
      rhi::RayTracingInstance native;
      native.blas = blas;
      native.custom_index = static_cast<uint32_t>(native_instances.size());
      for (uint32_t row = 0; row < 3; ++row)
        for (uint32_t column = 0; column < 4; ++column)
          native.transform[row * 4 + column] = instance.transform[column][row];
      native_instances.push_back(native);
    }
  }

  bool rebuild = !nodes_ || tlas_leaves != tlas_leaves_ || geometries.size() != geometries_.size();
  for (size_t i = 0; !rebuild && i < geometries.size(); ++i)
    rebuild = geometries[i].geometry != geometries_[i].geometry || geometries[i].count != geometries_[i].count;
  if (rebuild && !ray_query_) {
    nodes_ = device->CreateBuffer({node_count * 32});
    keys_ = device->CreateBuffer({uint64_t(max_leaves) * 8});
  }

  size_t instance_bytes = 16 + gpu_instances.size() * sizeof(GPUInstance);
  if (!instances_buffer_ || instances_buffer_->GetDesc().size < instance_bytes)
    instances_buffer_ = device->CreateBuffer({instance_bytes});
  std::array<uint32_t, 4> header{static_cast<uint32_t>(gpu_instances.size()), 0, 0, 0};
  UploadBuffer(core_->GetDevice(), instances_buffer_.Get(), header.data(), sizeof(header));
  if (!gpu_instances.empty())
    UploadBuffer(core_->GetDevice(), instances_buffer_.Get(), gpu_instances.data(),
                 gpu_instances.size() * sizeof(GPUInstance), 16);
  if (ray_query_) {
    if (!native_tlas_ || native_tlas_->GetDesc().instance_count < native_instances.size())
      native_tlas_ = device->CreateTopLevelAccelerationStructure(
          {static_cast<uint32_t>(std::max<size_t>(1, native_instances.size()))});
    if (!native_tlas_ || !commands->CmdBuildTopLevelAccelerationStructure(native_tlas_.Get(), native_instances))
      throw std::runtime_error("failed to build native ray query TLAS");
    if (!render_program_ || camera_file_ != camera_file || camera_entry_ != camera_entry ||
        materials != material_sources_ || buffer_count_ != buffers.size() || sdr_count_ != sdr_count ||
        hdr_count_ != hdr_count)
      CompileRenderer(materials, buffers.size(), sdr_count, hdr_count, camera_file, camera_entry);
    if (FrameProfile::active) {
      FrameProfile::active->counters["native_ray_query"] = 1;
      FrameProfile::active->counters["instances"] = instances_.size();
    }
    return;
  }
  if (builders_.empty() || builder_buffer_count_ != buffers.size())
    CompileBuilders(buffers.size());
  if (!render_program_ || camera_file_ != camera_file || camera_entry_ != camera_entry ||
      materials != material_sources_ || buffer_count_ != buffers.size() || sdr_count_ != sdr_count ||
      hdr_count_ != hdr_count)
    CompileRenderer(materials, buffers.size(), sdr_count, hdr_count, camera_file, camera_entry);
  std::vector<BuildPass> passes;
  if (rebuild)
    for (const auto &geometry : geometries) {
      BuildParameters parameters{};
      parameters.root = geometry.root;
      parameters.leaves = geometry.leaves;
      parameters.count = geometry.count;
      parameters.geometry = geometry.buffer_index;
      AppendBuild(passes, parameters);
    }
  BuildParameters tlas{};
  tlas.leaves = tlas_leaves;
  tlas.count = gpu_instances.size();
  tlas.instance_tree = 1;
  AppendBuild(passes, tlas);
  std::vector<BuildParameters> parameters;
  for (const auto &pass : passes)
    parameters.push_back(pass.parameters);
  size_t parameter_bytes = parameters.size() * sizeof(BuildParameters);
  if (!parameters_buffer_ || parameters_buffer_->GetDesc().size < parameter_bytes)
    parameters_buffer_ = device->CreateBuffer({parameter_bytes});
  UploadBuffer(core_->GetDevice(), parameters_buffer_.Get(), parameters.data(), parameter_bytes);
  prepare_profile.End();
  if (FrameProfile::active) {
    FrameProfile::active->counters["bvh_dispatches"] = passes.size();
    FrameProfile::active->counters["blas_rebuilt"] = rebuild;
    FrameProfile::active->counters["instances"] = instances_.size();
    FrameProfile::active->counters["bvh_nodes"] = node_count;
  }
  CpuProfileScope record_profile("bvh_record");
  for (size_t i = 0; i < passes.size(); ++i) {
    commands->CmdBindComputeProgram(builders_[passes[i].kernel].Get());
    build_group_->BindResource(0, nodes_.Get());
    build_group_->BindResource(1, buffers);
    build_group_->BindResource(
        2, rhi::BufferView{parameters_buffer_.Get(), i * sizeof(BuildParameters), sizeof(BuildParameters)});
    build_group_->BindResource(3, keys_.Get());
    build_group_->BindResource(4, instances_buffer_.Get());
    commands->CmdBindGroup(0, build_group_.Get());
    commands->CmdDispatch((passes[i].count + 63) / 64, 1, 1);
  }
  geometries_ = std::move(geometries);
  tlas_leaves_ = tlas_leaves;
}
}  // namespace sparkium::raytracing
