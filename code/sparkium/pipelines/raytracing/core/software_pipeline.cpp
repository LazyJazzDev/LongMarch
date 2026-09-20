#include "sparkium/pipelines/raytracing/core/software_pipeline.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "grassland/graphics/frame_profile.h"
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

static_assert(sizeof(GPUInstance) == 112, "HLSL software instance layout changed");
}  // namespace

SoftwarePipeline::SoftwarePipeline(Core *core, bool ray_query, bool optix)
    : core_(core),
      ray_query_(ray_query),
      optix_(optix) {
  if (ray_query_ && !core_->BackendDevice()->DeviceRayQuerySupport())
    throw std::runtime_error("native ray queries are unavailable on the selected backend");
  cpu_ = core_->BackendDevice()->API() == RenderBackend::CPU;
  if (cpu_)
    if (const char *choice = std::getenv("SPARKIUM_CPU_BVH")) {
      if (std::string(choice) == "heap")
        cpu_ = false;  // Differential/ablation reference.
      else if (std::string(choice) != "sah")
        throw std::invalid_argument("SPARKIUM_CPU_BVH must be sah or heap");
    }
  if (optix_ &&
      (core_->BackendDevice()->API() != RenderBackend::CUDA || !core_->BackendDevice()->DeviceRayTracingSupport()))
    throw std::runtime_error("OptiX traversal requires CUDA hardware ray tracing support");
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
  graphics::CpuProfileScope compile_profile("compile_builders");
  auto graphics = core_->BackendDevice();
  builders_.clear();
  if (builder_shaders_.empty()) {
    for (const char *entry : {"InitLeaves", "ReduceNodes", "MortonKeys", "BitonicSort", "SortLeaves"}) {
      std::unique_ptr<graphics::Shader> shader;
      if (graphics->CreateShader(core_->GetShadersVFS(), "software/build.hlsl", entry, "cs_6_0", {"-I."}, &shader))
        throw std::runtime_error(std::string("failed to compile software BVH kernel: ") + entry);
      builder_shaders_.push_back(std::move(shader));
    }
  }
  for (auto &shader : builder_shaders_) {
    std::unique_ptr<graphics::ComputeProgram> program;
    graphics->CreateComputeProgram(shader.get(), &program);
    program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
    program->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, buffer_count);
    program->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
    program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
    program->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
    program->Finalize();
    builders_.push_back(std::move(program));
  }
  builder_buffer_count_ = buffer_count;
}

void SoftwarePipeline::CompileRenderer(const std::vector<MaterialCode> &materials,
                                       uint32_t buffers,
                                       uint32_t sdr_count,
                                       uint32_t hdr_count) {
  graphics::CpuProfileScope compile_profile("compile_renderer");
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
float Transmission(SP_CONTEXT HitRecord hit, float3 direction) {
#ifdef SAMPLE_SHADOW_ANY_HIT
  return 1.0f - saturate(SampleShadowOpacity(SP_CONTEXT_ARG hit, direction));
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
#undef SAMPLE_SHADOW_ANY_HIT
#undef SAMPLE_SHADOW_NO_HITRECORD
)";
  }
  if (has_graph) {
    source << R"(
  ByteAddressBuffer SoftwareMaterialData(SP_CONTEXT HitRecord hit) {
    InstanceMetadata metadata = SP_BINDING_instance_metadatas.Load<InstanceMetadata>(sizeof(InstanceMetadata) * hit.object_index);
    return SP_BINDING_data_buffers[SP_NONUNIFORM(metadata.material_data_index)];
  }
  void SoftwareSampleMaterial(SP_CONTEXT uint material, inout RenderContext context, HitRecord hit) {
    GraphSurface graph;
    switch (material) {
  )";
    for (size_t i = 0; i < materials.size(); ++i) {
      source << "case " << i << ": ";
      if (materials[i].shader_graph)
        source
            << "graph = SoftwareMaterial" << i
            << "::EvaluateShaderGraph(SP_CONTEXT_ARG hit, -context.direction, context.bounce, context.ray_type, false, "
               "SoftwareMaterialData(SP_CONTEXT_ARG hit)); break;\n";
      else
        source << "SoftwareMaterial" << i << "::SampleMaterial(SP_CONTEXT_ARG context, hit); return;\n";
    }
    source << R"(
      default: context.throughput = float3(0, 0, 0); return;
    }
    SampleGraphSurface(SP_CONTEXT_ARG context, hit, graph);
  }
  float SoftwareShadowTransmission(SP_CONTEXT uint material, HitRecord hit, float3 direction) {
    switch (material) {
  )";
    for (size_t i = 0; i < materials.size(); ++i) {
      source << "case " << i << ": return ";
      if (materials[i].shader_graph)
        source << "1.0f - saturate(GraphShadowOpacity(SoftwareMaterial" << i
               << "::EvaluateShaderGraph(SP_CONTEXT_ARG hit, -direction, 1, RAY_TYPE_REFLECTION, true, "
                  "SoftwareMaterialData(SP_CONTEXT_ARG hit))));\n";
      else
        source << "SoftwareMaterial" << i << "::Transmission(SP_CONTEXT_ARG hit, direction);\n";
    }
    source << "default: return 0.0f;\n}}\n";
  } else {
    // Preserve the compact dispatch for scenes without material graphs.
    source << "void SoftwareSampleMaterial(SP_CONTEXT uint material, inout RenderContext context, HitRecord hit) {\n"
              "switch (material) {\n";
    for (size_t i = 0; i < materials.size(); ++i)
      source << "case " << i << ": SoftwareMaterial" << i << "::SampleMaterial(SP_CONTEXT_ARG context, hit); return;\n";
    source << "default: context.throughput = float3(0, 0, 0); break;\n}}\n"
              "float SoftwareShadowTransmission(SP_CONTEXT uint material, HitRecord hit, float3 direction) {\nswitch "
              "(material) {\n";
    for (size_t i = 0; i < materials.size(); ++i)
      source << "case " << i << ": return SoftwareMaterial" << i << "::Transmission(SP_CONTEXT_ARG hit, direction);\n";
    source << "default: return 0.0f;\n}}\n";
  }

  auto vfs = core_->GetShadersVFS();
  vfs.WriteFile("software_materials.hlsli", source.str());
  render_program_.reset();
  std::vector<std::string> args{"-I.", "-DSOFTWARE_DATA_BUFFER_COUNT=" + std::to_string(buffers)};
  if (cpu_)
    args.push_back("-DSPARKIUM_CPU_SAH");
  if (ray_query_)
    args.push_back("-DSPARKIUM_RAY_QUERY");
  if (optix_)
    args.push_back("-DSPARKIUM_OPTIX");
  if (has_graph)
    args.push_back("-DSPARKIUM_SHADER_GRAPHS");
  if (core_->BackendDevice()->CreateShader(vfs, "software/render.hlsl", "Main", ray_query_ ? "cs_6_5" : "cs_6_0", args,
                                           &render_shader_))
    throw std::runtime_error("failed to compile compute ray tracing shader");
  core_->BackendDevice()->CreateComputeProgram(render_shader_.get(), &render_program_);
  for (auto binding : std::vector<std::pair<graphics::ResourceType, uint32_t>>{
           {graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1},
           {graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1},
           {NativeTraversal() ? graphics::RESOURCE_TYPE_ACCELERATION_STRUCTURE : graphics::RESOURCE_TYPE_STORAGE_BUFFER,
            1},
           {graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1},
           {graphics::RESOURCE_TYPE_STORAGE_BUFFER, buffers + 6},
           {graphics::RESOURCE_TYPE_IMAGE, sdr_count},
           {graphics::RESOURCE_TYPE_IMAGE, hdr_count},
           {graphics::RESOURCE_TYPE_SAMPLER, 2}})
    render_program_->AddResourceBinding(binding.first, binding.second);
  render_program_->Finalize();
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

void SoftwarePipeline::Update(graphics::CommandContext *commands,
                              const std::vector<graphics::Buffer *> &buffers,
                              uint32_t sdr_count,
                              uint32_t hdr_count) {
  auto graphics = core_->BackendDevice();
  graphics::CpuProfileScope prepare_profile("software_prepare");
  std::vector<GeometryLayout> geometries;
  std::vector<GPUInstance> gpu_instances;
  std::vector<graphics::RayTracingInstance> native_instances;
  std::vector<MaterialCode> materials;
  if (16ull + instances_.size() * sizeof(GPUInstance) > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("software instance byte address overflow");
  const uint32_t tlas_leaves = (NativeTraversal() || cpu_) ? 1 : LeafCount(instances_.size());
  uint64_t node_count = uint64_t(tlas_leaves) * 2 - 1;
  uint32_t max_leaves = tlas_leaves;
  for (const auto &instance : instances_) {
    auto geometry = std::find_if(geometries.begin(), geometries.end(),
                                 [&](const GeometryLayout &g) { return g.geometry == instance.geometry; });
    if (geometry == geometries.end()) {
      const uint32_t count = instance.geometry->PrimitiveCount(),
                     leaves = (NativeTraversal() || cpu_) ? 1 : LeafCount(count);
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
    if (NativeTraversal()) {
      auto blas = instance.geometry->BLAS();
      if (!blas)
        throw std::runtime_error("failed to build native triangle BLAS");
      native_instances.push_back(
          blas->MakeInstance(instance.transform, static_cast<uint32_t>(native_instances.size())));
    }
  }

  bool rebuild = !nodes_ || tlas_leaves != tlas_leaves_ || geometries.size() != geometries_.size();
  for (size_t i = 0; !rebuild && i < geometries.size(); ++i)
    rebuild = geometries[i].geometry != geometries_[i].geometry || geometries[i].count != geometries_[i].count;
  if (rebuild && !NativeTraversal() && !cpu_) {
    graphics->CreateBuffer(node_count * 32, graphics::BUFFER_TYPE_STATIC, &nodes_);
    graphics->CreateBuffer(uint64_t(max_leaves) * 8, graphics::BUFFER_TYPE_STATIC, &keys_);
  }
  if (cpu_) {
    graphics::CpuProfileScope build_profile("cpu_bvh_build");
    if (rebuild) {
      cpu_meshes_.clear();
      for (const auto &geometry : geometries) {
        auto buffer = geometry.geometry->Buffer();
        std::vector<uint8_t> bytes(buffer->Size());
        buffer->DownloadData(bytes.data(), bytes.size());
        cpu_meshes_.push_back(BuildCpuMeshBvh(bytes, geometry.count));
      }
    }
    std::vector<CpuBounds> bounds;
    // Reserve enough TLAS space to keep BLAS byte offsets stable across refits.
    size_t offset = 16 + std::max<size_t>(1, gpu_instances.size() * 2) * sizeof(CpuBvhNode) +
                    gpu_instances.size() * sizeof(uint32_t);
    offset = (offset + 15) & ~size_t(15);
    const size_t blas_offset = offset;
    for (size_t i = 0; i < geometries.size(); ++i) {
      if (offset + cpu_meshes_[i].bytes.size() > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("CPU BVH byte offset overflow");
      geometries[i].root = uint32_t(offset);
      offset += cpu_meshes_[i].bytes.size();
    }
    for (size_t i = 0; i < instances_.size(); ++i) {
      auto it = std::find_if(geometries.begin(), geometries.end(),
                             [&](const GeometryLayout &g) { return g.geometry == instances_[i].geometry; });
      size_t index = it - geometries.begin();
      gpu_instances[i].root = it->root;
      CpuBounds box;
      const auto &local = cpu_meshes_[index].bounds;
      if (it->count)
        for (unsigned corner = 0; corner < 8; ++corner) {
          glm::vec3 p{corner & 1 ? local.hi.x : local.lo.x, corner & 2 ? local.hi.y : local.lo.y,
                      corner & 4 ? local.hi.z : local.lo.z};
          box.Extend(instances_[i].transform * glm::vec4(p, 1));
        }
      bounds.push_back(box);
    }
    std::vector<uint8_t> instance_key(gpu_instances.size() * sizeof(GPUInstance));
    if (!instance_key.empty())
      std::memcpy(instance_key.data(), gpu_instances.data(), instance_key.size());
    if (rebuild || instance_key != cpu_last_instances_) {
      auto tlas = BuildCpuBvh(bounds, 1);
      if (tlas.bytes.size() > blas_offset)
        throw std::logic_error("CPU TLAS reserve too small");
      if (!nodes_ || nodes_->Size() != offset)
        graphics->CreateBuffer(offset, graphics::BUFFER_TYPE_STATIC, &nodes_);
      nodes_->UploadData(tlas.bytes.data(), tlas.bytes.size());
      if (rebuild || instance_key.size() != cpu_last_instances_.size())
        for (size_t i = 0; i < geometries.size(); ++i)
          nodes_->UploadData(cpu_meshes_[i].bytes.data(), cpu_meshes_[i].bytes.size(), geometries[i].root);
      cpu_last_instances_ = std::move(instance_key);
    }
    if (graphics::FrameProfile::active) {
      graphics::FrameProfile::active->counters["cpu_sah_bvh"] = 1;
      graphics::FrameProfile::active->counters["blas_rebuilt"] = rebuild;
      graphics::FrameProfile::active->counters["instances"] = instances_.size();
    }
  }

  size_t instance_bytes = 16 + gpu_instances.size() * sizeof(GPUInstance);
  if (!instances_buffer_ || instances_buffer_->Size() < instance_bytes)
    graphics->CreateBuffer(instance_bytes, graphics::BUFFER_TYPE_STATIC, &instances_buffer_);
  std::array<uint32_t, 4> header{static_cast<uint32_t>(gpu_instances.size()), 0, 0, 0};
  instances_buffer_->UploadData(header.data(), sizeof(header));
  if (!gpu_instances.empty())
    instances_buffer_->UploadData(gpu_instances.data(), gpu_instances.size() * sizeof(GPUInstance), 16);
  if (NativeTraversal()) {
    if (!native_tlas_) {
      if (graphics->CreateTopLevelAccelerationStructure(native_instances, &native_tlas_))
        throw std::runtime_error("failed to create native TLAS");
    } else if (native_tlas_->UpdateInstances(native_instances)) {
      throw std::runtime_error("failed to update native TLAS");
    }
    if (!render_program_ || materials != material_sources_ || buffer_count_ != buffers.size() ||
        sdr_count_ != sdr_count || hdr_count_ != hdr_count)
      CompileRenderer(materials, buffers.size(), sdr_count, hdr_count);
    if (graphics::FrameProfile::active) {
      graphics::FrameProfile::active->counters[optix_ ? "optix_hardware_traversal" : "native_ray_query"] = 1;
      graphics::FrameProfile::active->counters["instances"] = instances_.size();
    }
    return;
  }
  if (cpu_) {
    if (!render_program_ || materials != material_sources_ || buffer_count_ != buffers.size() ||
        sdr_count_ != sdr_count || hdr_count_ != hdr_count)
      CompileRenderer(materials, buffers.size(), sdr_count, hdr_count);
    geometries_ = std::move(geometries);
    tlas_leaves_ = tlas_leaves;
    return;
  }
  if (builders_.empty() || builder_buffer_count_ != buffers.size())
    CompileBuilders(buffers.size());
  if (!render_program_ || materials != material_sources_ || buffer_count_ != buffers.size() ||
      sdr_count_ != sdr_count || hdr_count_ != hdr_count)
    CompileRenderer(materials, buffers.size(), sdr_count, hdr_count);
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
  if (!parameters_buffer_ || parameters_buffer_->Size() < parameter_bytes)
    graphics->CreateBuffer(parameter_bytes, graphics::BUFFER_TYPE_STATIC, &parameters_buffer_);
  parameters_buffer_->UploadData(parameters.data(), parameter_bytes);
  prepare_profile.End();
  if (graphics::FrameProfile::active) {
    graphics::FrameProfile::active->counters["bvh_dispatches"] = passes.size();
    graphics::FrameProfile::active->counters["blas_rebuilt"] = rebuild;
    graphics::FrameProfile::active->counters["instances"] = instances_.size();
    graphics::FrameProfile::active->counters["bvh_nodes"] = node_count;
  }
  graphics::CpuProfileScope record_profile("bvh_record");
  graphics::GpuProfileScope build_profile(commands, "bvh_build");
  for (size_t i = 0; i < passes.size(); ++i) {
    commands->CmdBindComputeProgram(builders_[passes[i].kernel].get());
    commands->CmdBindResources(0, {nodes_.get()}, graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(1, buffers, graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(2, {parameters_buffer_->Range(i * sizeof(BuildParameters), sizeof(BuildParameters))},
                               graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(3, {keys_.get()}, graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(4, {instances_buffer_.get()}, graphics::BIND_POINT_COMPUTE);
    commands->CmdDispatch((passes[i].count + 63) / 64, 1, 1);
  }
  geometries_ = std::move(geometries);
  tlas_leaves_ = tlas_leaves;
}
}  // namespace sparkium::raytracing
