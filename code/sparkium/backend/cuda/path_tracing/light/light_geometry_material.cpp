#include "sparkium/backend/cuda/path_tracing/light/light_geometry_material.h"

#include "sparkium/backend/cuda/path_tracing/core/core.h"
#include "sparkium/backend/cuda/path_tracing/core/geometry.h"
#include "sparkium/backend/cuda/path_tracing/core/material.h"
#include "sparkium/backend/cuda/path_tracing/core/scene.h"
#include "sparkium/backend/cuda/path_tracing/geometry/geometries.h"
#include "sparkium/backend/cuda/path_tracing/material/materials.h"

namespace sparkium::cuda_tracing {
LightGeometryMaterial::LightGeometryMaterial(Core *core,
                                             Geometry *geometry,
                                             Material *material,
                                             const glm::mat4x3 &transform)
    : Light(core),
      geometry_(geometry),
      material_(material),
      transform(transform) {
  core_->BackendDevice()->CreateBuffer(
      sizeof(glm::mat4x3) + sizeof(uint32_t) + geometry->PrimitiveCount() * sizeof(float), graphics::BUFFER_TYPE_STATIC,
      &direct_lighting_sampler_data_);

  auto vfs = core_->GetShadersVFS();
  vfs.WriteFile("geometry_sampler.hlsli", geometry_->SamplerImpl());
  vfs.WriteFile("material_evaluator.hlsli", material_->EvaluatorImpl());

  core_->BackendDevice()->CreateShader(vfs, "light/geometry_material/gather_primitive_power.hlsl",
                                       "GatherPrimitivePowerKernel", "cs_6_3", {"-I."},
                                       &gather_primitive_power_shader_);

  uint32_t primitive_count = geometry_->PrimitiveCount();
  uint32_t group_size = 64;

  core_->BackendDevice()->CreateComputeProgram(gather_primitive_power_shader_.get(), &gather_primitive_power_program_);
  gather_primitive_power_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
  gather_primitive_power_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
  gather_primitive_power_program_->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  gather_primitive_power_program_->Finalize();

  BlellochScanMetadata metadata{52, 4, primitive_count};
  while (metadata.element_count > 1) {
    metadatas_.emplace_back(metadata);
    metadata = {metadata.offset + (group_size - 1) * metadata.stride, metadata.stride * group_size,
                metadata.element_count / group_size};
  }
  core_->BackendDevice()->CreateBuffer(std::max<size_t>(1, metadatas_.size()) * sizeof(BlellochScanMetadata),
                                       graphics::BUFFER_TYPE_STATIC, &metadata_buffer_);
  if (!metadatas_.empty())
    metadata_buffer_->UploadData(metadatas_.data(), metadatas_.size() * sizeof(BlellochScanMetadata));

  direct_lighting_sampler_data_->UploadData(&transform, sizeof(glm::mat4x3), 0);
  direct_lighting_sampler_data_->UploadData(&primitive_count, sizeof(uint32_t), sizeof(glm::mat4x3));
  blelloch_scan_up_program_ = core_->GetComputeProgram("blelloch_scan_up");
  blelloch_scan_down_program_ = core_->GetComputeProgram("blelloch_scan_down");
}

int LightGeometryMaterial::SamplerShader(Scene *scene) {
  if (dynamic_cast<GeometryMesh *>(geometry_)) {
    if (dynamic_cast<MaterialLight *>(material_)) {
      return 0x1000001;
    }
    if (dynamic_cast<MaterialLambertian *>(material_)) {
      return 0x1000002;
    }
    if (dynamic_cast<MaterialPrincipled *>(material_)) {
      return 0x1000003;
    }
    if (dynamic_cast<MaterialShaderGraph *>(material_))
      return 0x1000004;
    if (dynamic_cast<MaterialSpecular *>(material_))
      return 0x1000005;
  }
  return -1;
}

graphics::Buffer *LightGeometryMaterial::SamplerData() {
  direct_lighting_sampler_data_->UploadData(&transform, sizeof(glm::mat4x3), 0);
  return direct_lighting_sampler_data_.get();
}

uint32_t LightGeometryMaterial::SamplerPreprocess(graphics::CommandContext *cmd_context) {
  uint32_t group_size = 64;
  cmd_context->CmdBindComputeProgram(gather_primitive_power_program_.get());
  cmd_context->CmdBindResources(0, {geometry_->Buffer()}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdBindResources(1, {material_->Buffer()}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdBindResources(2, {direct_lighting_sampler_data_.get()}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdDispatch((geometry_->PrimitiveCount() + 63) / 64, 1, 1);
  cmd_context->CmdBindComputeProgram(blelloch_scan_up_program_);
  cmd_context->CmdBindResources(0, {direct_lighting_sampler_data_.get()}, graphics::BIND_POINT_COMPUTE);

  for (size_t i = 1; i < metadatas_.size(); i++) {
    cmd_context->CmdBindResources(1, {metadata_buffer_->Range(sizeof(BlellochScanMetadata) * i)},
                                  graphics::BIND_POINT_COMPUTE);
    cmd_context->CmdDispatch((metadatas_[i].element_count + 63) / 64, 1, 1);
  }

  cmd_context->CmdBindComputeProgram(blelloch_scan_down_program_);

  for (size_t i = metadatas_.size() - 1; i < metadatas_.size(); i--) {
    if (metadatas_[i].element_count < group_size) {
      continue;
    }
    cmd_context->CmdBindResources(1, {metadata_buffer_->Range(sizeof(BlellochScanMetadata) * i)},
                                  graphics::BIND_POINT_COMPUTE);
    cmd_context->CmdDispatch((metadatas_[i].element_count + 63) / 64, 1, 1);
  }
  return sizeof(glm::mat4x3) + sizeof(uint32_t) + (geometry_->PrimitiveCount() - 1) * sizeof(float);
}
}  // namespace sparkium::cuda_tracing
