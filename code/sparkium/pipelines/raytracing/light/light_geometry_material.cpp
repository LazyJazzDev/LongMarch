#include "sparkium/pipelines/raytracing/light/light_geometry_material.h"

#include "sparkium/pipelines/raytracing/core/core.h"
#include "sparkium/pipelines/raytracing/core/geometry.h"
#include "sparkium/pipelines/raytracing/core/material.h"
#include "sparkium/pipelines/raytracing/core/scene.h"
#include "sparkium/pipelines/raytracing/geometry/geometries.h"
#include "sparkium/pipelines/raytracing/material/materials.h"

namespace sparkium::raytracing {
LightGeometryMaterial::LightGeometryMaterial(Core *core,
                                             Geometry *geometry,
                                             Material *material,
                                             const glm::mat4x3 &transform)
    : Light(core),
      geometry_(geometry),
      material_(material),
      transform(transform) {
  direct_lighting_sampler_data_ = core_->GetDevice()->CreateBuffer(
      {sizeof(glm::mat4x3) + sizeof(uint32_t) + geometry->PrimitiveCount() * sizeof(float)});

  gather_primitive_power_program_ =
      core_->GetGeometryLightPowerProgram(geometry_->SamplerImpl(), material_->EvaluatorImpl());
  uint32_t primitive_count = geometry_->PrimitiveCount();
  uint32_t group_size = 64;

  BlellochScanMetadata metadata{52, 4, primitive_count};
  while (metadata.element_count > 1) {
    metadatas_.emplace_back(metadata);
    metadata = {metadata.offset + (group_size - 1) * metadata.stride, metadata.stride * group_size,
                metadata.element_count / group_size};
  }
  metadata_buffer_ =
      core_->GetDevice()->CreateBuffer({std::max<size_t>(1, metadatas_.size()) * sizeof(BlellochScanMetadata)});
  if (!metadatas_.empty())
    UploadBuffer(core_->GetDevice(), metadata_buffer_.Get(), metadatas_.data(),
                 metadatas_.size() * sizeof(BlellochScanMetadata));

  UploadBuffer(core_->GetDevice(), direct_lighting_sampler_data_.Get(), &transform, sizeof(glm::mat4x3), 0);
  UploadBuffer(core_->GetDevice(), direct_lighting_sampler_data_.Get(), &primitive_count, sizeof(uint32_t),
               sizeof(glm::mat4x3));
  blelloch_scan_up_program_ = core_->GetComputeProgram("blelloch_scan_up");
  blelloch_scan_down_program_ = core_->GetComputeProgram("blelloch_scan_down");
  gather_group_ =
      core_->GetDevice()->CreateBindingGroup(gather_primitive_power_program_->GetDesc().pipeline_layout.groups[0]);
  scan_group_ = core_->GetDevice()->CreateBindingGroup(blelloch_scan_up_program_->GetDesc().pipeline_layout.groups[0]);
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
  if (!direct_lighting_sampler_ && !scene->SoftwareTracing()) {
    auto vfs = core_->GetShadersVFS();
    vfs.WriteFile("geometry_sampler.slang", geometry_->SamplerImpl());
    vfs.WriteFile("material_evaluator.slang", material_->EvaluatorImpl());
    direct_lighting_sampler_ =
        core_->GetDevice()->CreateShader("light/geometry_material/direct_lighting_sampler.slang",
                                         "SampleDirectLightingCallable", rhi::SHADER_STAGE_CALLABLE, &vfs, {"."}, {});
    if (!direct_lighting_sampler_)
      throw std::runtime_error("failed to compile geometry light callable");
  }
  return scene->RegisterCallableShader(direct_lighting_sampler_.Get());
}

rhi::IBuffer *LightGeometryMaterial::SamplerData() {
  UploadBuffer(core_->GetDevice(), direct_lighting_sampler_data_.Get(), &transform, sizeof(glm::mat4x3), 0);
  return direct_lighting_sampler_data_.Get();
}

uint32_t LightGeometryMaterial::SamplerPreprocess(rhi::ICommandContext *cmd_context) {
  uint32_t group_size = 64;
  cmd_context->CmdBindComputeProgram(gather_primitive_power_program_);
  gather_group_->BindResource(0, geometry_->Buffer());
  gather_group_->BindResource(1, material_->Buffer());
  gather_group_->BindResource(2, direct_lighting_sampler_data_.Get());
  cmd_context->CmdBindGroup(0, gather_group_.Get());
  cmd_context->CmdDispatch((geometry_->PrimitiveCount() + 63) / 64, 1, 1);
  cmd_context->CmdBindComputeProgram(blelloch_scan_up_program_);
  scan_group_->BindResource(0, direct_lighting_sampler_data_.Get());

  for (size_t i = 1; i < metadatas_.size(); i++) {
    scan_group_->BindResource(
        1, rhi::BufferView{metadata_buffer_.Get(), sizeof(BlellochScanMetadata) * i, sizeof(BlellochScanMetadata)});
    cmd_context->CmdBindGroup(0, scan_group_.Get());
    cmd_context->CmdDispatch((metadatas_[i].element_count + 63) / 64, 1, 1);
  }

  cmd_context->CmdBindComputeProgram(blelloch_scan_down_program_);

  for (size_t i = metadatas_.size() - 1; i < metadatas_.size(); i--) {
    if (metadatas_[i].element_count < group_size) {
      continue;
    }
    scan_group_->BindResource(
        1, rhi::BufferView{metadata_buffer_.Get(), sizeof(BlellochScanMetadata) * i, sizeof(BlellochScanMetadata)});
    cmd_context->CmdBindGroup(0, scan_group_.Get());
    cmd_context->CmdDispatch((metadatas_[i].element_count + 63) / 64, 1, 1);
  }
  return sizeof(glm::mat4x3) + sizeof(uint32_t) + (geometry_->PrimitiveCount() - 1) * sizeof(float);
}
}  // namespace sparkium::raytracing
