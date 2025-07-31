#include "light_geometry_surface.h"

#include "sparks/core/core.h"
#include "sparks/core/geometry.h"
#include "sparks/core/surface.h"
#include "sparks/light/light_geometry_surface.h"

namespace sparks {
LightGeometrySurface::LightGeometrySurface(Core *core,
                                           Geometry *geometry,
                                           Surface *surface,
                                           const glm::mat4x3 &transform)
    : Light(core), geometry_(geometry), surface_(surface), transform_(transform) {
  core_->GraphicsCore()->CreateBuffer(
      sizeof(glm::mat4x3) + sizeof(uint32_t) + geometry->PrimitiveCount() * sizeof(float), graphics::BUFFER_TYPE_STATIC,
      &direct_lighting_sampler_data_);
  CodeLines gather_primitive_power_kernel(core_->GetShadersVFS(), "light/geometry_surface/gather_primitive_power.hlsl");
  gather_primitive_power_kernel.InsertAfter(geometry_->SamplerImplementation(), "// Geometry Sampler Implementation");
  gather_primitive_power_kernel.InsertAfter(surface_->SamplerImplementation(), "// Surface Sampler Implementation");
  std::cout << gather_primitive_power_kernel << std::endl;

  auto vfs = core_->GetShadersVFS();
  vfs.WriteFile("light/geometry_surface/gather_primitive_power.hlsl", std::string(gather_primitive_power_kernel));
  core_->GraphicsCore()->CreateShader(vfs, "light/geometry_surface/gather_primitive_power.hlsl",
                                      "GatherPrimitivePowerKernel", "cs_6_3", {"-I."}, &gather_primitive_power_shader);

  uint32_t primitive_count = geometry_->PrimitiveCount();
  uint32_t wave_size = core_->GraphicsCore()->WaveSize();

  core_->GraphicsCore()->CreateComputeProgram(gather_primitive_power_shader.get(), &gather_primitive_power_program);
  gather_primitive_power_program->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
  gather_primitive_power_program->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
  gather_primitive_power_program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  gather_primitive_power_program->Finalize();

  BlellochScanMetadata metadata{52, 4, primitive_count};
  while (metadata.element_count > 1) {
    metadatas.emplace_back(metadata);
    metadata = {metadata.offset + (wave_size - 1) * metadata.stride, metadata.stride * wave_size,
                metadata.element_count / wave_size};
  }
  core_->GraphicsCore()->CreateBuffer(metadatas.size() * sizeof(BlellochScanMetadata), graphics::BUFFER_TYPE_STATIC,
                                      &metadata_buffer);
  metadata_buffer->UploadData(metadatas.data(), metadatas.size() * sizeof(BlellochScanMetadata));

  direct_lighting_sampler_data_->UploadData(&transform, sizeof(glm::mat4x3), 0);
  direct_lighting_sampler_data_->UploadData(&primitive_count, sizeof(uint32_t), sizeof(glm::mat4x3));
}

graphics::Shader *LightGeometrySurface::SamplerShader() {
  uint32_t wave_size = core_->GraphicsCore()->WaveSize();
  std::unique_ptr<graphics::CommandContext> cmd_context;
  core_->GraphicsCore()->CreateCommandContext(&cmd_context);
  blelloch_scan_up_program = core_->GetComputeProgram("blelloch_scan_up");
  blelloch_scan_down_program = core_->GetComputeProgram("blelloch_scan_down");
  cmd_context->CmdBindComputeProgram(gather_primitive_power_program.get());
  cmd_context->CmdBindResources(0, {geometry_->Buffer()}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdBindResources(1, {surface_->Buffer()}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdBindResources(2, {direct_lighting_sampler_data_.get()}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdDispatch((geometry_->PrimitiveCount() + 127) / 128, 1, 1);
  cmd_context->CmdBindComputeProgram(blelloch_scan_up_program);
  cmd_context->CmdBindResources(0, {direct_lighting_sampler_data_.get()}, graphics::BIND_POINT_COMPUTE);

  for (size_t i = 1; i < metadatas.size(); i++) {
    cmd_context->CmdBindResources(1, {metadata_buffer->Range(sizeof(BlellochScanMetadata) * i)},
                                  graphics::BIND_POINT_COMPUTE);
    cmd_context->CmdDispatch((metadatas[i].element_count + 63) / 64, 1, 1);
  }

  cmd_context->CmdBindComputeProgram(blelloch_scan_down_program);

  for (size_t i = metadatas.size() - 1; i < metadatas.size(); i--) {
    if (metadatas[i].element_count < wave_size) {
      continue;
    }
    cmd_context->CmdBindResources(1, {metadata_buffer->Range(sizeof(BlellochScanMetadata) * i)},
                                  graphics::BIND_POINT_COMPUTE);
    cmd_context->CmdDispatch((metadatas[i].element_count + 63) / 64, 1, 1);
  }

  core_->GraphicsCore()->SubmitCommandContext(cmd_context.get());
  // std::vector<float> primitive_power(geometry_->PrimitiveCount(), 0.0f);
  // direct_lighting_sampler_data_->DownloadData(primitive_power.data(), geometry_->PrimitiveCount() * sizeof(float),
  //                                             sizeof(glm::mat4x3) + sizeof(uint32_t));
  // float total_power = 0.0f;
  // total_power = primitive_power.back();
  // std::cout << "Total Power: " << total_power << std::endl;
  return nullptr;
}

graphics::Buffer *LightGeometrySurface::SamplerData() {
  return direct_lighting_sampler_data_.get();
}

void LightGeometrySurface::Update(Scene *scene, uint32_t light_index) {
}

}  // namespace sparks
