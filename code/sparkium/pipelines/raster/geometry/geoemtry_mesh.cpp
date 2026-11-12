#include "sparkium/pipelines/raster/geometry/geoemtry_mesh.h"

#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

GeometryMesh::GeometryMesh(sparkium::GeometryMesh &geometry)
    : geometry_(geometry),
      Geometry(DedicatedCast(geometry.GetCore())) {
  auto header = geometry_.GetHeader();
  std::vector<std::pair<std::string, std::string>> defines;
  if (header.normal_offset) {
    defines.emplace_back("HAS_NORMAL", "1");
  }
  if (header.tex_coord_offset) {
    defines.emplace_back("HAS_TEXCOORD", "1");
  }
  if (header.tangent_offset) {
    defines.emplace_back("HAS_TANGENT", "1");
  }
  auto shaders_vfs = core_->GetShadersVFS();
  vertex_shader_ = core_->GetDevice()->CreateShader("geometry/mesh/vertex_shader.slang", "VSMain",
                                                    rhi::SHADER_STAGE_VERTEX, &shaders_vfs, {"."}, defines);
}

rhi::IShader *GeometryMesh::VertexShader() {
  return vertex_shader_.Get();
}

void GeometryMesh::SetupProgram(rhi::ProgramDesc &program) {
  const auto &header = geometry_.GetHeader();
  auto attribute = [&](uint32_t location, uint32_t stride, rhi::InputType type) {
    const auto binding = static_cast<uint32_t>(program.vertex_bindings.size());
    program.vertex_bindings.push_back({stride});
    program.vertex_attributes.push_back({location, binding, type, 0});
  };
  attribute(0, header.position_stride, rhi::INPUT_TYPE_FLOAT3);
  if (header.normal_offset)
    attribute(1, header.normal_stride, rhi::INPUT_TYPE_FLOAT3);
  if (header.tex_coord_offset)
    attribute(2, header.tex_coord_stride, rhi::INPUT_TYPE_FLOAT2);
  if (header.tangent_offset) {
    attribute(3, header.tangent_stride, rhi::INPUT_TYPE_FLOAT3);
    attribute(4, header.signal_stride, rhi::INPUT_TYPE_FLOAT);
  }
}

void GeometryMesh::DispatchDrawCalls(rhi::ICommandContext *cmd_ctx) {
  auto &header = geometry_.GetHeader();
  std::vector<rhi::IBuffer *> buffers;
  std::vector<uint64_t> offsets;
  buffers.push_back(geometry_.GetBuffer());
  offsets.push_back(header.position_offset);

  if (header.normal_offset) {
    buffers.push_back(geometry_.GetBuffer());
    offsets.push_back(header.normal_offset);
  }
  if (header.tex_coord_offset) {
    buffers.push_back(geometry_.GetBuffer());
    offsets.push_back(header.tex_coord_offset);
  }
  if (header.tangent_offset) {
    buffers.push_back(geometry_.GetBuffer());
    buffers.push_back(geometry_.GetBuffer());
    offsets.push_back(header.tangent_offset);
    offsets.push_back(header.signal_offset);
  }
  for (uint32_t i = 0; i < buffers.size(); ++i)
    cmd_ctx->CmdBindVertexBuffer(i, buffers[i], offsets[i]);
  cmd_ctx->CmdBindIndexBuffer(geometry_.GetBuffer(), rhi::INDEX_FORMAT_UINT32, header.index_offset);
  cmd_ctx->CmdDrawIndexed(geometry_.PrimitiveCount() * 3, 1, 0, 0, 0);
}

glm::vec4 GeometryMesh::CentricArea(const glm::mat4x3 &affine) {
  return geometry_.CentricArea(affine);
}

}  // namespace sparkium::raster
