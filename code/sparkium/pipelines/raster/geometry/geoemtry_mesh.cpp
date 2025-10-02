#include "sparkium/pipelines/raster/geometry/geoemtry_mesh.h"

#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

GeometryMesh::GeometryMesh(sparkium::GeometryMesh &geometry)
    : geometry_(geometry), Geometry(DedicatedCast(geometry.GetCore())) {
  auto header = geometry_.GetHeader();
  std::vector<std::string> args;
  if (header.normal_offset) {
    args.push_back("-DHAS_NORMAL");
  }
  if (header.tex_coord_offset) {
    args.push_back("-DHAS_TEXCOORD");
  }
  if (header.tangent_offset) {
    args.push_back("-DHAS_TANGENT");
  }
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "geometry/mesh/vertex_shader.hlsl", "VSMain", "vs_6_0",
                                      args, &vertex_shader_);
}

graphics::Shader *GeometryMesh::VertexShader() {
  return vertex_shader_.get();
}

void GeometryMesh::SetupProgram(graphics::Program *program) {
  auto &header = geometry_.GetHeader();
  program->AddInputBinding(header.position_stride);
  int binding = 0;
  program->AddInputAttribute(binding++, graphics::INPUT_TYPE_FLOAT3, 0);
  if (header.normal_offset) {
    program->AddInputBinding(header.normal_stride);
    program->AddInputAttribute(binding++, graphics::INPUT_TYPE_FLOAT3, 0);
  }
  if (header.tex_coord_offset) {
    program->AddInputBinding(header.tex_coord_stride);
    program->AddInputAttribute(binding++, graphics::INPUT_TYPE_FLOAT2, 0);
  }
  if (header.tangent_offset) {
    program->AddInputBinding(header.tangent_stride);
    program->AddInputAttribute(binding++, graphics::INPUT_TYPE_FLOAT3, 0);
    program->AddInputBinding(header.signal_stride);
    program->AddInputAttribute(binding++, graphics::INPUT_TYPE_FLOAT, 0);
  }
}

void GeometryMesh::DispatchDrawCalls(graphics::CommandContext *cmd_ctx) {
  auto &header = geometry_.GetHeader();
  std::vector<graphics::Buffer *> buffers;
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
  cmd_ctx->CmdBindVertexBuffers(0, buffers, offsets);
  cmd_ctx->CmdBindIndexBuffer(geometry_.GetBuffer(), header.index_offset);
  cmd_ctx->CmdDrawIndexed(geometry_.PrimitiveCount() * 3, 1, 0, 0, 0);
}

}  // namespace sparkium::raster
