#include "sparkium/pipelines/raytracing/geometry/geometry_mesh.h"

#include "sparkium/pipelines/raytracing/core/core.h"

namespace sparkium::raytracing {

GeometryMesh::GeometryMesh(sparkium::GeometryMesh &geometry)
    : geometry_(geometry),
      Geometry(DedicatedCast(geometry.GetCore())) {
  auto &vfs = core_->GetShadersVFS();
  sampler_implementation_ = CodeLines(vfs, "geometry/mesh/geometry_sampler.slang");
  closest_hit_shader_implementation_ = CodeLines(vfs, "geometry/mesh/hit_group.slang");
}

rhi::IBuffer *GeometryMesh::Buffer() {
  return geometry_.GetBuffer();
}

const CodeLines &GeometryMesh::ClosestHitShaderImpl() const {
  return closest_hit_shader_implementation_;
}

int GeometryMesh::PrimitiveCount() {
  return geometry_.PrimitiveCount();
}

const CodeLines &GeometryMesh::SamplerImpl() const {
  return sampler_implementation_;
}

rhi::IBottomLevelAccelerationStructure *GeometryMesh::BLAS() {
  if (!blas_) {
    auto header = geometry_.GetHeader();

    rhi::BottomLevelAccelerationStructureDesc desc;
    desc.geometries.push_back(
        {header.num_vertices, header.position_stride, header.num_indices, rhi::INDEX_FORMAT_UINT32, false});
    blas_ = core_->GetDevice()->CreateBottomLevelAccelerationStructure(desc);
    if (!blas_)
      return nullptr;
    rhi::ObjPtr<rhi::ICommandContext> commands = core_->GetDevice()->CreateCommandContext();
    if (!commands) {
      blas_.Release();
      return nullptr;
    }
    if (!commands->CmdBuildBottomLevelAccelerationStructure(
            blas_.Get(),
            {{{geometry_.GetBuffer(), header.position_offset}, {geometry_.GetBuffer(), header.index_offset}}})) {
      blas_.Release();
      return nullptr;
    }
    if (!Execute(core_->GetDevice(), commands.Get())) {
      blas_.Release();
      return nullptr;
    }
  }
  return blas_.Get();
}

}  // namespace sparkium::raytracing
