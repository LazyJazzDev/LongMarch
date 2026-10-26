#pragma once
#include "sparkium/pipelines/raytracing/core/core_util.h"

namespace sparkium::raytracing {

// Shared compute path tracer: software BVH by default, or native inline ray queries.
// Geometry/light/material registration is shared with DXR/Vulkan pipeline RT.
class SoftwarePipeline {
 public:
  explicit SoftwarePipeline(Core *core, bool ray_query = false);
  void ClearInstances();
  void AddInstance(Geometry *geometry, Material *material, const glm::mat4x3 &transform, uint32_t geometry_index);
  void Update(rhi::ICommandContext *commands,
              const std::vector<rhi::IBuffer *> &buffers,
              uint32_t sdr_count,
              uint32_t hdr_count,
              Camera *camera);

  rhi::IComputeProgram *Program() const {
    return render_program_.Get();
  }

  rhi::ITopLevelAccelerationStructure *AccelerationStructure() const {
    return native_tlas_.Get();
  }

  rhi::IBindingGroup *Bindings() const {
    return render_group_.Get();
  }

  rhi::IBuffer *Nodes() const {
    return nodes_.Get();
  }

  rhi::IBuffer *Instances() const {
    return instances_buffer_.Get();
  }

 private:
  struct Instance {
    Geometry *geometry;
    Material *material;
    glm::mat4x3 transform;
    uint32_t geometry_index;
  };

  struct GeometryLayout {
    Geometry *geometry;
    uint32_t root;
    uint32_t leaves;
    uint32_t count;
    uint32_t buffer_index;
  };

  struct BuildParameters {
    uint32_t root{}, leaves{}, count{}, geometry{}, first{}, level_count{}, stage{}, stride{}, instance_tree{};
    uint32_t padding[55]{};
  };

  struct BuildPass {
    uint32_t kernel;
    BuildParameters parameters;
    uint32_t count;
  };

  struct MaterialCode {
    bool shader_graph;
    std::string source;

    bool operator==(const MaterialCode &other) const {
      return shader_graph == other.shader_graph && source == other.source;
    }
  };

  void CompileBuilders(uint32_t buffer_count);
  void CompileRenderer(const std::vector<MaterialCode> &materials,
                       uint32_t buffers,
                       uint32_t sdr_count,
                       uint32_t hdr_count,
                       const std::string &camera_file,
                       const std::string &camera_entry);
  void AppendBuild(std::vector<BuildPass> &passes, BuildParameters parameters);

  Core *core_;
  bool ray_query_;
  std::string camera_file_, camera_entry_;
  rhi::ObjPtr<rhi::ITopLevelAccelerationStructure> native_tlas_;
  std::vector<Instance> instances_;
  std::vector<GeometryLayout> geometries_;
  uint32_t tlas_leaves_{};
  std::vector<MaterialCode> material_sources_;
  uint32_t buffer_count_{}, sdr_count_{}, hdr_count_{};
  uint32_t builder_buffer_count_{};
  rhi::ObjPtr<rhi::IBuffer> nodes_, keys_, instances_buffer_, parameters_buffer_;
  std::vector<rhi::ObjPtr<rhi::IShader>> builder_shaders_;
  std::vector<rhi::ObjPtr<rhi::IComputeProgram>> builders_;
  rhi::ObjPtr<rhi::IShader> render_shader_;
  rhi::ObjPtr<rhi::IComputeProgram> render_program_;
  rhi::ObjPtr<rhi::IBindingGroup> render_group_, build_group_;
};
}  // namespace sparkium::raytracing
