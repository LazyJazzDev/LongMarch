#pragma once
#include "sparkium/pipelines/raytracing/core/core_util.h"

namespace sparkium::raytracing {

// Path-tracing compute kernels, with software BVH or native ray queries.
class SoftwarePipeline {
 public:
  explicit SoftwarePipeline(Core *core, bool ray_query = false);
  void ClearInstances();
  void AddInstance(Geometry *geometry, Material *material, const glm::mat4x3 &transform, uint32_t geometry_index);
  void Update(graphics::CommandContext *commands,
              const std::vector<graphics::Buffer *> &buffers,
              uint32_t sdr_count,
              uint32_t hdr_count);

  std::vector<uint32_t> VertexCounts() const;

  graphics::ComputeProgram *Program() const {
    return render_program_.get();
  }

  graphics::AccelerationStructure *AccelerationStructure() const {
    return native_tlas_.get();
  }

  graphics::Buffer *Nodes() const {
    return nodes_.get();
  }

  graphics::Buffer *Instances() const {
    return instances_buffer_.get();
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
                       uint32_t hdr_count);
  void AppendBuild(std::vector<BuildPass> &passes, BuildParameters parameters);

  Core *core_;
  bool ray_query_;
  std::vector<uint8_t> previous_instances_;
  std::unique_ptr<graphics::AccelerationStructure> native_tlas_;
  std::vector<Instance> instances_;
  std::vector<GeometryLayout> geometries_;
  uint32_t tlas_leaves_{};
  std::vector<MaterialCode> material_sources_;
  uint32_t buffer_count_{}, sdr_count_{}, hdr_count_{};
  uint32_t builder_buffer_count_{};
  std::unique_ptr<graphics::Buffer> nodes_, keys_, instances_buffer_, parameters_buffer_;
  std::vector<std::unique_ptr<graphics::Shader>> builder_shaders_;
  std::vector<std::unique_ptr<graphics::ComputeProgram>> builders_;
  std::unique_ptr<graphics::Shader> render_shader_;
  std::unique_ptr<graphics::ComputeProgram> render_program_;
};
}  // namespace sparkium::raytracing
