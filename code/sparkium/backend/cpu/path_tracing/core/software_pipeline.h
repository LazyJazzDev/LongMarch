#pragma once
#include "sparkium/backend/cpu/cpu_bvh.h"
#include "sparkium/backend/cpu/material_codegen.h"
#include "sparkium/backend/cpu/path_tracing/core/core_util.h"

namespace sparkium::cpu_tracing {

// CPU function-JIT tracer with SAH BVH and the optional heap-BVH reference.
class SoftwarePipeline {
 public:
  explicit SoftwarePipeline(Core *core);
  void ClearInstances();
  void AddInstance(Geometry *geometry, Material *material, const glm::mat4x3 &transform, uint32_t geometry_index);
  void Update(graphics::CommandContext *commands,
              const std::vector<graphics::Buffer *> &buffers,
              uint32_t sdr_count,
              uint32_t hdr_count);

  graphics::ComputeProgram *Program() const {
    return render_program_.get();
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

  using MaterialCode = backend::cpu::MaterialCode;

  void CompileBuilders(uint32_t buffer_count);
  void CompileRenderer(const std::vector<MaterialCode> &materials,
                       uint32_t buffers,
                       uint32_t sdr_count,
                       uint32_t hdr_count);
  void AppendBuild(std::vector<BuildPass> &passes, BuildParameters parameters);

  Core *core_;
  bool cpu_{true};
  std::vector<sparkium::raytracing::CpuBvhTree> cpu_meshes_;
  std::vector<uint8_t> cpu_last_instances_;

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
}  // namespace sparkium::cpu_tracing
