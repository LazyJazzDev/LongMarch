#pragma once
#include "snow_mount/visualizer/visualizer_program.h"
#include "snow_mount/visualizer/visualizer_util.h"

namespace snow_mount::visualizer {

class Entity {
  friend class Core;

 public:
  virtual ~Entity() = default;
  virtual int ExecuteStage(RenderStage render_stage, RenderContext *ctx);

  static void PyBind(pybind11::module &m);

 protected:
  Entity(const std::shared_ptr<Core> &core);
  std::shared_ptr<Core> core_;
};

class EntityMeshObject : public Entity {
 public:
  EntityMeshObject(const std::shared_ptr<Core> &core,
                   const std::weak_ptr<Mesh> &mesh,
                   const Material &material,
                   const Matrix4<float> &transform);

  int ExecuteStage(RenderStage render_stage, RenderContext *ctx) override;
  static void PyBind(pybind11::module &m);

 private:
  std::shared_ptr<ProgramNoNormal> program_;
  std::weak_ptr<Mesh> mesh_;
  std::unique_ptr<graphics::Buffer> info_buffer_;
  EntityInfo info_;
};

}  // namespace snow_mount::visualizer
