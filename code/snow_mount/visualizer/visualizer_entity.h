#pragma once
#include "snow_mount/visualizer/visualizer_program.h"
#include "snow_mount/visualizer/visualizer_util.h"

namespace snow_mount::visualizer {

class Entity {
  friend class Core;

 public:
  virtual ~Entity() = default;

  std::shared_ptr<Core> GetCore() const;

  virtual int ExecuteStage(RenderStage render_stage, const RenderContext &ctx);

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

  int ExecuteStage(RenderStage render_stage, const RenderContext &ctx) override;

  void SetMesh(const std::shared_ptr<Mesh> &mesh);

  void SetMaterial(const Material &material);

  void SetTransform(const Matrix4<float> &transform);

  static void PyBind(pybind11::module &m);

 private:
  std::shared_ptr<ProgramWithGeometryShader> program_;
  std::weak_ptr<Mesh> mesh_;
  std::unique_ptr<graphics::Buffer> info_buffer_;
  EntityInfo info_;
};

class EntityAmbientLight : public Entity {
 public:
  EntityAmbientLight(const std::shared_ptr<Core> &core, const Vector3<float> &intensity);

  int ExecuteStage(RenderStage render_stage, const RenderContext &ctx) override;

  void SetIntensity(const Vector3<float> &intensity);

  static void PyBind(pybind11::module &m);

 private:
  std::shared_ptr<ProgramCommonRaster> program_;
  std::unique_ptr<graphics::Buffer> intensity_buffer_;
  Vector3<float> intensity_;
};

class EntityDirectionalLight : public Entity {
 public:
  struct LightInfo {
    Vector3<float> direction;
    float padding;
    Vector3<float> intensity;
  };

  EntityDirectionalLight(const std::shared_ptr<Core> &core,
                         const Vector3<float> &direction,
                         const Vector3<float> &intensity);

  int ExecuteStage(RenderStage render_stage, const RenderContext &ctx) override;

  void SetIntensity(const Vector3<float> &intensity);

  void SetDirection(const Vector3<float> &direction);

  static void PyBind(pybind11::module &m);

 private:
  std::shared_ptr<ProgramCommonRaster> program_;
  std::unique_ptr<graphics::Buffer> light_info_buffer_;
  Vector3<float> direction_;
  Vector3<float> intensity_;
};

}  // namespace snow_mount::visualizer
