#include "sparkium/pipelines/raytracing/core/scene.h"

namespace sparkium::raytracing {

Scene::Scene(sparkium::Scene &scene) : scene_(scene) {
}

void Scene::Render(Camera *camera, Film *film) {
  LogInfo("Rendering");
}

int32_t Scene::RegisterLight(Light *light, int custom_index) {
  return 0;
}

int32_t Scene::RegisterInstance(graphics::AccelerationStructure *blas,
                                const glm::mat4x3 &transformation,
                                int32_t hit_group_index,
                                int32_t geometry_data_index,
                                int32_t material_data_index,
                                int32_t custom_index) {
  return 0;
}

int &Scene::LightCustomIndex(int32_t light_index) {
  return sdr_image_capacity_;
}

int &Scene::InstanceCustomIndex(int32_t instance_index) {
  return sdr_image_capacity_;
}

int32_t Scene::RegisterCallableShader(graphics::Shader *callable_shader) {
  return 0;
}

int32_t Scene::RegisterBuffer(graphics::Buffer *buffer) {
  return 0;
}

int32_t Scene::RegisterImage(graphics::Image *image) {
  return 0;
}

int32_t Scene::RegisterHitGroup(const InstanceHitGroups &hit_group) {
  return 0;
}

void Scene::UpdatePipeline(Camera *camera) {
}

Scene *DedicatedCast(sparkium::Scene *scene) {
  COMPONENT_CAST(scene, Scene);
}

}  // namespace sparkium::raytracing
