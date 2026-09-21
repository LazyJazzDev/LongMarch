#include "sparkium/scene/scene_definition.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace sparkium {
namespace {
void Require(bool condition, const char *message) {
  if (!condition)
    throw std::invalid_argument(message);
}

void ValidateTexture(const std::shared_ptr<const TextureData> &texture) {
  Require(texture && texture->width > 0 && texture->height > 0, "invalid texture dimensions");
  Require(uint64_t(texture->width) * texture->height <= std::numeric_limits<size_t>::max() / 4,
          "texture dimensions overflow");
  Require(texture->rgba.size() == size_t(texture->width) * texture->height * 4, "invalid texture pixel count");
}
}  // namespace

void SceneDefinition::Validate() const {
  Require(film.width > 0 && film.height > 0, "film dimensions must be positive");
  Require(uint64_t(film.width) * film.height <= std::numeric_limits<size_t>::max() / 16, "film dimensions overflow");
  Require(integrator.samples_per_dispatch > 0 && integrator.max_bounces > 0, "invalid integrator sample counts");
  Require(camera.fovy > 0 && camera.fovy < grassland::PI<float>() && camera.aspect > 0, "invalid camera projection");
  Require(std::isfinite(glm::determinant(camera.view)) && std::abs(glm::determinant(camera.view)) > 1e-20f,
          "invalid camera transform");
  for (const auto &[id, geometry] : geometries) {
    Require(bool(geometry), "null geometry");
    if (auto mesh = std::get_if<grassland::Mesh<float>>(&geometry->shape)) {
      Require(mesh->NumIndices() % 3 == 0, "incomplete mesh triangles");
      for (size_t i = 0; i < mesh->NumIndices(); ++i)
        Require(mesh->Indices()[i] < mesh->NumVertices(), "mesh index out of bounds");
    } else {
      const auto &hair = std::get<HairData>(geometry->shape);
      Require(hair.radial_segments >= 3 && hair.points.size() == hair.radii.size() && hair.offsets.size() >= 2,
              "invalid hair geometry");
      Require(hair.offsets.front() == 0 && hair.offsets.back() == hair.points.size(), "invalid hair offsets");
      for (size_t i = 1; i < hair.offsets.size(); ++i)
        Require(hair.offsets[i] > hair.offsets[i - 1] + 1 && hair.offsets[i] <= hair.points.size(),
                "invalid hair strand");
    }
  }
  for (const auto &[id, material] : materials) {
    for (const auto &[slot, texture] : material.textures)
      ValidateTexture(texture);
    for (const auto &[node, texture] : material.graph_textures)
      ValidateTexture(texture);
  }
  for (const auto &entity : entities)
    if (const auto *instance = std::get_if<InstanceDefinition>(&entity)) {
      Require(geometries.count(instance->geometry) && materials.count(instance->material),
              "instance references missing geometry or material");
      Require(std::isfinite(glm::determinant(instance->transform)) &&
                  std::abs(glm::determinant(instance->transform)) > 1e-20f,
              "invalid instance transform");
    }
}
}  // namespace sparkium
