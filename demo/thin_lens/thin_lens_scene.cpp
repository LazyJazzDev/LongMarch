#include "thin_lens_scene.h"

#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

using namespace grassland;

namespace {
Mesh<> FloorMesh() {
  const std::vector<Vector3<float>> positions{{-0.5f, 0, -0.5f}, {-0.5f, 0, 0.5f}, {0.5f, 0, 0.5f}, {0.5f, 0, -0.5f}};
  const uint32_t indices[]{0, 1, 2, 0, 2, 3};
  return Mesh<>(4, 6, indices, positions.data());
}

glm::mat4 View() {
  return glm::lookAt(glm::vec3(0, 1.5f, 6), glm::vec3(0, 1.5f, 0), glm::vec3(0, 1, 0));
}
}  // namespace

std::unique_ptr<ThinLensScene> ThinLensScene::Create(rhi::IDevice *device, int width, int height) {
  auto core = sparkium::Core::Create(device);
  if (!core || width <= 0 || height <= 0)
    return {};
  auto scene = std::unique_ptr<ThinLensScene>(new ThinLensScene(std::move(core), width, height));
  return scene->film_.IsValid() && scene->renderer_.IsValid() ? std::move(scene) : nullptr;
}

ThinLensScene::ThinLensScene(std::unique_ptr<sparkium::Core> core, int width, int height)
    : core_(std::move(core)),
      scene_(core_.get()),
      sphere_(core_.get(), Mesh<>::Sphere(32, 16)),
      tile_(core_.get(), FloorMesh()),
      film_(core_.get(), width, height),
      lens_(core_.get(), View(), glm::radians(45.0f), float(width) / height),
      pinhole_(core_.get(), View(), lens_.fovy, lens_.aspect),
      renderer_(core_.get()) {
  auto *device = core_->GetDevice();
  pipeline_ = device->SupportRayQuery()             ? sparkium::RENDER_PIPELINE_RAY_QUERY
              : device->SupportRayTracingPipeline() ? sparkium::RENDER_PIPELINE_RAY_TRACING
                                                    : sparkium::RENDER_PIPELINE_RT_FALLBACK;
  scene_.settings.max_bounces = 4;
  scene_.settings.background_color = glm::vec3(0.015f, 0.022f, 0.04f);
  film_.info.view_transform = 1;
  auto diffuse = [&](glm::vec3 color) -> sparkium::Material * {
    auto material = std::make_unique<sparkium::MaterialPrincipled>(core_.get(), color);
    material->roughness = 0.3f;
    material->specular = 0.4f;
    auto *ptr = material.get();
    materials_.push_back(std::move(material));
    return ptr;
  };
  auto light = [&](glm::vec3 color) -> sparkium::Material * {
    auto material = std::make_unique<sparkium::MaterialLight>(core_.get(), color, true);
    auto *ptr = material.get();
    materials_.push_back(std::move(material));
    return ptr;
  };
  auto add = [&](sparkium::Geometry *geometry, sparkium::Material *material, glm::vec3 position, glm::vec3 scale) {
    const auto transform = glm::scale(glm::translate(glm::mat4(1), position), scale);
    auto entity =
        std::make_unique<sparkium::EntityGeometryMaterial>(core_.get(), geometry, material, glm::mat4x3(transform));
    scene_.AddEntity(entity.get());
    entities_.push_back(std::move(entity));
  };
  auto *dark = diffuse({0.045f, 0.06f, 0.09f});
  auto *pale = diffuse({0.24f, 0.29f, 0.36f});
  for (int z = -9; z <= 4; ++z)
    for (int x = -7; x <= 7; ++x)
      add(&tile_, (x + z) % 2 ? dark : pale, {float(x), 0, float(z)}, glm::vec3(1));
  const glm::vec3 positions[]{{-1.25f, 0.65f, 2}, {0, 0.65f, 0}, {2.0f, 0.65f, -3}};
  const glm::vec3 colors[]{{0.9f, 0.16f, 0.055f}, {0.04f, 0.65f, 0.52f}, {0.08f, 0.3f, 0.9f}};
  for (int i = 0; i < 3; ++i)
    add(&sphere_, diffuse(colors[i]), positions[i], glm::vec3(0.65f));
  add(&sphere_, light({18, 16, 13}), {-3, 6, 2}, glm::vec3(2));
  auto *warm = light({12, 4, 1});
  auto *cool = light({1, 5, 12});
  for (int row = 0; row < 3; ++row)
    for (int x = -5; x <= 5; ++x)
      add(&sphere_, (x + row) % 2 ? warm : cool, {float(x) * 1.1f, 1.2f + row * 0.85f, -7.0f}, glm::vec3(0.035f));
}

bool ThinLensScene::Configure(const Settings &settings) {
  if (!std::isfinite(settings.focus_distance) || settings.focus_distance <= 0 ||
      !std::isfinite(settings.aperture_radius) || settings.aperture_radius < 0 ||
      !std::isfinite(settings.aperture_ratio) || settings.aperture_ratio <= 0 ||
      !std::isfinite(settings.aperture_rotation) || !std::isfinite(settings.geometry_offset) ||
      settings.geometry_offset < 0 || settings.geometry_offset > 1 ||
      (settings.aperture_blades != 0 && (settings.aperture_blades < 3 || settings.aperture_blades > 8)))
    return false;
  pinhole_mode_ = settings.pinhole_mode;
  lens_.focus_distance = settings.focus_distance;
  lens_.aperture_radius = settings.aperture_radius;
  lens_.aperture_blades = settings.aperture_blades;
  lens_.aperture_rotation = settings.aperture_rotation;
  lens_.aperture_ratio = settings.aperture_ratio;
  sphere_.SetShadowTerminatorGeometryOffset(settings.geometry_offset);
  film_.Reset();
  return core_->IsValid();
}

sparkium::Film *ThinLensScene::GetFilm() {
  return &film_;
}

rhi::ObjPtr<rhi::ISubmissionToken> ThinLensScene::Submit(int samples,
                                                         rhi::IImage *output,
                                                         bool hdr,
                                                         const rhi::SubmitInfo &dependencies) {
  scene_.settings.samples_per_dispatch = samples;
  return renderer_.Submit(&scene_, pinhole_mode_ ? static_cast<sparkium::Camera *>(&pinhole_) : &lens_, &film_,
                          pipeline_, output, hdr, dependencies);
}
