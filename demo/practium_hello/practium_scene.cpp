#include "practium_scene.h"

#include <algorithm>
#include <array>
#include <random>

#include "glm/gtc/matrix_transform.hpp"
#include "practium/practium.h"
using namespace grassland;

namespace {
class AreaLight {
  const uint32_t indices[6] = {0, 2, 1, 0, 3, 2};
  const Vector3<float> vertices[4] = {{-1.0f, -1.0f, 0.0f},
                                      {1.0f, -1.0f, 0.0f},
                                      {1.0f, 1.0f, 0.0f},
                                      {-1.0f, 1.0f, 0.0f}};

 public:
  AreaLight(sparkium::Core *core,
            const glm::vec3 &emission = {1.0f, 1.0f, 1.0f},
            float size = 1.0f,
            const glm::vec3 &position = {0.0f, 0.0f, 0.0f},
            const glm::vec3 &direction = {0.0f, 0.0f, 1.0f},
            const glm::vec3 &up = {0.0f, 1.0f, 0.0f})
      : light_(core, emission, false, false),
        emission(light_.emission),
        position(position),
        size(size),
        direction(direction),
        up(up) {
    mesh_ = std::make_unique<sparkium::GeometryMesh>(core, Mesh<>{4, 6, indices, vertices});
    entity_geometry_material_ = std::make_unique<sparkium::EntityGeometryMaterial>(core, mesh_.get(), &light_);
    Sync();
  }

  void Sync() {
    entity_geometry_material_->SetTransformation(glm::inverse(glm::lookAt(position, position + direction, up)) *
                                                 glm::scale(glm::mat4{1.0f}, glm::vec3{size}));
  }

  operator sparkium::Entity *() {
    return entity_geometry_material_.get();
  }

  glm::vec3 &emission;
  float size{1.0f};
  glm::vec3 position{0.0f, 0.0f, 0.0f};
  glm::vec3 direction{0.0f, -1.0f, 0.0f};
  glm::vec3 up{0.0f, 1.0f, 0.0f};

  std::unique_ptr<sparkium::EntityGeometryMaterial> entity_geometry_material_;

 private:
  std::unique_ptr<sparkium::GeometryMesh> mesh_;
  sparkium::MaterialLight light_;
};

}  // namespace

struct PractiumDemoScene::State {
  static constexpr int num_spheres = 32;
  std::unique_ptr<practium::Core> owner;
  practium::Scene scene;
  std::array<std::unique_ptr<sparkium::MaterialPrincipled>, 3> ground_surfaces;
  std::array<std::unique_ptr<sparkium::MaterialPrincipled>, num_spheres> sphere_surfaces;
  std::array<std::unique_ptr<practium::Entity>, 3> ground_entities;
  std::array<std::unique_ptr<practium::Entity>, num_spheres> sphere_entities;
  std::unique_ptr<AreaLight> area_light;
  std::unique_ptr<sparkium::Film> film;
  std::unique_ptr<sparkium::CameraPinhole> camera;
  std::unique_ptr<sparkium::Renderer> renderer;
  std::mt19937 rng{1234567};
  Settings settings;
  int remaining_samples{};

  explicit State(std::unique_ptr<practium::Core> core) : owner(std::move(core)), scene(owner.get()) {
  }

  bool Load(int width, int height) {
    auto &practium_core = *owner;
    Mesh<> cube_mesh;
    if (cube_mesh.LoadObjFile(FindAssetFile("meshes/cube.obj")) != 0)
      return false;

    Mesh<> sphere_mesh = Mesh<>::Sphere(30);
    Mesh<> sphere_collision = Mesh<>::Sphere(10);

    Matrix<float, 3, 4> transform;
    transform << 100.0f, 0.0f, 0.0f, 0.0f, 0.0f, 100.0f, 0.0f, 0.0f, 0.0f, 0.0f, 100.0f, 0.0f;
    auto &ground_surface = *(ground_surfaces[0] = std::make_unique<sparkium::MaterialPrincipled>(
                                 practium_core.GetRenderCore(), glm::vec3{0.8f}));
    practium::ModelMesh ground_model(&practium_core, cube_mesh.Transformed(transform), &ground_surface);
    practium::MaterialPBDRigid ground_material(&practium_core, 0.0f, 0.0f, true);
    auto &ground_entity = ground_entities[0];
    ground_entity = scene.AddEntity(&ground_model, &ground_material);
    auto ground_pbd_entity = dynamic_cast<practium::EntityPBDRigid *>(ground_entity.get());
    if (!ground_pbd_entity)
      return false;
    ground_pbd_entity->SetPosition({0.0f, -101.0f, 0.0f});

    auto &wall_surface0 = *(ground_surfaces[1] = std::make_unique<sparkium::MaterialPrincipled>(
                                practium_core.GetRenderCore(), glm::vec3{0.8f}));
    wall_surface0.roughness = 0.3f;
    wall_surface0.specular = 1.0f;
    ground_model.material_ = &wall_surface0;
    auto &wall_entity0 = ground_entities[1];
    wall_entity0 = scene.AddEntity(&ground_model, &ground_material);
    auto wall_pbd_entity0 = dynamic_cast<practium::EntityPBDRigid *>(wall_entity0.get());
    if (!wall_pbd_entity0)
      return false;
    wall_pbd_entity0->SetPosition({102.0f, 0.0f, 0.0f});

    auto &wall_surface1 = *(ground_surfaces[2] = std::make_unique<sparkium::MaterialPrincipled>(
                                practium_core.GetRenderCore(), glm::vec3{0.8f}));
    wall_surface1.roughness = 0.3f;
    wall_surface1.specular = 1.0f;
    ground_model.material_ = &wall_surface1;
    auto &wall_entity1 = ground_entities[2];
    wall_entity1 = scene.AddEntity(&ground_model, &ground_material);
    auto wall_pbd_entity1 = dynamic_cast<practium::EntityPBDRigid *>(wall_entity1.get());
    if (!wall_pbd_entity1)
      return false;
    wall_pbd_entity1->SetPosition({0.0f, 0.0f, -102.0f});

    for (int i = 0; i < num_spheres; i++) {
      Vector3<float> position{std::uniform_real_distribution<float>(-0.5f, 0.5f)(rng), i * 0.4f + 0.3f,
                              std::uniform_real_distribution<float>(-0.5f, 0.5f)(rng)};
      transform << 0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.0f;

      sphere_surfaces[i] = std::make_unique<sparkium::MaterialPrincipled>(
          practium_core.GetRenderCore(), glm::vec3{std::uniform_real_distribution<float>(0.0f, 1.0f)(rng),
                                                   std::uniform_real_distribution<float>(0.0f, 1.0f)(rng),
                                                   std::uniform_real_distribution<float>(0.0f, 1.0f)(rng)} *
                                                 0.5f +
                                             0.5f);
      sphere_surfaces[i]->roughness = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
      sphere_surfaces[i]->metallic = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
      sphere_surfaces[i]->specular = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
      if ((i & 3) == 3) {
        sphere_surfaces[i]->emission_color = sphere_surfaces[i]->base_color;
        sphere_surfaces[i]->emission_strength = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) * 5.0f;
      }

      practium::ModelMesh sphere_model(&practium_core, sphere_mesh.Transformed(transform),
                                       sphere_collision.Transformed(transform), sphere_surfaces[i].get());
      practium::MaterialPBDRigid sphere_material(&practium_core);
      auto &sphere_entity = sphere_entities[i];
      sphere_entity = scene.AddEntity(&sphere_model, &sphere_material);
      // dynamic cast to EntityPBDRigid
      auto sphere_pbd_entity = dynamic_cast<practium::EntityPBDRigid *>(sphere_entity.get());
      if (!sphere_pbd_entity)
        return false;
      sphere_pbd_entity->SetPosition(position);
      sphere_pbd_entity->SetMass(0.03351f);
      sphere_pbd_entity->SetInertia(0.000536f);
      sphere_pbd_entity->SetAngularVelocity({0.0f, 0.0f, 0.0f});
    }

    film = std::make_unique<sparkium::Film>(practium_core.GetRenderCore(), width, height);
    scene.GetRenderScene()->settings.raytracing.samples_per_dispatch = 32;
    camera = std::make_unique<sparkium::CameraPinhole>(
        practium_core.GetRenderCore(),
        glm::lookAt(glm::vec3{-5.0f, 0.3f, 7.0f}, glm::vec3{0.0f}, glm::vec3{0.0, 1.0, 0.0}), glm::radians(30.0f),
        static_cast<float>(width) / height);
    area_light =
        std::make_unique<AreaLight>(practium_core.GetRenderCore(), glm::vec3{1.0f}, 1.0f, glm::vec3{0.0f, 30.0f, 50.0f},
                                    glm::normalize(glm::vec3{0.0f, -3.0f, -5.0f}));
    area_light->emission = glm::vec3{1000.0f};
    scene.GetRenderScene()->AddEntity(*area_light);
    renderer = std::make_unique<sparkium::Renderer>(practium_core.GetRenderCore());
    return film->IsValid() && renderer->IsValid();
  }

  bool Configure(const Settings &next) {
    if (next.reset_sequence < settings.reset_sequence)
      return false;
    // A snapshot may coalesce multiple button presses; retain each RNG advance.
    while (settings.reset_sequence < next.reset_sequence) {
      for (int i = 0; i < num_spheres; i++) {
        Vector3<float> position{std::uniform_real_distribution<float>(-0.5f, 0.5f)(rng), i * 0.4f + 0.3f,
                                std::uniform_real_distribution<float>(-0.5f, 0.5f)(rng)};
        auto sphere_pbd_entity = dynamic_cast<practium::EntityPBDRigid *>(sphere_entities[i].get());
        sphere_pbd_entity->SetPosition(position);
        sphere_pbd_entity->SetVelocity({0.0f, 0.0f, 0.0f});
        sphere_pbd_entity->SetAngularVelocity({0.0f, 0.0f, 0.0f});
      }
      ++settings.reset_sequence;
    }
    scene.GetRenderScene()->SetEntityActive(*area_light, next.day_light);
    settings = next;
    remaining_samples = 0;
    return owner->GetRenderCore()->IsValid();
  }

  rhi::ObjPtr<rhi::ISubmissionToken> Submit(int samples,
                                            rhi::IImage *output,
                                            bool hdr,
                                            const rhi::SubmitInfo &dependencies) {
    if (samples <= 0)
      return {};
    if (!remaining_samples) {
      // One original render cycle advances physics once, even when its 32
      // tracing samples are split across several bounded GPU submissions.
      if (!settings.pause)
        scene.Step();
      film->info.persistence = settings.pause ? 1.0f : 0.98f;
      scene.SyncRenderState();
      remaining_samples = settings.ray_tracing ? 32 : 1;
    }
    const int batch_samples = std::min(samples, remaining_samples);
    scene.GetRenderScene()->settings.samples_per_dispatch = batch_samples;
    auto submitted = renderer->Submit(scene.GetRenderScene(), camera.get(), film.get(),
                                      settings.ray_tracing ? (owner->GetRenderCore()->GetDevice()->SupportRayQuery()
                                                                  ? sparkium::RENDER_PIPELINE_RAY_QUERY
                                                                  : sparkium::RENDER_PIPELINE_RT_FALLBACK)
                                                           : sparkium::RENDER_PIPELINE_RASTERIZATION,
                                      output, hdr, dependencies);
    if (submitted && owner->GetRenderCore()->IsValid())
      remaining_samples -= batch_samples;
    return submitted;
  }
};

std::unique_ptr<PractiumDemoScene> PractiumDemoScene::Create(rhi::IDevice *device, int width, int height) {
  if (width <= 0 || height <= 0)
    return {};
  auto core = practium::Core::Create(device);
  if (!core)
    return {};
  auto state = std::make_unique<State>(std::move(core));
  if (!state->Load(width, height))
    return {};
  return std::unique_ptr<PractiumDemoScene>(new PractiumDemoScene(std::move(state)));
}

PractiumDemoScene::PractiumDemoScene(std::unique_ptr<State> state) : state_(std::move(state)) {
}

PractiumDemoScene::~PractiumDemoScene() = default;

bool PractiumDemoScene::Configure(const Settings &settings) {
  return state_->Configure(settings);
}

sparkium::Film *PractiumDemoScene::GetFilm() {
  return state_->film.get();
}

rhi::ObjPtr<rhi::ISubmissionToken> PractiumDemoScene::Submit(int samples,
                                                             rhi::IImage *output,
                                                             bool hdr,
                                                             const rhi::SubmitInfo &dependencies) {
  return state_->Submit(samples, output, hdr, dependencies);
}
