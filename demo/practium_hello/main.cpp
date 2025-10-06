#include <long_march.h>

#include <random>

#include "glm/gtc/matrix_transform.hpp"

using namespace long_march;

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

int main() {
  std::unique_ptr<graphics::Core> core_;

  graphics::CreateCore(graphics::BACKEND_API_DEFAULT, graphics::Core::Settings{2, false}, &core_);
  core_->InitializeLogicalDeviceAutoSelect(false);

  practium::Core practium_core(core_.get());

  practium::Scene scene(&practium_core);

  Mesh<> cube_mesh;
  cube_mesh.LoadObjFile(FindAssetFile("meshes/cube.obj"));

  Mesh<> sphere_mesh = Mesh<>::Sphere(30);
  Mesh<> sphere_collision = Mesh<>::Sphere(10);

  Matrix<float, 3, 4> transform;
  transform << 100.0f, 0.0f, 0.0f, 0.0f, 0.0f, 100.0f, 0.0f, 0.0f, 0.0f, 0.0f, 100.0f, 0.0f;
  sparkium::MaterialPrincipled ground_surface(practium_core.GetRenderCore(), {0.8f, 0.8f, 0.8f});
  practium::ModelMesh ground_model(&practium_core, cube_mesh.Transformed(transform), &ground_surface);
  practium::MaterialPBDRigid ground_material(&practium_core, 0.0f, 0.0f, true);
  auto ground_entity = scene.AddEntity(&ground_model, &ground_material);
  auto ground_pbd_entity = dynamic_cast<practium::EntityPBDRigid *>(ground_entity.get());
  ground_pbd_entity->SetPosition({0.0f, -101.0f, 0.0f});

  sparkium::MaterialPrincipled wall_surface0(practium_core.GetRenderCore(), {0.8f, 0.8f, 0.8f});
  wall_surface0.roughness = 0.3f;
  wall_surface0.specular = 1.0f;
  ground_model.material_ = &wall_surface0;
  auto wall_entity0 = scene.AddEntity(&ground_model, &ground_material);
  auto wall_pbd_entity0 = dynamic_cast<practium::EntityPBDRigid *>(wall_entity0.get());
  wall_pbd_entity0->SetPosition({102.0f, 0.0f, 0.0f});

  sparkium::MaterialPrincipled wall_surface1(practium_core.GetRenderCore(), {0.8f, 0.8f, 0.8f});
  wall_surface1.roughness = 0.3f;
  wall_surface1.specular = 1.0f;
  ground_model.material_ = &wall_surface1;
  auto wall_entity1 = scene.AddEntity(&ground_model, &ground_material);
  auto wall_pbd_entity1 = dynamic_cast<practium::EntityPBDRigid *>(wall_entity1.get());
  wall_pbd_entity1->SetPosition({0.0f, 0.0f, -102.0f});

  const int num_spheres = 32;
  std::unique_ptr<practium::Entity> sphere_entities[num_spheres];
  std::unique_ptr<sparkium::MaterialPrincipled> sphere_surfaces[num_spheres];

  std::mt19937 rng{1234567};
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
    sphere_pbd_entity->SetPosition(position);
    sphere_pbd_entity->SetMass(0.03351f);
    sphere_pbd_entity->SetInertia(0.000536f);
    sphere_pbd_entity->SetAngularVelocity({0.0f, 0.0f, 0.0f});
  }

  graphics::Extent2D extent{640, 480};

  sparkium::Film film(practium_core.GetRenderCore(), extent.width, extent.height);
  scene.GetRenderScene()->settings.raytracing.samples_per_dispatch = 32;
  sparkium::Camera camera(
      practium_core.GetRenderCore(),
      glm::lookAt(glm::vec3{-5.0f, 0.3f, 7.0f}, glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0, 1.0, 0.0}),
      glm::radians(30.0f), static_cast<float>(film.GetWidth()) / film.GetHeight());

  std::unique_ptr<graphics::Image> srgb_image;
  core_->CreateImage(film.GetWidth(), film.GetHeight(), graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &srgb_image);

  std::unique_ptr<graphics::Window> window;
  core_->CreateWindowObject(film.GetWidth(), film.GetHeight(), "Practium Hello", &window);

  AreaLight area_light(practium_core.GetRenderCore(), glm::vec3{1.0f, 1.0f, 1.0f}, 1.0f, glm::vec3{0.0f, 30.0f, 50.0f},
                       glm::normalize(glm::vec3{0.0f, -3.0f, -5.0f}));
  area_light.emission = glm::vec3{1000.0f};
  scene.GetRenderScene()->AddEntity(area_light);

  bool ray_tracing = false;
  bool pause = true;
  window->InitImGui(nullptr, 20.0f);

  while (!window->ShouldClose()) {
    if (!pause) {
      scene.Step();
      film.info.persistence = 0.98f;
    } else {
      film.info.persistence = 1.0f;
    }

    window->BeginImGuiFrame();
    ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Once);
    ImGui::SetNextWindowBgAlpha(0.3);
    ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Checkbox("Pause", &pause);
    ImGui::Checkbox("Ray Tracing", &ray_tracing);
    if (ray_tracing && !core_->DeviceRayTracingSupport()) {
      ImGui::Text("Ray Tracing not supported on this device!");
    }
    if (ImGui::Button("Reset")) {
      for (int i = 0; i < num_spheres; i++) {
        Vector3<float> position{std::uniform_real_distribution<float>(-0.5f, 0.5f)(rng), i * 0.4f + 0.3f,
                                std::uniform_real_distribution<float>(-0.5f, 0.5f)(rng)};
        auto sphere_pbd_entity = dynamic_cast<practium::EntityPBDRigid *>(sphere_entities[i].get());
        sphere_pbd_entity->SetPosition(position);
        sphere_pbd_entity->SetVelocity({0.0f, 0.0f, 0.0f});
        sphere_pbd_entity->SetAngularVelocity({0.0f, 0.0f, 0.0f});
      }
    }
    ImGui::End();
    window->EndImGuiFrame();

    scene.SyncRenderState();
    practium_core.GetRenderCore()->Render(
        scene.GetRenderScene(), &camera, &film,
        ray_tracing ? sparkium::RENDER_PIPELINE_AUTO : sparkium::RENDER_PIPELINE_RASTERIZATION);
    film.Develop(srgb_image.get());

    std::unique_ptr<graphics::CommandContext> cmd_ctx;
    core_->CreateCommandContext(&cmd_ctx);
    cmd_ctx->CmdPresent(window.get(), srgb_image.get());
    core_->SubmitCommandContext(cmd_ctx.get());

    glfwPollEvents();
  }

  return 0;
}
