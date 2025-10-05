#include <long_march.h>

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

  Matrix<float, 3, 4> transform;
  transform << 100.0f, 0.0f, 0.0f, 0.0f, 0.0f, 100.0f, 0.0f, -101.0f, 0.0f, 0.0f, 100.0f, 0.0f;
  sparkium::MaterialPrincipled ground_surface(practium_core.GetRenderCore(), {0.8f, 0.8f, 0.8f});
  practium::ModelMesh ground_model(&practium_core, cube_mesh.Transformed(transform), &ground_surface);
  practium::MaterialPBDRigid ground_material(&practium_core, 0.0f, 0.0f, true);
  auto ground_entity = scene.AddEntity(&ground_model, &ground_material);

  transform << 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f;
  sparkium::MaterialPrincipled box_surface(practium_core.GetRenderCore(), {0.8f, 0.0f, 0.0f});
  practium::ModelMesh box_model(&practium_core, cube_mesh.Transformed(transform), &box_surface);
  practium::MaterialPBDRigid box_material(&practium_core, 0.0f, 0.0f, true);
  auto box_entity = scene.AddEntity(&box_model, &box_material);

  graphics::Extent2D extent{1280, 720};

  sparkium::Film film(practium_core.GetRenderCore(), extent.width, extent.height);
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

  while (!window->ShouldClose()) {
    practium_core.GetRenderCore()->Render(scene.GetRenderScene(), &camera, &film);
    film.Develop(srgb_image.get());

    std::unique_ptr<graphics::CommandContext> cmd_ctx;
    core_->CreateCommandContext(&cmd_ctx);
    cmd_ctx->CmdPresent(window.get(), srgb_image.get());
    core_->SubmitCommandContext(cmd_ctx.get());

    glfwPollEvents();
  }

  return 0;
}
