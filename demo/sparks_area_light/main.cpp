#include <long_march.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <random>

#include "glm/gtc/matrix_transform.hpp"
#include "stb_image_write.h"

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

  graphics::CreateCore(graphics::BACKEND_API_DEFAULT, graphics::Core::Settings{2, true}, &core_);
  core_->InitializeLogicalDeviceAutoSelect(true);
  sparkium::Core sparkium_core(core_.get());
  sparkium_core.GetShadersVFS().Print();

  sparkium::Scene scene(&sparkium_core);
  scene.settings.samples_per_dispatch = 8;
  sparkium::Film film(&sparkium_core, 1024, 1024);
  film.info.persistence = 0.99f;
  sparkium::Camera camera(&sparkium_core,
                          glm::lookAt(glm::vec3{0.0f, 2.0f, 7.0f}, glm::vec3{0.0f}, glm::vec3{0.0, 1.0, 0.0}),
                          glm::radians(60.0f), static_cast<float>(film.GetWidth()) / film.GetHeight());

  sparkium::MaterialLambertian material_white(&sparkium_core, {0.8f, 0.8f, 0.8f});

  Mesh<> matball_mesh;
  matball_mesh.LoadObjFile(FindAssetFile("meshes/preview_sphere.obj"));

  sparkium::GeometryMesh geometry_mesh(&sparkium_core, Mesh<>::Sphere(30));
  sparkium::GeometryMesh geometry_matball(&sparkium_core, matball_mesh);
  sparkium::EntityGeometryMaterial entity_mesh(&sparkium_core, &geometry_matball, &material_white);
  sparkium::EntityGeometryMaterial entity_shell(&sparkium_core, &geometry_mesh, &material_white,
                                                glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, -1001.0f, 0.0f}) *
                                                    glm::scale(glm::mat4(1.0f), glm::vec3(1000.0f)));
  sparkium::EntityPointLight entity_point_light(&sparkium_core, glm::vec3{0.0f, 2.0f, 0.0f}, glm::vec3{1.0f}, 10.0f);
  entity_point_light.position = {0.0f, 5.0f, 0.0f};
  entity_point_light.strength = 100.0f;

  const int num_lights = 6;
  std::unique_ptr<AreaLight> area_lights[num_lights];
  glm::vec3 positions[num_lights];

  for (int i = 0; i < num_lights; ++i) {
    area_lights[i] =
        std::make_unique<AreaLight>(&sparkium_core, glm::vec3{1.0f, 1.0f, 1.0f}, 1.0f, glm::vec3{0.0f, 0.0f, 0.0f});
    scene.AddEntity(*area_lights[i]);
    float frac = float(i) / num_lights;
    float theta = 2.0f * std::acos(-1.0f) * frac;
    float sin_theta = std::sin(theta);
    float cos_theta = std::cos(theta);
    float x = 3.0f;
    positions[i] = {sin_theta * x, 2.0f, cos_theta * x};
    area_lights[i]->size = 0.6f;
    area_lights[i]->emission = graphics::HSVtoRGB({frac, 1.0f, 1.0f});
    area_lights[i]->emission =
        area_lights[i]->emission * (20.0f / num_lights / graphics::GreyScale(area_lights[i]->emission) /
                                    area_lights[i]->size / area_lights[i]->size);
    area_lights[i]->Sync();
  }

  scene.AddEntity(&entity_mesh);
  scene.AddEntity(&entity_shell);

  std::unique_ptr<graphics::Image> srgb_image;
  core_->CreateImage(film.GetWidth(), film.GetHeight(), graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &srgb_image);

  std::unique_ptr<graphics::Window> window;
  core_->CreateWindowObject(film.GetWidth(), film.GetHeight(), "Sparkium", &window);
  FPSCounter fps_counter;
  float rotation_angle = 0.0f;
  while (!window->ShouldClose()) {
    for (int i = 0; i < num_lights; i++) {
      float frac = float(i) / num_lights;
      float theta = 2.0f * std::acos(-1.0f) * frac + rotation_angle;
      float sin_theta = std::sin(theta);
      float cos_theta = std::cos(theta);
      float x = 6.0f;
      positions[i] = {sin_theta * x, 4.0f, cos_theta * x};
      area_lights[i]->position = positions[i];
      area_lights[i]->direction = -area_lights[i]->position;
      // area_lights[i]->emission = glm::vec3{1.0f, 0.0f, 0.0f} * 0.0f;
      area_lights[i]->Sync();
    }
    rotation_angle += glm::radians(0.3f);
    sparkium_core.Render(&scene, &camera, &film);
    film.Develop(srgb_image.get());
    std::unique_ptr<graphics::CommandContext> cmd_context;
    core_->CreateCommandContext(&cmd_context);
    cmd_context->CmdPresent(window.get(), srgb_image.get());
    core_->SubmitCommandContext(cmd_context.get());
    glfwPollEvents();
    float fps = fps_counter.TickFPS();
    char fps_buf[16];
    sprintf(fps_buf, "%.2f", fps);
    float rps = film.GetWidth() * film.GetHeight() * fps * scene.settings.samples_per_dispatch;
    char rps_buf[16];
    sprintf(rps_buf, "%.2f", rps * 1e-6f);
    window->SetTitle(std::string("Sparkium Area Light - ") + fps_buf + "frames/s" + " - " + rps_buf + "Mrays/s");
  }

  film.Develop(srgb_image.get());
  std::vector<uint8_t> image_data(film.GetWidth() * film.GetHeight() * 4);
  srgb_image->DownloadData(image_data.data());
  stbi_write_bmp("output.bmp", film.GetWidth(), film.GetHeight(), 4, image_data.data());
}
