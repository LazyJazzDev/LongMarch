#include <long_march.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <random>

#include "glm/gtc/matrix_transform.hpp"
#include "stb_image_write.h"

using namespace long_march;

int main() {
  std::unique_ptr<graphics::Core> core_;

  graphics::CreateCore(graphics::BACKEND_API_DEFAULT, graphics::Core::Settings{2, false}, &core_);
  core_->InitializeLogicalDeviceAutoSelect(true);
  sparkium::Core sparkium_core(core_.get());

  sparkium::Scene scene(&sparkium_core);
  scene.settings.samples_per_dispatch = 32;
  sparkium::Film film(&sparkium_core, 1024, 1024);
  film.info.persistence = 0.98f;
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

  const int num_lights = 12;
  std::unique_ptr<sparkium::EntityPointLight> point_lights[num_lights];
  glm::vec3 positions[num_lights];

  for (int i = 0; i < num_lights; ++i) {
    point_lights[i] = std::make_unique<sparkium::EntityPointLight>(&sparkium_core, glm::vec3{}, glm::vec3{1.0f}, 1.0f);
    scene.AddEntity(point_lights[i].get());
    float frac = float(i) / num_lights;
    float theta = 2.0f * std::acos(-1.0f) * frac;
    float sin_theta = std::sin(theta);
    float cos_theta = std::cos(theta);
    float x = 3.0f;
    positions[i] = {sin_theta * x, 2.0f, cos_theta * x};
    point_lights[i]->color = graphics::HSVtoRGB({frac, 1.0f, 1.0f});
    point_lights[i]->strength = 1440.0f / num_lights / graphics::GreyScale(point_lights[i]->color);
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
      point_lights[i]->position = positions[i];
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
    window->SetTitle(std::string("Sparkium Point Light - ") + fps_buf + "frames/s" + " - " + rps_buf + "Mrays/s");
  }

  film.Develop(srgb_image.get());
  std::vector<uint8_t> image_data(film.GetWidth() * film.GetHeight() * 4);
  srgb_image->DownloadData(image_data.data());
  stbi_write_bmp("output.bmp", film.GetWidth(), film.GetHeight(), 4, image_data.data());
}
