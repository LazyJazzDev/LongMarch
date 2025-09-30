#include <long_march.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "glm/gtc/matrix_transform.hpp"
#include "stb_image_write.h"

using namespace long_march;

int main() {
  std::unique_ptr<graphics::Core> core_;

  graphics::CreateCore(graphics::BACKEND_API_DEFAULT, graphics::Core::Settings{2}, &core_);
  core_->InitializeLogicalDeviceAutoSelect(true);
  sparkium::Core sparkium_core(core_.get());
  sparkium_core.GetShadersVFS().Print();

  sparkium::Scene scene(&sparkium_core);
  scene.settings.samples_per_dispatch = 32;
  sparkium::Film film(&sparkium_core, 1024, 1024);
  sparkium::Camera camera(&sparkium_core,
                          glm::lookAt(glm::vec3{278.0f, 273.0f, -800.0f}, glm::vec3{278.0f, 273.0f, -800.0f + 1.0f},
                                      glm::vec3{0.0, 1.0, 0.0}),
                          glm::radians(40.0f), static_cast<float>(film.GetWidth()) / film.GetHeight());

  sparkium::MaterialLambertian material_white(&sparkium_core, {0.725, 0.71, 0.68});
  sparkium::MaterialLambertian material_red(&sparkium_core, {0.63, 0.065, 0.05});
  sparkium::MaterialLambertian material_green(&sparkium_core, {0.14, 0.45, 0.091});
  sparkium::MaterialLambertian material_light(&sparkium_core, {0.0f, 0.0f, 0.0f}, {30.0f, 30.0f, 30.0f});
  sparkium::MaterialSpecular material_specular(&sparkium_core, {0.8f, 0.8f, 0.8f});

  std::vector<glm::vec3> positions;
  std::vector<glm::vec2> tex_coords;
  std::vector<uint32_t> indices;

  positions = {{343.0f, 548.7f, 227.0f}, {343.0f, 548.7f, 332.0f}, {213.0f, 548.7f, 332.0f}, {213.0f, 548.7f, 227.0f}};
  tex_coords = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
  indices = {0, 1, 3, 1, 2, 3};
  Mesh<float> light(positions.size(), indices.size(), indices.data(),
                    reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                    reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  sparkium::GeometryMesh geometry_light(&sparkium_core, light);
  sparkium::EntityGeometryMaterial entity_light(&sparkium_core, &geometry_light, &material_light);
  // sparkium::EntityGeometryLight entity_light(&sparkium_core, &geometry_light, {30.0f, 30.0f, 30.0f}, true, true,
  //                                          glm::mat4x3(1.0f));

  positions = {{552.8f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 559.2f}, {549.6f, 0.0f, 559.2f}};
  Mesh<float> floor(positions.size(), indices.size(), indices.data(),
                    reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                    reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  sparkium::GeometryMesh geometry_floor(&sparkium_core, floor);
  sparkium::EntityGeometryMaterial entity_floor(&sparkium_core, &geometry_floor, &material_white);

  positions = {{556.0f, 548.8f, 0.0f}, {556.0f, 548.8f, 559.2f}, {0.0f, 548.8f, 559.2f}, {0.0f, 548.8f, 0.0f}};
  Mesh<float> ceiling(positions.size(), indices.size(), indices.data(),
                      reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                      reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  sparkium::GeometryMesh geometry_ceiling(&sparkium_core, ceiling);
  sparkium::EntityGeometryMaterial entity_ceiling(&sparkium_core, &geometry_ceiling, &material_white);

  positions = {{549.6f, 0.0f, 559.2f}, {0.0f, 0.0f, 559.2f}, {0.0f, 548.8f, 559.2f}, {556.0f, 548.8f, 559.2f}};
  Mesh<float> back_wall(positions.size(), indices.size(), indices.data(),
                        reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                        reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  sparkium::GeometryMesh geometry_back_wall(&sparkium_core, back_wall);
  sparkium::EntityGeometryMaterial entity_back_wall(&sparkium_core, &geometry_back_wall, &material_white);

  positions = {{0.0f, 0.0f, 559.2f}, {0.0f, 0.0f, 0.0f}, {0.0f, 548.8f, 0.0f}, {0.0f, 548.8f, 559.2f}};
  Mesh<float> right_wall(positions.size(), indices.size(), indices.data(),
                         reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                         reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  sparkium ::GeometryMesh geometry_right_wall(&sparkium_core, right_wall);
  sparkium::EntityGeometryMaterial entity_right_wall(&sparkium_core, &geometry_right_wall, &material_green);

  positions = {{552.8f, 0.0f, 0.0f}, {549.6f, 0.0f, 559.2f}, {556.0f, 548.8f, 559.2f}, {556.0f, 548.8f, 0.0f}};
  Mesh<float> left_wall(positions.size(), indices.size(), indices.data(),
                        reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                        reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  sparkium::GeometryMesh geometry_left_wall(&sparkium_core, left_wall);
  sparkium::EntityGeometryMaterial entity_left_wall(&sparkium_core, &geometry_left_wall, &material_red);

  indices = {0, 1, 3, 1, 2, 3, 4, 5, 7, 5, 6, 7, 8, 9, 11, 9, 10, 11, 12, 13, 15, 13, 14, 15, 16, 17, 19, 17, 18, 19};
  positions = {{130.0f, 165.0f, 65.0f}, {82.0f, 165.0f, 225.0f},  {240.0f, 165.0f, 272.0f}, {290.0f, 165.0f, 114.0f},
               {290.0f, 0.0f, 114.0f},  {290.0f, 165.0f, 114.0f}, {240.0f, 165.0f, 272.0f}, {240.0f, 0.0f, 272.0f},
               {130.0f, 0.0f, 65.0f},   {130.0f, 165.0f, 65.0f},  {290.0f, 165.0f, 114.0f}, {290.0f, 0.0f, 114.0f},
               {82.0f, 0.0f, 225.0f},   {82.0f, 165.0f, 225.0f},  {130.0f, 165.0f, 65.0f},  {130.0f, 0.0f, 65.0f},
               {240.0f, 0.0f, 272.0f},  {240.0f, 165.0f, 272.0f}, {82.0f, 165.0f, 225.0f},  {82.0f, 0.0f, 225.0f}};
  tex_coords = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f},
                {0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f},
                {1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
  sparkium::GeometryMesh short_box(&sparkium_core,
                                   Mesh<float>(positions.size(), indices.size(), indices.data(),
                                               reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                                               reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr));
  sparkium::EntityGeometryMaterial entity_short_box(&sparkium_core, &short_box, &material_white);

  positions = {{423.0f, 330.0f, 247.0f}, {265.0f, 330.0f, 296.0f}, {314.0f, 330.0f, 456.0f}, {472.0f, 330.0f, 406.0f},
               {423.0f, 0.0f, 247.0f},   {423.0f, 330.0f, 247.0f}, {472.0f, 330.0f, 406.0f}, {472.0f, 0.0f, 406.0f},
               {472.0f, 0.0f, 406.0f},   {472.0f, 330.0f, 406.0f}, {314.0f, 330.0f, 456.0f}, {314.0f, 0.0f, 456.0f},
               {314.0f, 0.0f, 456.0f},   {314.0f, 330.0f, 456.0f}, {265.0f, 330.0f, 296.0f}, {265.0f, 0.0f, 296.0f},
               {265.0f, 0.0f, 296.0f},   {265.0f, 330.0f, 296.0f}, {423.0f, 330.0f, 247.0f}, {423.0f, 0.0f, 247.0f}};
  Mesh<float> tall_box(positions.size(), indices.size(), indices.data(),
                       reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                       reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  sparkium::GeometryMesh geometry_tall_box(&sparkium_core, tall_box);
  sparkium::EntityGeometryMaterial entity_tall_box(&sparkium_core, &geometry_tall_box, &material_specular);
  scene.AddEntity(&entity_light);
  scene.AddEntity(&entity_floor);
  scene.AddEntity(&entity_ceiling);
  scene.AddEntity(&entity_back_wall);
  scene.AddEntity(&entity_right_wall);
  scene.AddEntity(&entity_left_wall);
  scene.AddEntity(&entity_short_box);
  scene.AddEntity(&entity_tall_box);

  std::unique_ptr<graphics::Image> srgb_image;
  core_->CreateImage(film.GetWidth(), film.GetHeight(), graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &srgb_image);

  std::unique_ptr<graphics::Window> window;
  core_->CreateWindowObject(film.GetWidth(), film.GetHeight(), "Sparkium Cornell Box", &window);
  FPSCounter fps_counter;
  while (!window->ShouldClose()) {
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
    window->SetTitle(std::string("Sparkium Specular Surface - ") + fps_buf + "frames/s" + " - " + rps_buf + "Mrays/s");
  }

  film.Develop(srgb_image.get());
  std::vector<uint8_t> image_data(film.GetWidth() * film.GetHeight() * 4);
  srgb_image->DownloadData(image_data.data());
  stbi_write_bmp("output.bmp", film.GetWidth(), film.GetHeight(), 4, image_data.data());
}
