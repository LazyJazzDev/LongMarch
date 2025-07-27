#include <long_march.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "glm/gtc/matrix_transform.hpp"
#include "stb_image_write.h"

using namespace long_march;

int main() {
  std::unique_ptr<graphics::Core> core_;

  graphics::CreateCore(graphics::BACKEND_API_VULKAN, graphics::Core::Settings{2}, &core_);
  core_->InitializeLogicalDeviceAutoSelect(true);
  sparks::Core sparks_core(core_.get());
  sparks_core.GetShadersVFS().Print();

  sparks::Scene scene(&sparks_core);
  sparks::Film film(&sparks_core, 1024, 1024);
  sparks::Camera camera(&sparks_core,
                        glm::lookAt(glm::vec3{278.0f, 273.0f, -800.0f}, glm::vec3{278.0f, 273.0f, -800.0f + 1.0f},
                                    glm::vec3{0.0, 1.0, 0.0}),
                        glm::radians(40.0f), static_cast<float>(film.GetWidth()) / film.GetHeight());

  sparks::Material material_white(&sparks_core, {{0.725, 0.71, 0.68}, {0.0f, 0.0f, 0.0f}});
  sparks::Material material_red(&sparks_core, {{0.63, 0.065, 0.05}, {0.0f, 0.0f, 0.0f}});
  sparks::Material material_green(&sparks_core, {{0.14, 0.45, 0.091}, {0.0f, 0.0f, 0.0f}});
  sparks::Material material_light(&sparks_core, {{0.0f, 0.0f, 0.0f}, {30.0f, 30.0f, 30.0f}});

  std::vector<glm::vec3> positions;
  std::vector<glm::vec2> tex_coords;
  std::vector<uint32_t> indices;

  positions = {{343.0f, 548.7f, 227.0f}, {343.0f, 548.7f, 332.0f}, {213.0f, 548.7f, 332.0f}, {213.0f, 548.7f, 227.0f}};
  tex_coords = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
  indices = {0, 1, 3, 1, 2, 3};
  Mesh<float> light(positions.size(), indices.size(), indices.data(),
                    reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                    reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  sparks::Geometry geometry_light(&sparks_core, light);
  sparks::EntityGeometryObject entity_light(&sparks_core, &geometry_light, &material_light);

  positions = {{552.8f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 559.2f}, {549.6f, 0.0f, 559.2f}};
  Mesh<float> floor(positions.size(), indices.size(), indices.data(),
                    reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                    reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  sparks::Geometry geometry_floor(&sparks_core, floor);
  sparks::EntityGeometryObject entity_floor(&sparks_core, &geometry_floor, &material_white);

  positions = {{556.0f, 548.8f, 0.0f}, {556.0f, 548.8f, 559.2f}, {0.0f, 548.8f, 559.2f}, {0.0f, 548.8f, 0.0f}};
  Mesh<float> ceiling(positions.size(), indices.size(), indices.data(),
                      reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                      reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  sparks::Geometry geometry_ceiling(&sparks_core, ceiling);
  sparks::EntityGeometryObject entity_ceiling(&sparks_core, &geometry_ceiling, &material_white);

  positions = {{549.6f, 0.0f, 559.2f}, {0.0f, 0.0f, 559.2f}, {0.0f, 548.8f, 559.2f}, {556.0f, 548.8f, 559.2f}};
  Mesh<float> back_wall(positions.size(), indices.size(), indices.data(),
                        reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                        reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  sparks::Geometry geometry_back_wall(&sparks_core, back_wall);
  sparks::EntityGeometryObject entity_back_wall(&sparks_core, &geometry_back_wall, &material_white);

  positions = {{0.0f, 0.0f, 559.2f}, {0.0f, 0.0f, 0.0f}, {0.0f, 548.8f, 0.0f}, {0.0f, 548.8f, 559.2f}};
  Mesh<float> right_wall(positions.size(), indices.size(), indices.data(),
                         reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                         reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  sparks ::Geometry geometry_right_wall(&sparks_core, right_wall);
  sparks::EntityGeometryObject entity_right_wall(&sparks_core, &geometry_right_wall, &material_green);

  positions = {{552.8f, 0.0f, 0.0f}, {549.6f, 0.0f, 559.2f}, {556.0f, 548.8f, 559.2f}, {556.0f, 548.8f, 0.0f}};
  Mesh<float> left_wall(positions.size(), indices.size(), indices.data(),
                        reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                        reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  sparks::Geometry geometry_left_wall(&sparks_core, left_wall);
  sparks::EntityGeometryObject entity_left_wall(&sparks_core, &geometry_left_wall, &material_red);

  indices = {0, 1, 3, 1, 2, 3, 4, 5, 7, 5, 6, 7, 8, 9, 11, 9, 10, 11, 12, 13, 15, 13, 14, 15, 16, 17, 19, 17, 18, 19};
  positions = {{130.0f, 165.0f, 65.0f}, {82.0f, 165.0f, 225.0f},  {240.0f, 165.0f, 272.0f}, {290.0f, 165.0f, 114.0f},
               {290.0f, 0.0f, 114.0f},  {290.0f, 165.0f, 114.0f}, {240.0f, 165.0f, 272.0f}, {240.0f, 0.0f, 272.0f},
               {130.0f, 0.0f, 65.0f},   {130.0f, 165.0f, 65.0f},  {290.0f, 165.0f, 114.0f}, {290.0f, 0.0f, 114.0f},
               {82.0f, 0.0f, 225.0f},   {82.0f, 165.0f, 225.0f},  {130.0f, 165.0f, 65.0f},  {130.0f, 0.0f, 65.0f},
               {240.0f, 0.0f, 272.0f},  {240.0f, 165.0f, 272.0f}, {82.0f, 165.0f, 225.0f},  {82.0f, 0.0f, 225.0f}};
  tex_coords = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f},
                {0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f},
                {1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
  sparks::Geometry short_box(&sparks_core, Mesh<float>(positions.size(), indices.size(), indices.data(),
                                                       reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                                                       reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr));
  sparks::EntityGeometryObject entity_short_box(&sparks_core, &short_box, &material_white);

  positions = {{423.0f, 330.0f, 247.0f}, {265.0f, 330.0f, 296.0f}, {314.0f, 330.0f, 456.0f}, {472.0f, 330.0f, 406.0f},
               {423.0f, 0.0f, 247.0f},   {423.0f, 330.0f, 247.0f}, {472.0f, 330.0f, 406.0f}, {472.0f, 0.0f, 406.0f},
               {472.0f, 0.0f, 406.0f},   {472.0f, 330.0f, 406.0f}, {314.0f, 330.0f, 456.0f}, {314.0f, 0.0f, 456.0f},
               {314.0f, 0.0f, 456.0f},   {314.0f, 330.0f, 456.0f}, {265.0f, 330.0f, 296.0f}, {265.0f, 0.0f, 296.0f},
               {265.0f, 0.0f, 296.0f},   {265.0f, 330.0f, 296.0f}, {423.0f, 330.0f, 247.0f}, {423.0f, 0.0f, 247.0f}};
  Mesh<float> tall_box(positions.size(), indices.size(), indices.data(),
                       reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                       reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  sparks::Geometry geometry_tall_box(&sparks_core, tall_box);
  sparks::EntityGeometryObject entity_tall_box(&sparks_core, &geometry_tall_box, &material_white);
  scene.AddEntity(&entity_light);
  scene.AddEntity(&entity_floor);
  scene.AddEntity(&entity_ceiling);
  scene.AddEntity(&entity_back_wall);
  scene.AddEntity(&entity_right_wall);
  scene.AddEntity(&entity_left_wall);
  scene.AddEntity(&entity_short_box);
  scene.AddEntity(&entity_tall_box);

  std::unique_ptr<graphics::Image> image;
  core_->CreateImage(film.GetWidth(), film.GetHeight(), graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image);

  std::unique_ptr<graphics::Window> window;
  core_->CreateWindowObject(film.GetWidth(), film.GetHeight(), "Sparks Cornell Box", &window);
  FPSCounter fps_counter;
  while (!window->ShouldClose()) {
    scene.Render(&camera, &film);
    sparks_core.ConvertFilmToImage(film, image.get());
    std::unique_ptr<graphics::CommandContext> cmd_context;
    core_->CreateCommandContext(&cmd_context);
    cmd_context->CmdPresent(window.get(), image.get());
    core_->SubmitCommandContext(cmd_context.get());
    glfwPollEvents();
    float fps = fps_counter.TickFPS();
    float rps = film.GetWidth() * film.GetHeight() * fps * 128;
    char buf[16];
    sprintf(buf, "%.2f", rps * 1e-6f);  // buf will be "3.14"
    window->SetTitle("Sparks Cornell Box - " + std::to_string(fps) + "frams/s" + " - " + buf + "Mrays/s");
  }

  sparks_core.ConvertFilmToImage(film, image.get());
  std::vector<uint8_t> image_data(film.GetWidth() * film.GetHeight() * 4);
  image->DownloadData(image_data.data());
  stbi_write_bmp("output.bmp", film.GetWidth(), film.GetHeight(), 4, image_data.data());
}
