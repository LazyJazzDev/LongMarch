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

  graphics::CreateCore(graphics::BACKEND_API_DEFAULT, graphics::Core::Settings{2, false}, &core_);
  core_->InitializeLogicalDeviceAutoSelect(true);
  sparkium::Core sparkium_core(core_.get());

  sparkium::Scene scene(&sparkium_core);
  scene.settings.samples_per_dispatch = 16;
  sparkium::Film film(&sparkium_core, 2048, 1024);
  film.info.persistence = 1.0f;
  sparkium::Camera camera(
      &sparkium_core, glm::lookAt(glm::vec3{-5.0f, 0.3f, 7.0f}, glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0, 1.0, 0.0}),
      glm::radians(30.0f), static_cast<float>(film.GetWidth()) / film.GetHeight());

  Mesh<> matball_mesh;
  matball_mesh.LoadObjFile(FindAssetFile("meshes/preview_sphere.obj"));
  matball_mesh.GenerateTangents();
  Mesh<> cube_mesh;
  cube_mesh.LoadObjFile(FindAssetFile("meshes/cube.obj"));
  sparkium::GeometryMesh geometry_sphere(&sparkium_core, Mesh<>::Sphere(30));
  sparkium::GeometryMesh geometry_matball(&sparkium_core, matball_mesh);
  sparkium::GeometryMesh geometry_cube(&sparkium_core, cube_mesh);

  sparkium::MaterialPrincipled material_matball0(&sparkium_core, {0.8f, 0.8f, 0.8f});
  std::unique_ptr<graphics::Image> base_color_texture0;
  std::unique_ptr<graphics::Image> roughness_texture0;
  std::unique_ptr<graphics::Image> metallic_texture0;
  std::unique_ptr<graphics::Image> normal_texture0;

  graphics::LoadImageFromFile(core_.get(), FindAssetFile("textures/Metal053C_2K-PNG/Metal053C_2K-PNG_Color.png"),
                              &base_color_texture0);
  graphics::LoadImageFromFile(core_.get(), FindAssetFile("textures/Metal053C_2K-PNG/Metal053C_2K-PNG_Roughness.png"),
                              &roughness_texture0);
  graphics::LoadImageFromFile(core_.get(), FindAssetFile("textures/Metal053C_2K-PNG/Metal053C_2K-PNG_Metalness.png"),
                              &metallic_texture0);
  graphics::LoadImageFromFile(core_.get(), FindAssetFile("textures/Metal053C_2K-PNG/Metal053C_2K-PNG_NormalGL.png"),
                              &normal_texture0);

  material_matball0.textures.base_color = base_color_texture0.get();
  material_matball0.textures.roughness = roughness_texture0.get();
  material_matball0.textures.metallic = metallic_texture0.get();
  material_matball0.textures.normal = normal_texture0.get();
  sparkium::EntityGeometryMaterial entity_matball0(&sparkium_core, &geometry_matball, &material_matball0,
                                                   glm::translate(glm::mat4{1.0f}, glm::vec3{-2.2f, 0.0f, 0.0f}));
  scene.AddEntity(&entity_matball0);

  sparkium::MaterialPrincipled material_matball1(&sparkium_core, {0.8f, 0.8f, 0.8f});
  std::unique_ptr<graphics::Image> base_color_texture1;
  std::unique_ptr<graphics::Image> roughness_texture1;
  std::unique_ptr<graphics::Image> metallic_texture1;
  std::unique_ptr<graphics::Image> normal_texture1;

  graphics::LoadImageFromFile(
      core_.get(), FindAssetFile("textures/ChristmasTreeOrnament013_4K-JPG/ChristmasTreeOrnament013_4K-JPG_Color.jpg"),
      &base_color_texture1);
  graphics::LoadImageFromFile(
      core_.get(),
      FindAssetFile("textures/ChristmasTreeOrnament013_4K-JPG/ChristmasTreeOrnament013_4K-JPG_Roughness.jpg"),
      &roughness_texture1);
  graphics::LoadImageFromFile(
      core_.get(),
      FindAssetFile("textures/ChristmasTreeOrnament013_4K-JPG/ChristmasTreeOrnament013_4K-JPG_Metalness.jpg"),
      &metallic_texture1);
  graphics::LoadImageFromFile(
      core_.get(),
      FindAssetFile("textures/ChristmasTreeOrnament013_4K-JPG/ChristmasTreeOrnament013_4K-JPG_NormalGL.jpg"),
      &normal_texture1);

  material_matball1.textures.base_color = base_color_texture1.get();
  material_matball1.textures.roughness = roughness_texture1.get();
  material_matball1.textures.metallic = metallic_texture1.get();
  material_matball1.textures.normal = normal_texture1.get();
  sparkium::EntityGeometryMaterial entity_matball1(&sparkium_core, &geometry_matball, &material_matball1,
                                                   glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, 0.0f, 0.0f}));
  scene.AddEntity(&entity_matball1);

  sparkium::MaterialPrincipled material_matball2(&sparkium_core, {0.8f, 0.8f, 0.8f});
  std::unique_ptr<graphics::Image> base_color_texture2;
  std::unique_ptr<graphics::Image> roughness_texture2;
  std::unique_ptr<graphics::Image> metallic_texture2;
  std::unique_ptr<graphics::Image> normal_texture2;
  std::unique_ptr<graphics::Image> anisotropic_texture2;
  std::unique_ptr<graphics::Image> anisotropic_rotation_texture2;

  graphics::LoadImageFromFile(core_.get(), FindAssetFile("textures/copper/Sphere_Base_color.png"),
                              &base_color_texture2);
  graphics::LoadImageFromFile(core_.get(), FindAssetFile("textures/copper/Sphere_Roughness.png"), &roughness_texture2);
  graphics::LoadImageFromFile(core_.get(), FindAssetFile("textures/copper/Sphere_Metallic.png"), &metallic_texture2);
  graphics::LoadImageFromFile(core_.get(), FindAssetFile("textures/copper/Sphere_Normal.png"), &normal_texture2);
  graphics::LoadImageFromFile(core_.get(), FindAssetFile("textures/copper/Sphere_Anisotropy_level.png"),
                              &anisotropic_texture2);
  graphics::LoadImageFromFile(core_.get(), FindAssetFile("textures/copper/Sphere_Anisotropy_angle.png"),
                              &anisotropic_rotation_texture2);

  material_matball2.textures.base_color = base_color_texture2.get();
  material_matball2.textures.metallic = metallic_texture2.get();
  material_matball2.textures.roughness = roughness_texture2.get();
  material_matball2.textures.normal = normal_texture2.get();
  material_matball2.textures.anisotropic = anisotropic_texture2.get();
  material_matball2.textures.anisotropic_rotation = anisotropic_rotation_texture2.get();
  sparkium::EntityGeometryMaterial entity_matball2(&sparkium_core, &geometry_matball, &material_matball2,
                                                   glm::translate(glm::mat4{1.0f}, glm::vec3{2.2f, 0.0f, 0.0f}));
  scene.AddEntity(&entity_matball2);

  sparkium::MaterialPrincipled material_matball3(&sparkium_core, {1.0f, 1.0f, 1.0f});
  material_matball3.roughness = 0.1f;
  material_matball3.transmission = 1.0f;
  material_matball3.ior = 1.3f;
  sparkium::EntityGeometryMaterial entity_matball3(&sparkium_core, &geometry_matball, &material_matball3,
                                                   glm::translate(glm::mat4{1.0f}, glm::vec3{4.4f, 0.0f, 0.0f}));
  scene.AddEntity(&entity_matball3);

  sparkium::MaterialPrincipled material_ground(&sparkium_core, {0.7f, 0.7f, 0.7f});
  material_ground.roughness = 0.2f;
  material_ground.metallic = 1.0f;
  sparkium::EntityGeometryMaterial entity_ground(&sparkium_core, &geometry_cube, &material_ground,
                                                 glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, -1001.0f, 0.0f}) *
                                                     glm::scale(glm::mat4(1.0f), glm::vec3(1000.0f)));

  sparkium::MaterialLight material_sky(&sparkium_core, {0.8f, 0.8f, 0.8f}, true, false);
  sparkium::EntityGeometryMaterial entity_sky(
      &sparkium_core, &geometry_sphere, &material_sky,
      glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, 0.0f, 0.0f}) * glm::scale(glm::mat4(1.0f), glm::vec3(60.0f)));
  entity_sky.raster_light = false;
  scene.settings.raster.ambient_light = glm::vec3{0.8f, 0.8f, 0.8f};

  AreaLight area_light(&sparkium_core, glm::vec3{1.0f, 1.0f, 1.0f}, 1.0f, glm::vec3{0.0f, 30.0f, 50.0f},
                       glm::normalize(glm::vec3{0.0f, -3.0f, -5.0f}));
  area_light.emission = glm::vec3{1000.0f};

  scene.AddEntity(&entity_ground);
  scene.AddEntity(&entity_sky);
  scene.AddEntity(area_light);

  std::unique_ptr<graphics::Image> srgb_image;
  core_->CreateImage(film.GetWidth(), film.GetHeight(), graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &srgb_image);

  std::unique_ptr<graphics::Window> window;
  core_->CreateWindowObject(film.GetWidth(), film.GetHeight(), "Sparkium", &window);
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
    window->SetTitle(std::string("Sparkium Textured PBR - ") + fps_buf + "frames/s" + " - " + rps_buf + "Mrays/s");
  }

  film.Develop(srgb_image.get());
  std::vector<uint8_t> image_data(film.GetWidth() * film.GetHeight() * 4);
  srgb_image->DownloadData(image_data.data());
  stbi_write_bmp("output.bmp", film.GetWidth(), film.GetHeight(), 4, image_data.data());
}
