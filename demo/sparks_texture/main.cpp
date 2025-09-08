#include <long_march.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <random>

#include "glm/gtc/matrix_transform.hpp"
#include "stb_image_write.h"

using namespace CZ;

int main() {
  std::unique_ptr<graphics::Core> core_;

  graphics::CreateCore(graphics::BACKEND_API_DEFAULT, graphics::Core::Settings{2, false}, &core_);
  core_->InitializeLogicalDeviceAutoSelect(true);
  XH::Core sparks_core(core_.get());
  sparks_core.GetShadersVFS().Print();

  XH::Scene scene(&sparks_core);
  scene.settings.samples_per_dispatch = 16;
  XH::Film film(&sparks_core, 2048, 1024);
  film.info.persistence = 1.0f;
  XH::Camera camera(&sparks_core,
                    glm::lookAt(glm::vec3{-5.0f, 0.3f, 7.0f}, glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0, 1.0, 0.0}),
                    glm::radians(30.0f), static_cast<float>(film.GetWidth()) / film.GetHeight());

  Mesh<> matball_mesh;
  matball_mesh.LoadObjFile(FindAssetFile("meshes/preview_sphere.obj"));
  matball_mesh.GenerateTangents();
  Mesh<> cube_mesh;
  cube_mesh.LoadObjFile(FindAssetFile("meshes/cube.obj"));
  XH::GeometryMesh geometry_sphere(&sparks_core, Mesh<>::Sphere(30));
  XH::GeometryMesh geometry_matball(&sparks_core, matball_mesh);
  XH::GeometryMesh geometry_cube(&sparks_core, cube_mesh);

  XH::MaterialPrincipled material_matball0(&sparks_core, {0.8f, 0.8f, 0.8f});
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
  XH::EntityGeometryMaterial entity_matball0(&sparks_core, &geometry_matball, &material_matball0,
                                             glm::translate(glm::mat4{1.0f}, glm::vec3{-2.2f, 0.0f, 0.0f}));
  scene.AddEntity(&entity_matball0);

  XH::MaterialPrincipled material_matball1(&sparks_core, {0.8f, 0.8f, 0.8f});
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
  XH::EntityGeometryMaterial entity_matball1(&sparks_core, &geometry_matball, &material_matball1,
                                             glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, 0.0f, 0.0f}));
  scene.AddEntity(&entity_matball1);

  XH::MaterialPrincipled material_matball2(&sparks_core, {0.8f, 0.8f, 0.8f});
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
  XH::EntityGeometryMaterial entity_matball2(&sparks_core, &geometry_matball, &material_matball2,
                                             glm::translate(glm::mat4{1.0f}, glm::vec3{2.2f, 0.0f, 0.0f}));
  scene.AddEntity(&entity_matball2);

  XH::MaterialPrincipled material_matball3(&sparks_core, {1.0f, 1.0f, 1.0f});
  material_matball3.roughness = 0.1f;
  material_matball3.transmission = 1.0f;
  material_matball3.ior = 1.3f;
  XH::EntityGeometryMaterial entity_matball3(&sparks_core, &geometry_matball, &material_matball3,
                                             glm::translate(glm::mat4{1.0f}, glm::vec3{4.4f, 0.0f, 0.0f}));
  scene.AddEntity(&entity_matball3);

  XH::MaterialPrincipled material_ground(&sparks_core, {0.1f, 0.2f, 0.4f});
  material_ground.roughness = 0.2f;
  material_ground.metallic = 0.0f;
  XH::EntityGeometryMaterial entity_ground(&sparks_core, &geometry_cube, &material_ground,
                                           glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, -1001.0f, 0.0f}) *
                                               glm::scale(glm::mat4(1.0f), glm::vec3(1000.0f)));

  XH::MaterialLight material_sky(&sparks_core, {0.8f, 0.8f, 0.8f}, true, false);
  XH::EntityGeometryMaterial entity_sky(
      &sparks_core, &geometry_sphere, &material_sky,
      glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, 0.0f, 0.0f}) * glm::scale(glm::mat4(1.0f), glm::vec3(60.0f)));
  XH::EntityAreaLight area_light(&sparks_core, glm::vec3{1.0f, 1.0f, 1.0f}, 1.0f, glm::vec3{0.0f, 30.0f, 50.0f},
                                 glm::normalize(glm::vec3{0.0f, -3.0f, -5.0f}));
  area_light.emission = glm::vec3{1000.0f};

  scene.AddEntity(&entity_ground);
  scene.AddEntity(&entity_sky);
  scene.AddEntity(&area_light);

  std::unique_ptr<graphics::Image> raw_image;
  core_->CreateImage(film.GetWidth(), film.GetHeight(), graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &raw_image);
  std::unique_ptr<graphics::Image> srgb_image;
  core_->CreateImage(film.GetWidth(), film.GetHeight(), graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &srgb_image);

  std::unique_ptr<graphics::Window> window;
  core_->CreateWindowObject(film.GetWidth(), film.GetHeight(), "Sparks", &window);
  FPSCounter fps_counter;
  while (!window->ShouldClose()) {
    // area_light.position = glm::mat3{glm::rotate(glm::mat4{1.0f}, glm::radians(0.3f), glm::vec3{0.0f, 1.0f, 0.0f})} *
    // area_light.position; if (area_light.position.y < 0.0) area_light.position = -area_light.position;
    // area_light.direction = -area_light.position;
    scene.Render(&camera, &film);
    sparks_core.ConvertFilmToRawImage(film, raw_image.get());
    sparks_core.ToneMapping(raw_image.get(), srgb_image.get());
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
    window->SetTitle(std::string("Sparks Textured PBR - ") + fps_buf + "frames/s" + " - " + rps_buf + "Mrays/s");
  }

  sparks_core.ConvertFilmToRawImage(film, raw_image.get());
  sparks_core.ToneMapping(raw_image.get(), srgb_image.get());
  std::vector<uint8_t> image_data(film.GetWidth() * film.GetHeight() * 4);
  srgb_image->DownloadData(image_data.data());
  stbi_write_bmp("output.bmp", film.GetWidth(), film.GetHeight(), 4, image_data.data());
}
