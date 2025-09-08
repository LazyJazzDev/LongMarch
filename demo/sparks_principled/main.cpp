#include <long_march.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
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
  scene.settings.samples_per_dispatch = 32;
  XH::Film film(&sparks_core, 1024, 1024);
  XH::Camera camera(&sparks_core,
                    glm::lookAt(glm::vec3{278.0f, 273.0f, -800.0f}, glm::vec3{278.0f, 273.0f, -800.0f + 1.0f},
                                glm::vec3{0.0, 1.0, 0.0}),
                    glm::radians(40.0f), static_cast<float>(film.GetWidth()) / film.GetHeight());

  XH::MaterialLambertian material_white(&sparks_core, {0.725, 0.71, 0.68});
  XH::MaterialLambertian material_red(&sparks_core, {0.63, 0.065, 0.05});
  XH::MaterialLambertian material_green(&sparks_core, {0.14, 0.45, 0.091});
  XH::MaterialPrincipled material_light(&sparks_core, {0.725, 0.71, 0.68});
  // XH::MaterialLambertian material_white(&sparks_core, {0.8, 0.8, 0.8});
  // XH::MaterialLambertian material_red(&sparks_core, {0.8, 0.0, 0.0});
  // XH::MaterialLambertian material_green(&sparks_core, {0.0, 0.8, 0.0});
  // XH::MaterialPrincipled material_light(&sparks_core, {0.8, 0.8, 0.8});
  material_light.emission_color = {1.0f, 1.0f, 1.0f};
  material_light.emission_strength = 30.0f;
  XH::MaterialPrincipled material_principled(&sparks_core, {0.725, 0.71, 0.68});

  std::vector<glm::vec3> positions;
  std::vector<glm::vec2> tex_coords;
  std::vector<uint32_t> indices;

  positions = {{343.0f, 548.7f, 227.0f}, {343.0f, 548.7f, 332.0f}, {213.0f, 548.7f, 332.0f}, {213.0f, 548.7f, 227.0f}};
  tex_coords = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
  indices = {0, 1, 3, 1, 2, 3};
  Mesh<float> light(positions.size(), indices.size(), indices.data(),
                    reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                    reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  XH::GeometryMesh geometry_light(&sparks_core, light);
  XH::EntityGeometryMaterial entity_light(&sparks_core, &geometry_light, &material_light);
  // XH::EntityGeometryLight entity_light(&sparks_core, &geometry_light, {30.0f, 30.0f, 30.0f}, true, true,
  //                                          glm::mat4x3(1.0f));

  positions = {{552.8f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 559.2f}, {549.6f, 0.0f, 559.2f}};
  Mesh<float> floor(positions.size(), indices.size(), indices.data(),
                    reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                    reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  XH::GeometryMesh geometry_floor(&sparks_core, floor);
  XH::EntityGeometryMaterial entity_floor(&sparks_core, &geometry_floor, &material_white);

  positions = {{556.0f, 548.8f, 0.0f}, {556.0f, 548.8f, 559.2f}, {0.0f, 548.8f, 559.2f}, {0.0f, 548.8f, 0.0f}};
  Mesh<float> ceiling(positions.size(), indices.size(), indices.data(),
                      reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                      reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  XH::GeometryMesh geometry_ceiling(&sparks_core, ceiling);
  XH::EntityGeometryMaterial entity_ceiling(&sparks_core, &geometry_ceiling, &material_white);

  positions = {{549.6f, 0.0f, 559.2f}, {0.0f, 0.0f, 559.2f}, {0.0f, 548.8f, 559.2f}, {556.0f, 548.8f, 559.2f}};
  Mesh<float> back_wall(positions.size(), indices.size(), indices.data(),
                        reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                        reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  XH::GeometryMesh geometry_back_wall(&sparks_core, back_wall);
  XH::EntityGeometryMaterial entity_back_wall(&sparks_core, &geometry_back_wall, &material_white);

  positions = {{0.0f, 0.0f, 559.2f}, {0.0f, 0.0f, 0.0f}, {0.0f, 548.8f, 0.0f}, {0.0f, 548.8f, 559.2f}};
  Mesh<float> right_wall(positions.size(), indices.size(), indices.data(),
                         reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                         reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  XH::GeometryMesh geometry_right_wall(&sparks_core, right_wall);
  XH::EntityGeometryMaterial entity_right_wall(&sparks_core, &geometry_right_wall, &material_green);

  positions = {{552.8f, 0.0f, 0.0f}, {549.6f, 0.0f, 559.2f}, {556.0f, 548.8f, 559.2f}, {556.0f, 548.8f, 0.0f}};
  Mesh<float> left_wall(positions.size(), indices.size(), indices.data(),
                        reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                        reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  XH::GeometryMesh geometry_left_wall(&sparks_core, left_wall);
  XH::EntityGeometryMaterial entity_left_wall(&sparks_core, &geometry_left_wall, &material_red);

  indices = {0, 1, 3, 1, 2, 3, 4, 5, 7, 5, 6, 7, 8, 9, 11, 9, 10, 11, 12, 13, 15, 13, 14, 15, 16, 17, 19, 17, 18, 19};
  positions = {{130.0f, 165.0f, 65.0f}, {82.0f, 165.0f, 225.0f},  {240.0f, 165.0f, 272.0f}, {290.0f, 165.0f, 114.0f},
               {290.0f, 0.0f, 114.0f},  {290.0f, 165.0f, 114.0f}, {240.0f, 165.0f, 272.0f}, {240.0f, 0.0f, 272.0f},
               {130.0f, 0.0f, 65.0f},   {130.0f, 165.0f, 65.0f},  {290.0f, 165.0f, 114.0f}, {290.0f, 0.0f, 114.0f},
               {82.0f, 0.0f, 225.0f},   {82.0f, 165.0f, 225.0f},  {130.0f, 165.0f, 65.0f},  {130.0f, 0.0f, 65.0f},
               {240.0f, 0.0f, 272.0f},  {240.0f, 165.0f, 272.0f}, {82.0f, 165.0f, 225.0f},  {82.0f, 0.0f, 225.0f}};
  tex_coords = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f},
                {0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f},
                {1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
  XH::GeometryMesh short_box(&sparks_core, Mesh<float>(positions.size(), indices.size(), indices.data(),
                                                       reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                                                       reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr));
  XH::EntityGeometryMaterial entity_short_box(&sparks_core, &short_box, &material_white);

  positions = {{423.0f, 330.0f, 247.0f}, {265.0f, 330.0f, 296.0f}, {314.0f, 330.0f, 456.0f}, {472.0f, 330.0f, 406.0f},
               {423.0f, 0.0f, 247.0f},   {423.0f, 330.0f, 247.0f}, {472.0f, 330.0f, 406.0f}, {472.0f, 0.0f, 406.0f},
               {472.0f, 0.0f, 406.0f},   {472.0f, 330.0f, 406.0f}, {314.0f, 330.0f, 456.0f}, {314.0f, 0.0f, 456.0f},
               {314.0f, 0.0f, 456.0f},   {314.0f, 330.0f, 456.0f}, {265.0f, 330.0f, 296.0f}, {265.0f, 0.0f, 296.0f},
               {265.0f, 0.0f, 296.0f},   {265.0f, 330.0f, 296.0f}, {423.0f, 330.0f, 247.0f}, {423.0f, 0.0f, 247.0f}};
  Mesh<float> tall_box(positions.size(), indices.size(), indices.data(),
                       reinterpret_cast<Vector3<float> *>(positions.data()), nullptr,
                       reinterpret_cast<Vector2<float> *>(tex_coords.data()), nullptr);
  XH::GeometryMesh geometry_tall_box(&sparks_core, tall_box);
  XH::EntityGeometryMaterial entity_tall_box(&sparks_core, &geometry_tall_box, &material_principled);
  scene.AddEntity(&entity_light);
  scene.AddEntity(&entity_floor);
  scene.AddEntity(&entity_ceiling);
  scene.AddEntity(&entity_back_wall);
  scene.AddEntity(&entity_right_wall);
  scene.AddEntity(&entity_left_wall);
  scene.AddEntity(&entity_short_box);
  scene.AddEntity(&entity_tall_box);

  std::unique_ptr<graphics::Image> raw_image;
  core_->CreateImage(film.GetWidth(), film.GetHeight(), graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &raw_image);
  std::unique_ptr<graphics::Image> srgb_image;
  core_->CreateImage(film.GetWidth(), film.GetHeight(), graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &srgb_image);

  std::unique_ptr<graphics::Window> window;
  core_->CreateWindowObject(film.GetWidth(), film.GetHeight(), "Sparks Cornell Box", &window);
  FPSCounter fps_counter;
  window->InitImGui(nullptr, 20.0f);
  while (!window->ShouldClose()) {
    bool updated = false;

    window->BeginImGuiFrame();
    ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Once);
    ImGui::SetNextWindowBgAlpha(0.3);
    ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("Global");
    ImGui::Separator();
    ImGui::SliderInt("Samples per frame", &scene.settings.samples_per_dispatch, 1, 256);
    updated |= ImGui::SliderInt("Max Bounces", &scene.settings.max_bounces, 1, 128);
    updated |= ImGui::SliderFloat("Light Strength", &material_light.emission_strength, 0.0f, 1e6f, nullptr,
                                  ImGuiSliderFlags_Logarithmic);
    updated |= ImGui::Checkbox("Alpha Shadow", reinterpret_cast<bool *>(&scene.settings.alpha_shadow));
    ImGui::NewLine();
    ImGui::Text("Material");
    ImGui::Separator();
    updated |= ImGui::ColorEdit3("Base Color", &material_principled.base_color[0], ImGuiColorEditFlags_Float);
    updated |= ImGui::SliderFloat("Metallic", &material_principled.metallic, 0.0f, 1.0f);
    updated |= ImGui::SliderFloat("Roughness", &material_principled.roughness, 0.0f, 1.0f);
    updated |= ImGui::SliderFloat("Specular", &material_principled.specular, 0.0f, 1.0f);
    updated |= ImGui::SliderFloat("Specular Tint", &material_principled.specular_tint, 0.0f, 1.0f);
    updated |= ImGui::SliderFloat("Anisotropic", &material_principled.anisotropic, 0.0f, 1.0f);
    updated |= ImGui::SliderFloat("Anisotropic Rotation", &material_principled.anisotropic_rotation, 0.0f, 1.0f);
    updated |= ImGui::SliderFloat("Sheen", &material_principled.sheen, 0.0f, 1.0f);
    updated |= ImGui::SliderFloat("Sheen Tint", &material_principled.sheen_tint, 0.0f, 1.0f);
    updated |= ImGui::SliderFloat("Clearcoat", &material_principled.clearcoat, 0.0f, 1.0f);
    updated |= ImGui::SliderFloat("Clearcoat Roughness", &material_principled.clearcoat_roughness, 0.0f, 1.0f);
    updated |= ImGui::SliderFloat("Transmission", &material_principled.transmission, 0.0f, 1.0f);
    updated |= ImGui::SliderFloat("Transmission Roughness", &material_principled.transmission_roughness, 0.0f, 1.0f);
    updated |= ImGui::SliderFloat("IOR", &material_principled.ior, 1.0f, 2.5f);
    updated |= ImGui::SliderFloat("Subsurface", &material_principled.subsurface, 0.0f, 1.0f);
    updated |=
        ImGui::ColorEdit3("Subsurface Color", &material_principled.subsurface_color[0], ImGuiColorEditFlags_Float);
    updated |= ImGui::SliderFloat3("Subsurface Radius", &material_principled.subsurface_radius[0], 0.0f, 10.0f);
    updated |= ImGui::ColorEdit3("Emission Color", &material_principled.emission_color[0], ImGuiColorEditFlags_Float);
    updated |= ImGui::SliderFloat("Emission Strength", &material_principled.emission_strength, 0.0f, 1e6f, nullptr,
                                  ImGuiSliderFlags_Logarithmic);

    ImGui::End();
    window->EndImGuiFrame();

    if (updated) {
      film.Reset();
    }

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
    window->SetTitle(std::string("Sparks Principled BSDF - ") + fps_buf + "frames/s" + " - " + rps_buf + "Mrays/s");
  }
  window->TerminateImGui();

  sparks_core.ConvertFilmToRawImage(film, raw_image.get());
  sparks_core.ToneMapping(raw_image.get(), srgb_image.get());
  std::vector<uint8_t> image_data(film.GetWidth() * film.GetHeight() * 4);
  srgb_image->DownloadData(image_data.data());
  stbi_write_bmp("output.bmp", film.GetWidth(), film.GetHeight(), 4, image_data.data());
}
