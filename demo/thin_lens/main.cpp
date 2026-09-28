#include <long_march.h>

#include <cmath>
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

#include "../sparkium_backend.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace long_march;

int main(int argc, char **argv) {
  try {
    auto backend = graphics::BACKEND_API_DEFAULT;
    bool headless = false, pinhole_mode = false;
    int frames = 0, blades = 6;
    float aperture = 0.22f, focus = 5.4f;
    std::filesystem::path output;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        std::cout << "Thin Lens: interactive depth of field and bokeh\n"
                     "--backend auto|metal|vulkan|d3d12 --frames N --headless --output image.png\n"
                     "--pinhole --focus distance --aperture radius --blades N (0 = circle)\n";
        return 0;
      } else if (arg == "--backend" && i + 1 < argc)
        backend = ParseSparkiumBackend(argv[++i]);
      else if (arg == "--frames" && i + 1 < argc) {
        frames = std::stoi(argv[++i]);
        if (frames <= 0)
          throw std::invalid_argument("--frames must be positive");
      } else if (arg == "--headless")
        headless = true;
      else if (arg == "--pinhole")
        pinhole_mode = true;
      else if (arg == "--output" && i + 1 < argc)
        output = argv[++i];
      else if (arg == "--focus" && i + 1 < argc)
        focus = std::stof(argv[++i]);
      else if (arg == "--aperture" && i + 1 < argc)
        aperture = std::stof(argv[++i]);
      else if (arg == "--blades" && i + 1 < argc)
        blades = std::stoi(argv[++i]);
      else
        throw std::invalid_argument("unknown or incomplete argument: " + arg);
    }
    if (!std::isfinite(focus) || focus <= 0 || !std::isfinite(aperture) || aperture < 0 ||
        (blades != 0 && (blades < 3 || blades > 8)))
      throw std::invalid_argument("invalid focus, aperture or blade count");
    if (headless && frames == 0)
      throw std::invalid_argument("--headless requires --frames");

    std::unique_ptr<graphics::Core> graphics;
    if (graphics::CreateCore(backend, graphics::Core::Settings{}, &graphics) ||
        graphics->InitializeLogicalDeviceAutoSelect(false))
      throw std::runtime_error("cannot initialize graphics device");
    sparkium::Core core(graphics.get());
    sparkium::Scene scene(&core);
    scene.settings.samples_per_dispatch = 8;
    scene.settings.max_bounces = 4;
    scene.settings.background_color = glm::vec3(0.015f, 0.022f, 0.04f);
    sparkium::GeometryMesh sphere(&core, Mesh<>::Sphere(32, 16));
    std::vector<std::unique_ptr<sparkium::Material>> materials;
    std::vector<std::unique_ptr<sparkium::EntityGeometryMaterial>> entities;
    auto diffuse = [&](glm::vec3 color) -> sparkium::Material * {
      auto material = std::make_unique<sparkium::MaterialPrincipled>(&core, color);
      material->roughness = 0.3f;
      material->specular = 0.4f;
      auto *ptr = material.get();
      materials.push_back(std::move(material));
      return ptr;
    };
    auto light = [&](glm::vec3 color) -> sparkium::Material * {
      auto material = std::make_unique<sparkium::MaterialLight>(&core, color, true);
      auto *ptr = material.get();
      materials.push_back(std::move(material));
      return ptr;
    };
    auto add = [&](sparkium::Geometry *geometry, sparkium::Material *material, glm::vec3 position, glm::vec3 scale) {
      const auto transform = glm::scale(glm::translate(glm::mat4(1), position), scale);
      auto entity =
          std::make_unique<sparkium::EntityGeometryMaterial>(&core, geometry, material, glm::mat4x3(transform));
      scene.AddEntity(entity.get());
      entities.push_back(std::move(entity));
    };
    const std::vector<Vector3<float>> floor_positions{
        {-0.5f, 0, -0.5f}, {-0.5f, 0, 0.5f}, {0.5f, 0, 0.5f}, {0.5f, 0, -0.5f}};
    const uint32_t floor_indices[]{0, 1, 2, 0, 2, 3};
    sparkium::GeometryMesh tile(&core, Mesh<>(4, 6, floor_indices, floor_positions.data()));
    auto *dark = diffuse({0.045f, 0.06f, 0.09f});
    auto *pale = diffuse({0.24f, 0.29f, 0.36f});
    for (int z = -9; z <= 4; ++z)
      for (int x = -7; x <= 7; ++x)
        add(&tile, (x + z) % 2 ? dark : pale, {float(x), 0, float(z)}, glm::vec3(1));
    const glm::vec3 positions[]{{-1.25f, 0.65f, 2}, {0, 0.65f, 0}, {2.0f, 0.65f, -3}};
    const glm::vec3 colors[]{{0.9f, 0.16f, 0.055f}, {0.04f, 0.65f, 0.52f}, {0.08f, 0.3f, 0.9f}};
    for (int i = 0; i < 3; ++i)
      add(&sphere, diffuse(colors[i]), positions[i], glm::vec3(0.65f));
    add(&sphere, light({18, 16, 13}), {-3, 6, 2}, glm::vec3(2));
    auto *warm = light({12, 4, 1});
    auto *cool = light({1, 5, 12});
    for (int row = 0; row < 3; ++row)
      for (int x = -5; x <= 5; ++x)
        add(&sphere, (x + row) % 2 ? warm : cool, {float(x) * 1.1f, 1.2f + row * 0.85f, -7.0f}, glm::vec3(0.035f));

    constexpr int width = 1100, height = 700;
    sparkium::Film film(&core, width, height);
    film.info.view_transform = 1;
    const auto view = glm::lookAt(glm::vec3(0, 1.5f, 6), glm::vec3(0, 1.5f, 0), glm::vec3(0, 1, 0));
    sparkium::CameraThinLens lens(&core, view, glm::radians(45.0f), float(width) / height);
    lens.aperture_radius = aperture;
    lens.focus_distance = focus;
    lens.aperture_blades = blades;
    sparkium::CameraPinhole pinhole(&core, view, lens.fovy, lens.aspect);
    std::unique_ptr<graphics::Image> image;
    graphics->CreateImage(width, height, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image);
    std::unique_ptr<graphics::Window> window;
    if (!headless) {
      graphics->CreateWindowObject(width, height, "Sparkium - Thin Lens", &window);
      window->InitImGui(nullptr, 18.0f);
      ImGui::GetIO().IniFilename = nullptr;
    }
    const auto pipeline = graphics->DeviceRayQuerySupport()     ? sparkium::RENDER_PIPELINE_RAY_QUERY
                          : graphics->DeviceRayTracingSupport() ? sparkium::RENDER_PIPELINE_RAY_TRACING
                                                                : sparkium::RENDER_PIPELINE_RT_FALLBACK;
    std::cout << "Device: " << graphics->DeviceName() << ", pipeline: " << int(pipeline) << std::endl;
    int rendered = 0;
    while ((!window || !window->ShouldClose()) && (!frames || rendered < frames)) {
      if (window) {
        window->BeginImGuiFrame();
        ImGui::SetNextWindowPos({16, 16}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({330, 0}, ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.92f);
        ImGui::Begin("Thin Lens", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove);
        ImGui::TextWrapped("Focus one subject. Watch the background lights change into aperture-shaped bokeh.");
        ImGui::PushItemWidth(160.0f);
        bool changed = ImGui::Checkbox("Pinhole comparison", &pinhole_mode);
        ImGui::BeginDisabled(pinhole_mode);
        changed |= ImGui::SliderFloat("Focus", &lens.focus_distance, 2.5f, 13.0f, "%.2f");
        if (ImGui::Button("Near / orange")) {
          lens.focus_distance = 3.4f;
          changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Mid / teal")) {
          lens.focus_distance = 5.4f;
          changed = true;
        }
        if (ImGui::Button("Far / blue")) {
          lens.focus_distance = 8.4f;
          changed = true;
        }
        changed |= ImGui::SliderFloat("Aperture radius", &lens.aperture_radius, 0.0f, 0.45f, "%.3f");
        int shape = lens.aperture_blades == 0 ? 0 : lens.aperture_blades - 2;
        if (ImGui::Combo("Shape", &shape, "Circle\0Triangle\0Square\0Pentagon\0Hexagon\0Heptagon\0Octagon\0")) {
          lens.aperture_blades = shape == 0 ? 0 : shape + 2;
          changed = true;
        }
        changed |= ImGui::SliderAngle("Rotation", &lens.aperture_rotation, 0, 180);
        changed |= ImGui::SliderFloat("Horizontal ratio", &lens.aperture_ratio, 0.4f, 2.5f, "%.2f");
        ImGui::EndDisabled();
        if (ImGui::Button("Reset lens")) {
          pinhole_mode = false;
          lens.aperture_radius = 0.22f;
          lens.focus_distance = 5.4f;
          lens.aperture_blades = 6;
          lens.aperture_rotation = 0;
          lens.aperture_ratio = 1;
          changed = true;
        }
        if (changed)
          film.Reset();
        ImGui::Text("Samples: %d", film.info.accumulated_samples);
        ImGui::TextWrapped("Larger aperture = shallower depth of field. Zero aperture matches a pinhole.");
        ImGui::PopItemWidth();
        ImGui::End();
        window->EndImGuiFrame();
      }
      core.Render(&scene, pinhole_mode ? static_cast<sparkium::Camera *>(&pinhole) : &lens, &film, pipeline);
      film.Develop(image.get());
      if (window) {
        std::unique_ptr<graphics::CommandContext> commands;
        graphics->CreateCommandContext(&commands);
        commands->CmdPresent(window.get(), image.get());
        graphics->SubmitCommandContext(commands.get());
        graphics::Window::PollEvents();
      }
      ++rendered;
    }
    graphics->WaitGPU();
    if (!output.empty()) {
      if (output.has_parent_path())
        std::filesystem::create_directories(output.parent_path());
      std::vector<uint8_t> pixels(width * height * 4);
      image->DownloadData(pixels.data());
      if (!stbi_write_png(output.string().c_str(), width, height, 4, pixels.data(), width * 4))
        throw std::runtime_error("cannot save output image");
    }
    if (window)
      window->TerminateImGui();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "thin_lens: " << error.what() << '\n';
    return 1;
  }
}
