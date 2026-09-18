// Dumps the float accumulation film for a JSON scene on a chosen pipeline, so
// backends can be compared before the 8-bit develop step.
#include <long_march.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

using namespace long_march;

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "usage: film_probe <scene.json> <auto|cpu|cuda> <frames> <out.bin> [resolution]\n";
    return 2;
  }
  const std::string mode = argv[2];
  const int frames = std::stoi(argv[3]);
  const int resolution = argc > 5 ? std::stoi(argv[5]) : 0;
  std::unique_ptr<graphics::Core> graphics_core;
  if (graphics::CreateCore(graphics::BACKEND_API_DEFAULT, graphics::Core::Settings{2, false}, &graphics_core) != 0)
    return 1;
  if (graphics_core->InitializeLogicalDeviceAutoSelect(false) != 0)
    return 1;
  sparkium::Core core(graphics_core.get());
  std::string error;
  auto loaded = sparkium::JsonScene::Load(&core, argv[1], &error);
  if (!loaded) {
    std::cerr << error << '\n';
    return 1;
  }
  auto pipeline = sparkium::RENDER_PIPELINE_AUTO;
  if (mode == "cpu") pipeline = sparkium::RENDER_PIPELINE_NATIVE_CPU;
  else if (mode == "cuda") pipeline = sparkium::RENDER_PIPELINE_NATIVE_CUDA;
  else pipeline = loaded->GetRenderPipeline();

  sparkium::Film *film = loaded->GetFilm();
  std::unique_ptr<sparkium::Film> scaled;
  std::unique_ptr<sparkium::Camera> camera_copy;
  sparkium::Camera *camera = loaded->GetCamera();
  if (resolution > 0) {
    scaled = std::make_unique<sparkium::Film>(&core, resolution, resolution);
    scaled->info = film->info;
    film = scaled.get();
    camera_copy = std::make_unique<sparkium::Camera>(&core, camera->view, camera->fovy, 1.0f);
    camera_copy->aperture_radius = camera->aperture_radius;
    camera_copy->focus_distance = camera->focus_distance;
    camera_copy->aperture_blades = camera->aperture_blades;
    camera_copy->aperture_rotation = camera->aperture_rotation;
    camera_copy->aperture_ratio = camera->aperture_ratio;
    camera = camera_copy.get();
  }
  for (int frame = 0; frame < frames; ++frame)
    core.Render(loaded->GetScene(), camera, film, pipeline);
  std::vector<glm::vec4> pixels(static_cast<size_t>(film->GetWidth()) * film->GetHeight());
  film->GetRawImage()->DownloadData(pixels.data());
  std::ofstream stream(argv[4], std::ios::binary);
  const int32_t header[2]{film->GetWidth(), film->GetHeight()};
  stream.write(reinterpret_cast<const char *>(header), sizeof(header));
  stream.write(reinterpret_cast<const char *>(pixels.data()), pixels.size() * sizeof(glm::vec4));
  std::cout << "wrote " << film->GetWidth() << 'x' << film->GetHeight() << " film, "
            << film->info.accumulated_samples << " spp\n";
  return 0;
}
