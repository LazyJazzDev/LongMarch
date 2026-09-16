#include "RenderSession.h"

#include <stdexcept>
using namespace grassland;
RenderSession::RenderSession(const std::filesystem::path &resources,
                             const std::string &scene,
                             int max_dimension,
                             bool prepare) {
  graphics::ConfigureShaderCache({resources / "shaders", !prepare, true});
  if (graphics::CreateCore(graphics::BACKEND_API_METAL, graphics::Core::Settings{1, false}, &graphics_) ||
      graphics_->InitializeLogicalDeviceAutoSelect(false))
    throw std::runtime_error("Cannot initialize Metal");
  if (!graphics_->DeviceRayQuerySupport())
    throw std::runtime_error(
        "This device does not support Metal ray queries. Use a supported iPhone or iPad; the simulator may not support "
        "ray tracing.");
  auto sobol = resources / "assets/data/new-joe-kuo-7.21201";
  if (!std::filesystem::is_regular_file(sobol))
    throw std::runtime_error("Bundled Sobol table is missing");
  FileProbe::GetInstance().AddSearchPath((resources / "assets").string() + "/");
  core_ = std::make_unique<sparkium::Core>(graphics_.get());
  std::string error;
  scene_ = sparkium::JsonScene::Load(core_.get(), resources / "assets" / "scenes" / scene / "scene.json", &error,
                                     max_dimension);
  if (!scene_)
    throw std::runtime_error(error);
  scene_->GetScene()->settings.samples_per_dispatch = 1;
  // Retain the scene's camera, material, exposure and bounce settings.
  graphics_->CreateImage(Width(), Height(), graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image_);
}
RenderSession::~RenderSession() {
  // Complete queued uploads before releasing scene resources, including after a failed render.
  try {
    if (graphics_)
      graphics_->WaitGPU();
  } catch (...) {
  }
}
std::vector<uint8_t> RenderSession::Step() {
  core_->Render(scene_->GetScene(), scene_->GetCamera(), scene_->GetFilm(), sparkium::RENDER_PIPELINE_RAY_QUERY);
  scene_->GetFilm()->Develop(image_.get());
  std::vector<uint8_t> pixels(static_cast<size_t>(Width()) * Height() * 4);
  image_->DownloadData(pixels.data());
  return pixels;
}
int RenderSession::Width() const {
  return scene_->GetFilm()->GetWidth();
}
int RenderSession::Height() const {
  return scene_->GetFilm()->GetHeight();
}
int RenderSession::Samples() const {
  return scene_->GetFilm()->info.accumulated_samples;
}
std::string RenderSession::Device() const {
  return graphics_->DeviceName();
}
