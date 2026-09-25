#include "RenderSession.h"

#include <stdexcept>
using namespace grassland;

RenderSession::RenderSession(const std::filesystem::path &resources,
                             const std::string &scene,
                             int max_dimension,
                             bool prepare,
                             double aspect_ratio,
                             graphics::BackendAPI backend,
                             bool allow_compute_fallback) {
  graphics::ConfigureShaderCache({resources / "shaders", !prepare, backend == grassland::graphics::BACKEND_API_METAL});
  if (graphics::CreateCore(backend, graphics::Core::Settings{1, false}, &graphics_) ||
      graphics_->InitializeLogicalDeviceAutoSelect(false))
    throw std::runtime_error("Cannot initialize requested graphics backend");
  compute_fallback_ = !graphics_->DeviceRayQuerySupport();
  if (compute_fallback_ && !allow_compute_fallback)
    throw std::runtime_error("This device does not support ray queries required by this render session.");
  auto sobol = resources / "assets/data/new-joe-kuo-7.21201";
  if (!std::filesystem::is_regular_file(sobol))
    throw std::runtime_error("Bundled Sobol table is missing");
  FileProbe::GetInstance().AddSearchPath((resources / "assets").string() + "/");
  core_ = std::make_unique<sparkium::Core>(graphics_.get());
  std::string error;
  scene_ = sparkium::JsonScene::Load(core_.get(), resources / "assets" / "scenes" / scene / "scene.json", &error,
                                     max_dimension, aspect_ratio);
  if (!scene_)
    throw std::runtime_error(error);
  scene_->GetScene()->settings.samples_per_dispatch = 1;
  scene_exposure_ = scene_->GetFilm()->info.exposure;
  graphics_->CreateImage(Width(), Height(), graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &hdr_image_);
  // Retain camera pose and vertical FOV, materials, exposure and bounce settings.
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

void RenderSession::Render() {
  core_->Render(scene_->GetScene(), scene_->GetCamera(), scene_->GetFilm(),
                compute_fallback_ ? sparkium::RENDER_PIPELINE_RT_FALLBACK : sparkium::RENDER_PIPELINE_RAY_QUERY);
}

std::vector<uint8_t> RenderSession::Step() {
  Render();
  return Display(false);
}

graphics::Image *RenderSession::Develop(bool hdr, float exposure) {
  scene_->GetFilm()->info.exposure = scene_exposure_ + std::clamp(exposure, -10.f, 10.f);
  auto *target = hdr ? hdr_image_.get() : image_.get();
  scene_->GetFilm()->Develop(target, hdr);
  return target;
}

std::vector<uint8_t> RenderSession::Display(bool hdr, float exposure) {
  auto *target = Develop(hdr, exposure);
  std::vector<uint8_t> pixels(static_cast<size_t>(Width()) * Height() * (hdr ? 16 : 4));
  target->DownloadData(pixels.data());
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

int RenderSession::MaxBounces() const {
  return scene_->GetScene()->settings.max_bounces;
}

void RenderSession::ResetFilm() {
  graphics_->WaitGPU();
  scene_->GetFilm()->Reset();
  graphics_->WaitGPU();
}

std::string RenderSession::Device() const {
  return graphics_->DeviceName();
}
