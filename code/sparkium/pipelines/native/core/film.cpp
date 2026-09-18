#include "sparkium/pipelines/native/core/film.h"

namespace sparkium::native {

Film::Film(sparkium::Film &film) : film_(film) {
  film_.RegisterResetCallback([this]() { Reset(); });
  const size_t pixels = static_cast<size_t>(film_.GetWidth()) * static_cast<size_t>(film_.GetHeight());
  accumulated_color_.resize(pixels);
  accumulated_samples_.resize(pixels);
  Reset();
}

void Film::Reset() {
  std::fill(accumulated_color_.begin(), accumulated_color_.end(), float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::fill(accumulated_samples_.begin(), accumulated_samples_.end(), 0.0f);
  film_.info.accumulated_samples = 0;
}

void Film::PublishRawImage() {
  std::vector<float4> image(accumulated_color_.size());
  for (size_t i = 0; i < image.size(); ++i)
    image[i] = FilmToImage(accumulated_color_[i], accumulated_samples_[i]);
  film_.GetRawImage()->UploadData(image.data());
}

int Film::GetWidth() const {
  return film_.GetWidth();
}

int Film::GetHeight() const {
  return film_.GetHeight();
}

Film *DedicatedCast(sparkium::Film *film) {
  COMPONENT_CAST(film, Film);
}

}  // namespace sparkium::native
