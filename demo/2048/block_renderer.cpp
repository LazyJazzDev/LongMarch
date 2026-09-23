#include "block_renderer.h"

#include "rounded_rectangle.h"

glm::vec3 BlockRenderer::GetNumberForegroundColor(int number) {
  switch (number) {
    case 2:
    case 4:
      return glm::vec3{117.0f, 110.0f, 102.0f} / 255.0f;
    default:
      return glm::vec3{255.0f, 255.0f, 255.0f} / 255.0f;
  }
}

glm::vec3 BlockRenderer::GetNumberBackgroundColor(int number) {
  switch (number) {
    case 2:
      return glm::vec3{236.0f, 228.0f, 219.0f} / 255.0f;
    case 4:
      return glm::vec3{235.0f, 224.0f, 202.0f} / 255.0f;
    case 8:
      return glm::vec3{232.0f, 179.0f, 129.0f} / 255.0f;
    case 16:
      return glm::vec3{223.0f, 145.0f, 95.0f} / 255.0f;
    case 32:
      return glm::vec3{230.0f, 130.0f, 102.0f} / 255.0f;
    case 64:
      return glm::vec3{217.0f, 98.0f, 67.0f} / 255.0f;
    case 128:
      return glm::vec3{238.0f, 217.0f, 123.0f} / 255.0f;
    case 256:
      return glm::vec3{235.0f, 209.0f, 99.0f} / 255.0f;
    case 512:
      return glm::vec3{222.0f, 193.0f, 76.0f} / 255.0f;
    case 1024:
      return glm::vec3{220.0f, 187.0f, 66.0f} / 255.0f;
    default:
      if (number == (number & -number)) {
        return glm::vec3{229.0f, 197.0f, 66.0f} / 255.0f;
      } else {
        return glm::vec3{0.0f};
      }
  }
}

BlockRenderer::BlockRenderer(Application *app, font::Factory *font_factory) : app_(app) {
  font_factory_ = font_factory;
  for (int i = 1; i <= 4096; i *= 2) {
    LoadNumber(i);
  }
  auto background_model =
      GenerateRoundedRectangle(1.0f / 16.0f, 1.0f / 16.0f, 15.0f / 16.0f, 15.0f / 16.0f, 1.0f / 32.0f, glm::vec3{1.0f});
  background_device_model_ = std::make_unique<DeviceModel>(app_, background_model);
}

void BlockRenderer::LoadNumber(int number) {
  if (foreground_device_models_.count(number)) {
    return;
  }
  std::wstring number_wstr;
  std::string number_str = std::to_string(number);
  for (auto c : number_str) {
    number_wstr.push_back(c);
  }
  float scale = 0.5;
  auto number_mesh = font_factory_->GetString(number_wstr);
  float height = std::min(scale, 0.8f / (number_mesh.advance));
  float blank = (1.0f - height) * 0.5f + height * 0.125f;

  number_mesh.advance *= height;
  for (auto &vertex : number_mesh.GetVertices()) {
    vertex *= height;
    vertex += glm::vec2{(1.0f - number_mesh.advance) * 0.5f, blank};
  }

  auto foreground_model = Model(ComposeVertices(number_mesh.GetVertices(), glm::vec4{1.0f}), number_mesh.GetIndices());

  foreground_device_models_[number] = std::make_unique<DeviceModel>(app_, foreground_model);
}

void BlockRenderer::Render(int number, float x, float y, float size, float alpha, float depth_offset) {
  LoadNumber(number);

  app_->DrawModel(
      background_device_model_.get(),
      {board_to_world_ * GetModelMatrixZO({x, y}, {size, size}, depth_offset),
       glm::vec4{GetNumberBackgroundColor(number >> 1) * (1.0f - alpha) + GetNumberBackgroundColor(number) * alpha,
                 1.0f},
       glm::uvec4{0}});

  app_->DrawModel(foreground_device_models_[number].get(),
                  {board_to_world_ * GetModelMatrixZO({x, y}, {size, size}, depth_offset - 0.1f / 16.0f),
                   glm::vec4{GetNumberForegroundColor(number), 1.0f}, glm::uvec4{0}});
}

void BlockRenderer::SetBoardToWorld(glm::mat4 mat) {
  board_to_world_ = mat;
}
