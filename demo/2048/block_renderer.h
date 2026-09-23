#pragma once
#include "application/model.h"
#include "font/factory.h"

class BlockRenderer {
 public:
  explicit BlockRenderer(Application *app, font::Factory *font_factory);
  void LoadNumber(int number);
  void Render(int number, float x, float y, float size, float alpha, float depth_offset);
  static glm::vec3 GetNumberForegroundColor(int number);
  static glm::vec3 GetNumberBackgroundColor(int number);
  void SetBoardToWorld(glm::mat4 mat);

 private:
  Application *app_{};
  glm::mat4 board_to_world_{1.0f};
  font::Factory *font_factory_{nullptr};
  std::map<int, std::unique_ptr<DeviceModel>> foreground_device_models_;
  std::unique_ptr<DeviceModel> background_device_model_;
};
