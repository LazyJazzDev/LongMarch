#include "draw_n_gui.h"

#include "glm/gtc/matrix_transform.hpp"
#include "snow_mount/draw/draw_font.h"

DrawNGUI::DrawNGUI(CD::graphics::BackendAPI api) {
  CD::graphics::CreateCore(api, {}, &core_);
  core_->InitializeLogicalDeviceAutoSelect(false);
}

DrawNGUI::~DrawNGUI() {
  core_.reset();
}

void DrawNGUI::Run() {
  OnInit();
  while (!window_->ShouldClose()) {
    glfwPollEvents();
    OnUpdate();
    OnRender();
  }
  core_->WaitGPU();
  OnClose();
}

void DrawNGUI::OnInit() {
  core_->CreateWindowObject(1280, 720, "Draw & GUI", false, true, &window_);
  core_->CreateImage(1280, 720, CD::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &color_image_);
  window_->ResizeEvent().RegisterCallback([this](int width, int height) {
    core_->WaitGPU();
    color_image_.reset();
    core_->CreateImage(width, height, CD::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &color_image_);
  });

  snow_mount::draw::CreateCore(core_.get(), &draw_core_);
  draw_core_->CreateModel(&model_);
  std::vector<snow_mount::draw::Vertex> vertices;
  std::vector<uint32_t> indices;
  const int precision = 100;
  const float inv_precision = 1.0f / precision;
  for (int i = 0; i < precision; i++) {
    float angle = 2.0f * glm::pi<float>() * i * inv_precision;
    float x = glm::cos(angle);
    float y = glm::sin(angle);
    vertices.push_back({{x, y}, {x * 0.5f + 0.5f, y * 0.5f + 0.5f}, {1.0f, 1.0f, 1.0f, 1.0f}});
    indices.push_back(i);
    indices.push_back((i + 1) % precision);
    indices.push_back(precision);
  }
  vertices.push_back({{0.0f, 0.0f}, {0.5f, 0.5f}, {1.0f, 1.0f, 1.0f, 1.0f}});
  draw_core_->CreateTexture(256, 256, &texture_);
  std::vector<uint32_t> texture_data(256 * 256, 0xFFFFFFFF);
  for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 256; j++) {
      texture_data[i * 256 + j] = 0xFF000000 | ((i ^ j) << 16) | (j << 8) | (i);
    }
  }
  texture_->UploadData(texture_data.data());
  model_->SetModelData(vertices, indices);

  draw_core_->SetFontTypeFile(CD::FindAssetFile("fonts/simhei.ttf"));
  draw_core_->SetASCIIFontTypeFile(CD::FindAssetFile("fonts/georgia.ttf"));
}

void DrawNGUI::OnClose() {
  model_.reset();
  texture_.reset();
  draw_core_.reset();
  color_image_.reset();
  window_.reset();
}

void DrawNGUI::OnUpdate() {
  draw_core_->BeginDraw();
  auto extent = color_image_->Extent();
  draw_core_->CmdSetDrawRegion(0, 0, extent.width, extent.height);
  float alpha = 0.5f + 0.5f * glm::sin(glfwGetTime() * 5.0f);
  float theta = glfwGetTime();
  draw_core_->CmdDrawInstance(
      model_.get(), texture_.get(),
      glm::rotate(glm::scale(glm::translate(glm::rotate(glm::mat4{1.0f}, theta, glm::vec3{0.0f, 0.0f, 1.0f}),
                                            {1.0f, 0.0f, 0.0f}),
                             {0.5f, 0.5f, 1.0f}),
                  -2.0f * theta, glm::vec3{0.0f, 0.0f, 1.0f}),
      glm::vec4{1.0f, 1.0f, 1.0f, alpha});
  theta += glm::pi<float>();
  draw_core_->CmdDrawInstance(
      model_.get(), texture_.get(),
      glm::rotate(glm::scale(glm::translate(glm::rotate(glm::mat4{1.0f}, theta, glm::vec3{0.0f, 0.0f, 1.0f}),
                                            {0.0f, 0.0f, 0.0f}),
                             {0.5f, 0.5f, 1.0f}),
                  -2.0f * theta, glm::vec3{0.0f, 0.0f, 1.0f}),
      glm::vec4{1.0f, 1.0f, 1.0f, alpha});
  const int font_size = 128;
  draw_core_->GetFontCore()->SetFontSize(font_size);
  auto text_width_1 = draw_core_->GetTextWidth(u8"我爱你，中国！");
  auto text_width_2 = draw_core_->GetTextWidth(u8"I LOVE U, CHINA!");
  draw_core_->CmdDrawText({(extent.width - text_width_1) / 2, font_size * 0.9f}, u8"我爱你，中国！",
                          {1.0f, 1.0f, 1.0f, 1.0f});
  draw_core_->CmdDrawText({(extent.width - text_width_2) / 2, font_size * 0.9f + font_size}, u8"I LOVE U, CHINA!",
                          {1.0f, 1.0f, 1.0f, 1.0f});
  draw_core_->EndDraw();
}

void DrawNGUI::OnRender() {
  std::unique_ptr<CD::graphics::CommandContext> ctx;
  core_->CreateCommandContext(&ctx);
  ctx->CmdClearImage(color_image_.get(), {0.6f, 0.7f, 0.8f, 1.0f});
  draw_core_->Render(ctx.get(), color_image_.get());
  ctx->CmdPresent(window_.get(), color_image_.get());
  core_->SubmitCommandContext(ctx.get());
}
