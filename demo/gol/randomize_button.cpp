#include "randomize_button.h"

#include "glm/gtc/matrix_transform.hpp"

RandomizeButton::RandomizeButton(Application *app, std::vector<uint8_t> *cells, DeviceModel *background)
    : Button(app, 0, 0, 100, 100),
      cells_(cells),
      background_(background) {
  std::vector<glm::vec2> positions;
  std::vector<uint32_t> indices;
  // Match the rounded-square silhouette of the existing controls with a five-pip die.
  constexpr int segments = 64;
  for (int i = 0; i < segments; ++i) {
    float angle = glm::two_pi<float>() * float(i) / segments;
    glm::vec2 point{std::cos(angle), std::sin(angle)};
    float norm = std::pow(std::pow(std::abs(point.x), 8.0f) + std::pow(std::abs(point.y), 8.0f), 0.125f);
    positions.push_back(point * (0.60f / norm));
    positions.push_back(point * (0.49f / norm));
    uint32_t a = i * 2, b = ((i + 1) % segments) * 2;
    indices.insert(indices.end(), {a, b, a + 1, a + 1, b, b + 1});
  }
  for (glm::vec2 center : {glm::vec2{-0.27f, -0.27f}, {0.27f, -0.27f}, {0, 0}, {-0.27f, 0.27f}, {0.27f, 0.27f}}) {
    uint32_t start = positions.size();
    positions.push_back(center);
    constexpr int pip_segments = 24;
    for (int i = 0; i < pip_segments; ++i) {
      float angle = glm::two_pi<float>() * float(i) / pip_segments;
      positions.push_back(center + glm::vec2{std::cos(angle), std::sin(angle)} * 0.085f);
      indices.insert(indices.end(), {start, start + 1 + uint32_t(i), start + 1 + uint32_t((i + 1) % pip_segments)});
    }
  }
  icon_ =
      std::make_unique<DeviceModel>(app, Model(ComposeVertices(positions, glm::vec4{0.8f, 0.8f, 0.8f, 1.0f}), indices));
  background_color_ =
      MixValue<glm::vec4>({{0.10f, 0.16f, 0.24f, 1}, {0.12f, 0.20f, 0.30f, 1}, {0.18f, 0.28f, 0.40f, 1}});
}

void RandomizeButton::Update(float delta_time) {
  rotation_.Update(delta_time * 2.0f);
  background_animation_.Update(delta_time * 10.0f);
}

void RandomizeButton::Draw() {
  glm::vec2 position{left_, top_}, size{right_ - left_, bottom_ - top_};
  application_->DrawModel(
      background_, {GetModelMatrix(position, size, 0.6f), background_color_.GetValue(float(background_animation_)),
                    glm::uvec4{1, 0, 0, 0}});
  auto transform =
      GetModelMatrix(position, size, 0.4f) * glm::rotate(glm::mat4{1.0f}, float(rotation_), glm::vec3{0, 0, 1});
  application_->DrawModel(icon_.get(), {transform, glm::vec4{1.0f}, glm::uvec4{1, 0, 0, 0}});
}

void RandomizeButton::OnClick() {
  std::bernoulli_distribution alive(0.5);
  for (auto &cell : *cells_)
    cell = alive(random_engine_) ? 1 : 0;
  rotation_.AddTarget(glm::half_pi<float>());
}

void RandomizeButton::OnStateChange(int state) {
  background_animation_.UpdateTarget(float(state));
}
