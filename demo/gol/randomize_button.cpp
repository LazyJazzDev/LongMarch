#include "randomize_button.h"

#include "glm/gtc/matrix_transform.hpp"

RandomizeButton::RandomizeButton(Application *app, std::vector<uint8_t> *cells, DeviceModel *background)
    : Button(app, 0, 0, 100, 100),
      cells_(cells),
      background_(background) {
  constexpr float half_extent = 0.43f;
  face_ = std::make_unique<DeviceModel>(app, Model(ComposeVertices({{-half_extent, -half_extent},
                                                                    {half_extent, -half_extent},
                                                                    {half_extent, half_extent},
                                                                    {-half_extent, half_extent}},
                                                                   glm::vec4{1.0f}),
                                                   {0, 1, 2, 0, 2, 3}));
  // Opposite faces add up to seven. Each face has its own real, circular pip geometry.
  const std::array<std::vector<glm::vec2>, 6> centers{
      {{{0, 0}},
       {{-0.22f, -0.22f}, {-0.22f, 0}, {-0.22f, 0.22f}, {0.22f, -0.22f}, {0.22f, 0}, {0.22f, 0.22f}},
       {{-0.22f, -0.22f}, {0.22f, 0.22f}},
       {{-0.22f, -0.22f}, {0.22f, -0.22f}, {0, 0}, {-0.22f, 0.22f}, {0.22f, 0.22f}},
       {{-0.22f, -0.22f}, {0, 0}, {0.22f, 0.22f}},
       {{-0.22f, -0.22f}, {0.22f, -0.22f}, {-0.22f, 0.22f}, {0.22f, 0.22f}}}};
  for (size_t face = 0; face < centers.size(); ++face) {
    std::vector<glm::vec2> positions;
    std::vector<uint32_t> indices;
    for (auto center : centers[face]) {
      uint32_t start = positions.size();
      positions.push_back(center);
      constexpr int segments = 24;
      for (int i = 0; i < segments; ++i) {
        float angle = glm::two_pi<float>() * float(i) / segments;
        positions.push_back(center + glm::vec2{std::cos(angle), std::sin(angle)} * 0.063f);
        indices.insert(indices.end(), {start, start + 1 + uint32_t(i), start + 1 + uint32_t((i + 1) % segments)});
      }
    }
    pips_[face] = std::make_unique<DeviceModel>(app, Model(ComposeVertices(positions, glm::vec4{1.0f}), indices));
  }
  background_color_ =
      MixValue<glm::vec4>({{0.10f, 0.16f, 0.24f, 1}, {0.12f, 0.20f, 0.30f, 1}, {0.18f, 0.28f, 0.40f, 1}});
}

void RandomizeButton::Update(float delta_time) {
  rotation_.Update(delta_time * 1.5f);
  background_animation_.Update(delta_time * 10.0f);
}

void RandomizeButton::Draw() {
  glm::vec2 position{left_, top_}, size{right_ - left_, bottom_ - top_};
  application_->DrawModel(
      background_, {GetModelMatrix(position, size, 0.6f), background_color_.GetValue(float(background_animation_)),
                    glm::uvec4{1, 0, 0, 0}});
  // The renderer accepts planar vertices, but its instance matrix places each plane in 3D.
  // Keep cube depth in the UI's [0, 1] depth range, independently of the pixel scale.
  auto placement = glm::translate(glm::mat4{1.0f}, glm::vec3{position + size * 0.5f, 0.32f}) *
                   glm::scale(glm::mat4{1.0f}, glm::vec3{size * 0.5f, 0.16f});
  auto orientation = glm::rotate(glm::mat4{1.0f}, 0.50f, glm::vec3{1, 0, 0}) *
                     glm::rotate(glm::mat4{1.0f}, 0.60f, glm::vec3{0, 1, 0}) *
                     glm::rotate(glm::mat4{1.0f}, float(rotation_), glm::normalize(glm::vec3{1, 1, 1}));
  const std::array<glm::vec3, 6> normals{{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};
  auto light = glm::normalize(glm::vec3{-0.4f, -0.6f, -1.0f});
  for (size_t i = 0; i < normals.size(); ++i) {
    auto normal = normals[i];
    auto rotated_normal = glm::mat3(orientation) * normal;
    if (rotated_normal.z >= 0.0f)
      continue;
    auto tangent =
        std::abs(normal.y) > 0.5f ? glm::vec3{1, 0, 0} : glm::normalize(glm::cross(glm::vec3{0, 1, 0}, normal));
    glm::mat4 frame{glm::vec4{tangent, 0}, glm::vec4{glm::cross(normal, tangent), 0}, glm::vec4{normal, 0},
                    glm::vec4{normal * 0.43f, 1}};
    float shade = 0.48f + 0.52f * std::max(0.0f, glm::dot(rotated_normal, light));
    auto transform = placement * orientation * frame;
    application_->DrawModel(face_.get(),
                            {transform, glm::vec4{glm::vec3{0.96f, 0.97f, 1.0f} * shade, 1}, glm::uvec4{0}});
    // Offset the pips along the face normal so depth testing remains stable during a tumble.
    transform = transform * glm::translate(glm::mat4{1.0f}, glm::vec3{0, 0, 0.002f});
    application_->DrawModel(pips_[i].get(), {transform, glm::vec4{0.08f, 0.12f, 0.18f, 1}, glm::uvec4{0}});
  }
}

void RandomizeButton::OnClick() {
  std::bernoulli_distribution alive(0.5);
  for (auto &cell : *cells_)
    cell = alive(random_engine_) ? 1 : 0;
  rotation_.AddTarget(glm::two_pi<float>() * (4.0f / 3.0f));
}

void RandomizeButton::OnStateChange(int state) {
  background_animation_.UpdateTarget(float(state));
}
