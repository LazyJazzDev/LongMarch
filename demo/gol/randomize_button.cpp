#include "randomize_button.h"

#include <array>

#include "glm/gtc/matrix_transform.hpp"

namespace {
const std::array<glm::vec3, 6> kNormals{{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};
constexpr std::array<uint32_t, 6> kFaceValues{1, 6, 2, 5, 3, 4};

glm::mat4 FaceFrame(glm::vec3 normal) {
  auto tangent =
      std::abs(normal.y) > 0.5f ? glm::vec3{1, 0, 0} : glm::normalize(glm::cross(glm::vec3{0, 1, 0}, normal));
  return {glm::vec4{tangent, 0}, glm::vec4{glm::cross(normal, tangent), 0}, glm::vec4{normal, 0},
          glm::vec4{normal * 0.43f, 1}};
}

glm::quat DisplayOrientation(int face, int quarter_turns) {
  auto tilt = glm::angleAxis(0.50f, glm::vec3{1, 0, 0}) * glm::angleAxis(0.60f, glm::vec3{0, 1, 0});
  auto roll = glm::angleAxis(float(quarter_turns) * glm::half_pi<float>(), glm::vec3{0, 0, 1});
  auto toward_camera = glm::angleAxis(glm::pi<float>(), glm::vec3{1, 0, 0});
  auto align_face = glm::quat_cast(glm::transpose(glm::mat3(FaceFrame(kNormals[face]))));
  return glm::normalize(tilt * roll * toward_camera * align_face);
}
}  // namespace

RandomizeButton::RandomizeButton(Application *app, std::vector<uint8_t> *cells, DeviceModel *background)
    : Button(app, 0, 0, 100, 100),
      cells_(cells),
      background_(background) {
  // The fragment shader cuts rounded corners and pip holes from this reusable face.
  face_ = std::make_unique<DeviceModel>(
      app, Model(ComposeVertices({{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}, glm::vec4{1.0f}), {0, 1, 2, 0, 2, 3}));
  orientation_ = start_orientation_ = target_orientation_ = DisplayOrientation(selected_face_, 0);
  background_color_ =
      MixValue<glm::vec4>({{0.10f, 0.16f, 0.24f, 1}, {0.12f, 0.20f, 0.30f, 1}, {0.18f, 0.28f, 0.40f, 1}});
}

void RandomizeButton::Update(float delta_time) {
  if (rotation_progress_ < 1.0f) {
    rotation_progress_ = std::min(1.0f, rotation_progress_ + delta_time / 0.8f);
    float t = rotation_progress_ * rotation_progress_ * (3.0f - 2.0f * rotation_progress_);
    // A complete tumble returns to the interpolated pose exactly at the end.
    orientation_ = glm::normalize(glm::angleAxis(glm::two_pi<float>() * t, spin_axis_) *
                                  glm::slerp(start_orientation_, target_orientation_, t));
    if (rotation_progress_ == 1.0f)
      orientation_ = target_orientation_;
  }
  background_animation_.Update(delta_time * 10.0f);
}

void RandomizeButton::Draw() {
  glm::vec2 position{left_, top_}, size{right_ - left_, bottom_ - top_};
  application_->DrawModel(
      background_, {GetModelMatrix(position, size, 0.6f), background_color_.GetValue(float(background_animation_)),
                    glm::uvec4{1, 0, 0, 0}});
  // All visible faces belong to a convex cube, so cull back faces and let them share one
  // depth: they never overlap and the die needs no sorting. The small depth extent below
  // exists only so the shader can converge the plates toward a vanishing point.
  auto placement = glm::translate(glm::mat4{1.0f}, glm::vec3{position + size * 0.5f, 0.34f}) *
                   glm::scale(glm::mat4{1.0f}, glm::vec3{size * 0.5f, 0.10f});
  auto orientation = glm::mat4_cast(orientation_);
  for (size_t i = 0; i < kNormals.size(); ++i) {
    auto normal = kNormals[i];
    auto rotated_normal = glm::mat3(orientation) * normal;
    if (rotated_normal.z >= 0.0f)
      continue;
    // A 0.38 half-width on faces spaced 0.43 from the center leaves open seams.
    auto transform =
        placement * orientation * FaceFrame(normal) * glm::scale(glm::mat4{1.0f}, glm::vec3{0.38f, 0.38f, 1});
    application_->DrawModel(face_.get(),
                            {transform, glm::vec4{0.96f, 0.97f, 1.0f, 1.0f}, glm::uvec4{3, kFaceValues[i], 0, 0}});
  }
}

void RandomizeButton::OnClick() {
  std::bernoulli_distribution alive(0.5);
  for (auto &cell : *cells_)
    cell = alive(random_engine_) ? 1 : 0;
  // Pick uniformly from the other five faces; use a separate random roll for variety.
  int next_face = std::uniform_int_distribution<int>(0, 4)(random_engine_);
  if (next_face >= selected_face_)
    ++next_face;
  selected_face_ = next_face;
  start_orientation_ = orientation_;
  target_orientation_ = DisplayOrientation(selected_face_, std::uniform_int_distribution<int>(0, 3)(random_engine_));
  std::uniform_real_distribution<float> axis(-1.0f, 1.0f);
  do {
    spin_axis_ = glm::vec3{axis(random_engine_), axis(random_engine_), axis(random_engine_)};
  } while (glm::dot(spin_axis_, spin_axis_) < 0.01f);
  spin_axis_ = glm::normalize(spin_axis_);
  rotation_progress_ = 0.0f;
}

void RandomizeButton::OnStateChange(int state) {
  background_animation_.UpdateTarget(float(state));
}
