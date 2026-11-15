#pragma once

#include <array>
#include <string>

#include "application/model.h"

namespace life_demo {
inline Model StrokeText(const std::string &text) {
  // Normal-width glyphs with bold, uniform strokes and small rounded corners.
  using Path = std::vector<glm::vec2>;
  static const std::array<std::vector<Path>, 13> glyphs{
      {{{{0, 0}, {6, 0}, {6, 6}, {0, 6}, {0, 0}}},
       {{{1.5f, 1.5f}, {3, 0}, {3, 6}}, {{0, 6}, {6, 6}}},
       {{{0, 0}, {6, 0}, {6, 3}, {0, 3}, {0, 6}, {6, 6}}},
       {{{0, 0}, {6, 0}, {6, 6}, {0, 6}}, {{1.5f, 3}, {6, 3}}},
       {{{0, 0}, {0, 3}, {6, 3}}, {{6, 0}, {6, 6}}},
       {{{6, 0}, {0, 0}, {0, 3}, {6, 3}, {6, 6}, {0, 6}}},
       {{{6, 0}, {0, 0}, {0, 6}, {6, 6}, {6, 3}, {0, 3}}},
       {{{0, 0}, {6, 0}, {6, 6}}},
       {{{0, 0}, {6, 0}, {6, 6}, {0, 6}, {0, 0}}, {{0, 3}, {6, 3}}},
       {{{6, 3}, {0, 3}, {0, 0}, {6, 0}, {6, 6}, {0, 6}}},
       {{{0, 0}, {0, 6}, {3, 6}, {3, 2.5f}}, {{3, 6}, {6, 6}, {6, 0}}},
       {{{0, 0}, {0, 6}}, {{6, 0}, {6, 6}}, {{0, 3}, {6, 3}}},
       {{{0, 3}, {6, 3}}}}};
  constexpr float advance = 6.2f;
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  auto capsule = [&](glm::vec2 from, glm::vec2 to) {
    constexpr float radius = 0.95f;
    constexpr int steps = 10;
    const float angle = std::atan2(to.y - from.y, to.x - from.x);
    const uint32_t base = uint32_t(vertices.size());
    for (int cap = 0; cap < 2; ++cap) {
      const auto center = cap == 0 ? to : from;
      for (int j = 0; j <= steps; ++j) {
        const float theta = angle - glm::half_pi<float>() + (float(cap) + float(j) / steps) * glm::pi<float>();
        vertices.push_back({center + radius * glm::vec2{std::cos(theta), std::sin(theta)}, glm::vec4{1.0f}});
      }
    }
    for (uint32_t j = 1; j + 1 < 2 * (steps + 1); ++j)
      for (auto index : {base, base + j, base + j + 1})
        indices.push_back(index);
  };
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == ' ')
      continue;
    const int glyph = text[i] == 'W' ? 10 : text[i] == 'H' ? 11 : text[i] == '-' ? 12 : text[i] - '0';
    for (auto path : glyphs[glyph]) {
      // Narrow the centerlines before stroking, so vertical strokes retain their weight.
      for (auto &point : path)
        point.x *= 2.0f / 3.0f;
      const bool closed = path.front() == path.back();
      const size_t count = path.size() - (closed ? 1 : 0);
      Path rounded;
      for (size_t j = 0; j < count; ++j) {
        if (!closed && (j == 0 || j + 1 == count)) {
          rounded.push_back(path[j]);
          continue;
        }
        const auto before = path[(j + count - 1) % count] - path[j];
        const auto after = path[(j + 1) % count] - path[j];
        const float trim = std::min(0.65f, std::min(glm::length(before), glm::length(after)) * 0.4f);
        const auto start = path[j] + glm::normalize(before) * trim;
        const auto end = path[j] + glm::normalize(after) * trim;
        for (int sample = 0; sample <= 6; ++sample) {
          const float t = float(sample) / 6.0f;
          rounded.push_back(glm::mix(glm::mix(start, path[j], t), glm::mix(path[j], end, t), t));
        }
      }
      if (closed)
        rounded.push_back(rounded.front());
      for (size_t j = 1; j < rounded.size(); ++j)
        capsule(rounded[j - 1] + glm::vec2{float(i) * advance, 0}, rounded[j] + glm::vec2{float(i) * advance, 0});
    }
  }
  return Model(vertices, indices);
}
}  // namespace life_demo
