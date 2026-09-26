#pragma once

#include <cmath>

#include "imgui.h"

namespace grassland::imgui {

inline thread_local bool linear_vertex_colors = false;

// Keep ImGui's public vertex ABI unchanged; only the GPU upload uses float colors.
struct GPUVertex {
  ImVec2 pos;
  ImVec2 uv;
  ImVec4 col;
};

inline void UploadVertices(void *destination, const ImDrawVert *source, int count) {
  auto *vertices = static_cast<GPUVertex *>(destination);
  for (int i = 0; i < count; ++i) {
    auto color = ImGui::ColorConvertU32ToFloat4(source[i].col);
    if (linear_vertex_colors) {
      auto decode = [](float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); };
      color.x = decode(color.x);
      color.y = decode(color.y);
      color.z = decode(color.z);
    }
    vertices[i] = {source[i].pos, source[i].uv, color};
  }
}

}  // namespace grassland::imgui
