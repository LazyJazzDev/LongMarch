#pragma once

#include <map>
#include <string>
#include <vector>

#include "ft2build.h"
#include "long_march.h"
#include FT_FREETYPE_H

namespace font {

typedef wchar_t Char_T;

// Glyph geometry as a triangle mesh, in units of the font's pixel size.
struct Mesh {
  Mesh();
  Mesh(const std::vector<glm::vec2> &vertices, const std::vector<uint32_t> &indices, float advance);
  Mesh(const std::vector<glm::vec2> &triangle_vertices, float advection);
  std::vector<glm::vec2> &GetVertices();
  [[nodiscard]] const std::vector<glm::vec2> &GetVertices() const;
  std::vector<uint32_t> &GetIndices();
  [[nodiscard]] const std::vector<uint32_t> &GetIndices() const;
  float &GetAdvance();
  [[nodiscard]] float GetAdvance() const;

  std::vector<glm::vec2> vertices;
  std::vector<uint32_t> indices;
  float advance{0.0f};
};

// Triangulates glyph outlines loaded with FreeType, so that text is drawn as
// ordinary vector geometry.
class Factory {
 public:
  explicit Factory(const std::string &font_file_path);
  ~Factory();
  const Mesh &GetChar(Char_T c);
  Mesh GetString(const std::wstring &wide_str);

 private:
  void LoadChar(Char_T c);
  FT_Library library_{};
  FT_Face face_{};
  std::map<Char_T, Mesh> loaded_fonts_;
};

}  // namespace font
