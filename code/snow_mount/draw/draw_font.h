#pragma once

#include "ft2build.h"
#include "snow_mount/draw/draw_util.h"
#include FT_FREETYPE_H

namespace XS::draw {

struct CharModel {
  Texture *char_tex_;
  float advance_x_;
  float advance_y_;
  float bearing_x_;
  float bearing_y_;
  float width_;
  float height_;
};

class FontCore {
 public:
  FontCore(Core *core);
  ~FontCore();

  void SetFontTypeFile(const std::string &filename);
  void SetASCIIFontTypeFile(const std::string &filename);
  void SetFontSize(uint32_t size);
  CharModel GetCharModel(uint32_t char_code);

  uint32_t GetFontSize() const;

 private:
  void UpdateActiveCharMap();

  typedef std::map<uint32_t, CharModel> CharMap;
  typedef std::map<uint32_t, CharMap> SizeMap;
  typedef std::map<FT_Face, SizeMap> FaceMap;

  Core *core_;
  FT_Library library_;
  std::map<std::string, FT_Face> faces_;
  FT_Face active_face_;
  uint32_t active_size_;
  FaceMap face_map_;
  FT_Face active_ascii_face_{nullptr};
};
}  // namespace XS::draw
