#include "snowberg/draw/draw_font.h"

#include "snowberg/draw/draw_commands.h"
#include "snowberg/draw/draw_core.h"
#include "snowberg/draw/draw_model.h"
#include "snowberg/draw/draw_texture.h"

namespace snowberg::draw {

FontCore::FontCore(Core *core) : core_(core) {
  FT_Init_FreeType(&library_);
  active_size_ = 16;
}

FontCore::~FontCore() {
  for (auto &size_map : face_map_) {
    for (auto &char_map : size_map.second) {
      for (auto &char_model : char_map.second) {
        delete char_model.second.char_tex_;
      }
    }
  }
  face_map_.clear();
  for (auto &face : faces_) {
    FT_Done_Face(face.second);
  }
  FT_Done_FreeType(library_);
}

void FontCore::SetFontTypeFile(const std::string &filename) {
  if (faces_.find(filename) == faces_.end()) {
    FT_Face face;
    FT_New_Face(library_, filename.c_str(), 0, &face);
    faces_[filename] = face;
  }
  active_face_ = faces_[filename];
  UpdateActiveCharMap();
}

void FontCore::SetASCIIFontTypeFile(const std::string &filename) {
  if (faces_.find(filename) == faces_.end()) {
    FT_Face face;
    FT_New_Face(library_, filename.c_str(), 0, &face);
    faces_[filename] = face;
  }
  active_ascii_face_ = faces_[filename];
  UpdateActiveCharMap();
}

void FontCore::SetFontSize(uint32_t size) {
  active_size_ = size;
  UpdateActiveCharMap();
}

CharModel FontCore::GetCharModel(uint32_t char_code) {
  auto active_face = active_face_;
  auto active_char_map = &face_map_[active_face_][active_size_];
  if (active_ascii_face_ && char_code < 128) {
    active_face = active_ascii_face_;
    active_char_map = &face_map_[active_ascii_face_][active_size_];
  }
  if (active_char_map->find(char_code) == active_char_map->end()) {
    FT_Load_Char(active_face, char_code, FT_LOAD_RENDER);
    auto glyph = active_face->glyph;
    auto bitmap = glyph->bitmap;
    auto width = bitmap.width;
    auto height = bitmap.rows;
    auto advance_x = glyph->advance.x * (1.0f / 64.0f);
    auto advance_y = glyph->advance.y * (1.0f / 64.0f);
    auto bearing_x = glyph->bitmap_left;
    auto bearing_y = glyph->bitmap_top;
    Texture *tex = nullptr;
    if (width && height) {
      core_->CreateTexture(width, height, &tex);
      std::vector<uint32_t> data(width * height);
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          data[y * width + x] = (bitmap.buffer[y * width + x] << 24) | 0x00ffffff;
        }
      }
      tex->UploadData(data.data());
    }
    CharModel model{};
    model.char_tex_ = tex;
    model.advance_x_ = advance_x;
    model.advance_y_ = advance_y;
    model.bearing_x_ = bearing_x;
    model.bearing_y_ = bearing_y;
    model.width_ = width;
    model.height_ = height;
    (*active_char_map)[char_code] = model;
  }
  return active_char_map->at(char_code);
}

uint32_t FontCore::GetFontSize() const {
  return active_size_;
}

void FontCore::UpdateActiveCharMap() {
  FT_Set_Pixel_Sizes(active_face_, 0, active_size_);
  if (active_ascii_face_) {
    FT_Set_Pixel_Sizes(active_ascii_face_, 0, active_size_);
  }
}

}  // namespace snowberg::draw
