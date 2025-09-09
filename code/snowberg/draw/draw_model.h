#pragma once
#include "snowberg/draw/draw_util.h"

namespace snowberg::draw {

class Model {
 public:
  Model(Core *core);

  Core *DrawCore() const {
    return core_;
  }

  graphics::Buffer *VertexBuffer() const {
    return vertex_buffer_.get();
  }

  graphics::Buffer *IndexBuffer() const {
    return index_buffer_.get();
  }

  uint32_t IndexCount() const {
    return index_count_;
  }

  void SetModelData(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices);

 private:
  Core *core_;
  std::unique_ptr<graphics::Buffer> vertex_buffer_;
  std::unique_ptr<graphics::Buffer> index_buffer_;
  uint32_t index_count_{0};
};

}  // namespace snowberg::draw
