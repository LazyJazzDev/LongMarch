#pragma once
#include "snowberg/visualizer/visualizer_util.h"

namespace snowberg::visualizer {
class Mesh {
  friend class Core;
  Mesh(const std::shared_ptr<Core> &core);

 public:
  std::shared_ptr<Core> GetCore() const;

  void SetVertices(const Vertex *vertices);
  void SetIndices(const uint32_t *indices);
  void SetVertices(const Vertex *vertices, int num_vertex);
  void SetIndices(const uint32_t *indices, int num_indices);

  int VertexCount() const;
  int IndexCount() const;

  graphics::Buffer *GetVertexBuffer() const;
  graphics::Buffer *GetIndexBuffer() const;

 private:
  std::shared_ptr<Core> core_;
  std::unique_ptr<graphics::Buffer> vertex_buffer_;
  std::unique_ptr<graphics::Buffer> index_buffer_;
  int num_vertices_;
  int num_indices_;
};
}  // namespace snowberg::visualizer
