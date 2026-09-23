#pragma once
#include "application.h"
#include "glm/gtc/matrix_transform.hpp"
#include "interpolation.h"

struct Vertex {
  glm::vec2 position;
  glm::vec4 color;
};

class Model {
 public:
  Model(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices);

  [[nodiscard]] std::vector<Vertex> &Vertices() {
    return vertices_;
  }

  [[nodiscard]] const std::vector<Vertex> &Vertices() const {
    return vertices_;
  }

  [[nodiscard]] const std::vector<uint32_t> &Indices() const {
    return indices_;
  }

 private:
  std::vector<Vertex> vertices_;
  std::vector<uint32_t> indices_;
};

enum class MixStyle : uint32_t { kLinear, kAngularClockwise };

class MixModel {
 public:
  MixModel(const std::vector<std::vector<Vertex>> &vertices, const std::vector<uint32_t> &indices);
  Model &GetModel(float alpha, MixStyle mix_style);

 private:
  std::vector<std::vector<Vertex>> vertices_;
  std::optional<Model> mixed_model_;
};

class DeviceModel {
 public:
  DeviceModel(Application *app, uint32_t num_vertices, uint32_t num_indices);
  DeviceModel(Application *app, const Model &model);

  [[nodiscard]] graphics::Buffer *VertexBuffer() const {
    return vertex_buffer_.get();
  }

  [[nodiscard]] graphics::Buffer *IndexBuffer() const {
    return index_buffer_.get();
  }

  [[nodiscard]] uint32_t IndexCount() const {
    return index_count_;
  }

  void UploadVertices(const std::vector<Vertex> &vertices);
  void UploadIndices(const std::vector<uint32_t> &indices);

 private:
  Application *app_{};
  std::unique_ptr<graphics::Buffer> vertex_buffer_;
  std::unique_ptr<graphics::Buffer> index_buffer_;
  uint32_t index_count_{};
};

std::vector<Vertex> ComposeVertices(const std::vector<glm::vec2> &positions, const glm::vec4 &color);

glm::mat4 GetModelMatrix(const glm::vec2 &position, const glm::vec2 &scale, float depth = 0.5f);

glm::mat4 GetModelMatrixZO(const glm::vec2 &position, const glm::vec2 &scale, float depth = 0.5f);
