#include "model.h"

Model::Model(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices) {
  vertices_ = vertices;
  indices_ = indices;
}

DeviceModel::DeviceModel(Application *app, uint32_t num_vertices, uint32_t num_indices) : app_(app) {
  app_->Core()->CreateBuffer(std::max<size_t>(num_vertices, 1) * sizeof(Vertex), graphics::BUFFER_TYPE_DYNAMIC,
                             &vertex_buffer_);
  app_->Core()->CreateBuffer(std::max<size_t>(num_indices, 1) * sizeof(uint32_t), graphics::BUFFER_TYPE_DYNAMIC,
                             &index_buffer_);
}

DeviceModel::DeviceModel(Application *app, const Model &model)
    : DeviceModel(app, model.Vertices().size(), model.Indices().size()) {
  UploadVertices(model.Vertices());
  UploadIndices(model.Indices());
}

void DeviceModel::UploadVertices(const std::vector<Vertex> &vertices) {
  const size_t size = vertices.size() * sizeof(Vertex);
  if (size > vertex_buffer_->Size()) {
    vertex_buffer_->Resize(size);
  }
  vertex_buffer_->UploadData(vertices.data(), size);
}

void DeviceModel::UploadIndices(const std::vector<uint32_t> &indices) {
  const size_t size = indices.size() * sizeof(uint32_t);
  if (size > index_buffer_->Size()) {
    index_buffer_->Resize(size);
  }
  index_buffer_->UploadData(indices.data(), size);
  index_count_ = static_cast<uint32_t>(indices.size());
}

std::vector<Vertex> ComposeVertices(const std::vector<glm::vec2> &positions, const glm::vec4 &color) {
  std::vector<Vertex> vertices;
  vertices.reserve(positions.size());
  for (auto &position : positions) {
    vertices.push_back({position, color});
  }
  return vertices;
}

glm::mat4 GetModelMatrix(const glm::vec2 &position, const glm::vec2 &scale, float depth) {
  return glm::translate(glm::mat4{1.0f}, glm::vec3{position + scale * 0.5f, depth}) *
         glm::scale(glm::mat4{1.0f}, glm::vec3{scale * 0.5f, 1.0f});
}

MixModel::MixModel(const std::vector<std::vector<Vertex>> &vertices, const std::vector<uint32_t> &indices) {
  for (size_t i = 1; i < vertices.size(); i++) {
    assert(vertices[0].size() == vertices[i].size());
  }
  vertices_ = vertices;
  mixed_model_ = std::make_optional<Model>(vertices[0], indices);
}

Model &MixModel::GetModel(float alpha, MixStyle mix_style) {
  auto &mixed_vertices = mixed_model_->Vertices();
  float index;
  alpha = std::modf(alpha, &index);
  int i0 = int(index) % int(vertices_.size());
  int i1 = (i0 + 1) % int(vertices_.size());

  auto &vertices0 = vertices_[i0];
  auto &vertices1 = vertices_[i1];
  switch (mix_style) {
    case MixStyle::kLinear:
      for (size_t i = 0; i < mixed_vertices.size(); i++) {
        mixed_vertices[i].position = Mix(vertices0[i].position, vertices1[i].position, alpha);
        mixed_vertices[i].color = Mix(vertices0[i].color, vertices1[i].color, alpha);
      }
      break;
    case MixStyle::kAngularClockwise:
      for (size_t i = 0; i < mixed_vertices.size(); i++) {
        auto p0 = glm::vec2{vertices0[i].position};
        auto p1 = glm::vec2{vertices1[i].position};
        auto am0 = glm::vec2{std::atan2(p0.x, p0.y), glm::length(p0)};
        auto am1 = glm::vec2{std::atan2(p1.x, p1.y), glm::length(p1)};
        if (am1.x < am0.x)
          am1.x += glm::pi<float>() * 2.0f;
        am1 = Mix(am0, am1, alpha);
        mixed_vertices[i].position = glm::vec3(glm::vec2(std::sin(am1.x), std::cos(am1.x)) * am1.y, 1.0f);
        mixed_vertices[i].color = Mix(vertices0[i].color, vertices1[i].color, alpha);
      }
      break;
  }
  return mixed_model_.value();
}
