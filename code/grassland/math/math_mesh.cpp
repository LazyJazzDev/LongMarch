#include "grassland/math/math_mesh.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

namespace grassland {

template <typename Scalar>
int Mesh<Scalar>::MergeVertices() {
  std::vector<Vector3<Scalar>> new_positions;
  std::vector<Vector3<Scalar>> new_normals;
  std::vector<Vector3<Scalar>> new_tangents;
  std::vector<Vector2<Scalar>> new_tex_coords;
  std::vector<float> new_signals;
  auto vector_comparator = [](const Vector3<Scalar> &a, const Vector3<Scalar> &b) {
    if (a[0] != b[0]) {
      return a[0] < b[0];
    }
    if (a[1] != b[1]) {
      return a[1] < b[1];
    }
    return a[2] < b[2];
  };
  auto vertex_index_comparator = [this, vector_comparator](uint32_t a, uint32_t b) {
    if (positions_[a] != positions_[b]) {
      return vector_comparator(positions_[a], positions_[b]);
    }
    if (!normals_.empty() && normals_[a] != normals_[b]) {
      return vector_comparator(normals_[a], normals_[b]);
    }
    if (!tangents_.empty() && tangents_[a] != tangents_[b]) {
      return vector_comparator(tangents_[a], tangents_[b]);
    }
    if (!tex_coords_.empty() && tex_coords_[a] != tex_coords_[b]) {
      return tex_coords_[a][0] < tex_coords_[b][0] ||
             (tex_coords_[a][0] == tex_coords_[b][0] && tex_coords_[a][1] < tex_coords_[b][1]);
    }
    if (!signals_.empty() && signals_[a] != signals_[b]) {
      return signals_[a] < signals_[b];
    }
    return false;
  };
  std::map<uint32_t, uint32_t, decltype(vertex_index_comparator)> vertex_map(vertex_index_comparator);
  std::vector<uint32_t> index_map;

  for (size_t i = 0; i < num_vertices_; i++) {
    if (vertex_map.find(i) == vertex_map.end()) {
      new_positions.push_back(positions_[i]);
      if (!normals_.empty()) {
        new_normals.push_back(normals_[i]);
      }
      if (!tangents_.empty()) {
        new_tangents.push_back(tangents_[i]);
      }
      if (!tex_coords_.empty()) {
        new_tex_coords.push_back(tex_coords_[i]);
      }
      if (!signals_.empty()) {
        new_signals.push_back(signals_[i]);
      }
      vertex_map[i] = new_positions.size() - 1;
      index_map.push_back(new_positions.size() - 1);
    } else {
      index_map.push_back(vertex_map[i]);
    }
  }

  positions_ = new_positions;
  normals_ = new_normals;
  tangents_ = new_tangents;
  tex_coords_ = new_tex_coords;
  signals_ = new_signals;

  for (size_t i = 0; i < num_indices_; i++) {
    indices_[i] = index_map[indices_[i]];
  }

  num_vertices_ = positions_.size();

  num_indices_ = 0;

  for (size_t i = 0; i < indices_.size(); i += 3) {
    auto i0 = indices_[i];
    auto i1 = indices_[i + 1];
    auto i2 = indices_[i + 2];
    if (i0 != i1 && i0 != i2 && i1 != i2) {
      indices_[num_indices_ + 0] = i0;
      indices_[num_indices_ + 1] = i1;
      indices_[num_indices_ + 2] = i2;
      num_indices_ += 3;
    }
  }
  indices_.resize(num_indices_);

  return 0;
}

template <typename Scalar>
Mesh<Scalar>::Mesh(size_t num_vertices,
                   size_t num_indices,
                   const uint32_t *indices,
                   const Vector3<Scalar> *positions,
                   const Vector3<Scalar> *normals,
                   const Vector2<Scalar> *tex_coords,
                   const Vector3<Scalar> *tangents) {
  num_vertices_ = num_vertices;
  num_indices_ = num_indices;
  indices_.resize(num_indices);
  if (indices) {
    std::copy(indices, indices + num_indices, indices_.begin());
  }
  positions_.resize(num_vertices);
  if (positions) {
    std::copy(positions, positions + num_vertices, positions_.begin());
  }
  if (normals) {
    normals_.resize(num_vertices);
    std::copy(normals, normals + num_vertices, normals_.begin());
    if (tangents) {
      tangents_.resize(num_vertices);
      std::copy(tangents, tangents + num_vertices, tangents_.begin());
      signals_.resize(num_vertices);
      std::fill(signals_.begin(), signals_.end(), 1.0f);
    }
  }
  if (tex_coords) {
    tex_coords_.resize(num_vertices);
    std::copy(tex_coords, tex_coords + num_vertices, tex_coords_.begin());
  }
}

template <typename Scalar>
int Mesh<Scalar>::LoadObjFile(const std::string &filename) {
  if (!std::filesystem::is_regular_file(filename)) {
    return -1;
  }

  std::filesystem::path path(filename);
  tinyobj::ObjReaderConfig reader_config;
  reader_config.mtl_search_path = path.parent_path().string();
  tinyobj::ObjReader reader;

  if (!reader.ParseFromFile(filename, reader_config)) {
    if (!reader.Error().empty()) {
      LogError("TinyObjReader: {}", reader.Error());
    }
    return -1;
  }

  if (!reader.Warning().empty()) {
    LogWarning("TinyObjReader: {}", reader.Warning());
  }

  auto &attrib = reader.GetAttrib();
  auto &shapes = reader.GetShapes();
  auto &materials = reader.GetMaterials();

  std::vector<Vector3<Scalar>> positions;
  std::vector<Vector3<Scalar>> normals;
  std::vector<Vector2<Scalar>> tex_coords;

  std::vector<uint32_t> indices;

  bool have_normal = false;
  bool have_texcoord = false;

  // Loop over shapes
  for (size_t s = 0; s < shapes.size(); s++) {
    // Loop over faces(polygon)
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

      // Loop over vertices in the face.
      for (size_t v = 0; v < fv; v++) {
        Vector3<Scalar> position;
        Vector3<Scalar> normal;
        Vector2<Scalar> tex_coord;

        // access to vertex
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
        tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
        tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
        tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
        position = Vector3<Scalar>(vx, vy, vz);

        // Check if `normal_index` is zero or positive. negative = no normal
        // data
        if (idx.normal_index >= 0) {
          tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
          tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
          tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
          normal = Vector3<Scalar>(nx, ny, nz);
          have_normal = true;
        }

        // Check if `texcoord_index` is zero or positive. negative = no texcoord
        // data
        if (idx.texcoord_index >= 0) {
          tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
          tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
          tex_coord = Vector2<Scalar>(tx, ty);
          have_texcoord = true;
        }

        positions.push_back(position);
        normals.push_back(normal);
        tex_coords.push_back(tex_coord);

        // Optional: vertex colors
        // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
        // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
        // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
        if (v >= 2) {
          indices.push_back(index_offset);
          indices.push_back(index_offset + v - 1);
          indices.push_back(index_offset + v);
        }
      }

      index_offset += fv;

      // per-face material
      // shapes[s].mesh.material_ids[f];
    }
  }

  positions_ = positions;
  if (have_normal)
    normals_ = normals;
  else
    normals_.clear();
  if (have_texcoord)
    tex_coords_ = tex_coords;
  else
    tex_coords_.clear();
  tangents_.clear();
  signals_.clear();
  indices_ = indices;

  num_vertices_ = positions_.size();
  num_indices_ = indices.size();

  MergeVertices();

  return 0;
}

template <typename Scalar>
int Mesh<Scalar>::SaveObjFile(const std::string &filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    return -1;
  }
  for (size_t i = 0; i < num_vertices_; i++) {
    file << "v " << positions_[i][0] << " " << positions_[i][1] << " " << positions_[i][2] << std::endl;
  }
  if (!normals_.empty()) {
    for (size_t i = 0; i < num_vertices_; i++) {
      file << "vn " << normals_[i][0] << " " << normals_[i][1] << " " << normals_[i][2] << std::endl;
    }
  }
  if (!tex_coords_.empty()) {
    for (size_t i = 0; i < num_vertices_; i++) {
      file << "vt " << tex_coords_[i][0] << " " << tex_coords_[i][1] << std::endl;
    }
  }
  auto output_face_index = [this, &file](size_t i) {
    file << i;
    file << "/";
    if (!tex_coords_.empty()) {
      file << i;
    }
    file << "/";
    if (!normals_.empty()) {
      file << i;
    }
  };
  for (size_t i = 0; i < num_indices_; i += 3) {
    file << "f ";
    output_face_index(indices_[i] + 1);
    file << " ";
    output_face_index(indices_[i + 1] + 1);
    file << " ";
    output_face_index(indices_[i + 2] + 1);
    file << std::endl;
  }
  return 0;
}

template <typename Scalar>
int Mesh<Scalar>::SplitVertices() {
  std::vector<Vector3<Scalar>> new_positions;
  std::vector<Vector3<Scalar>> new_normals;
  std::vector<Vector3<Scalar>> new_tangents;
  std::vector<Vector2<Scalar>> new_tex_coords;
  std::vector<float> new_signals;

  for (size_t i = 0; i < num_indices_; i++) {
    new_positions.push_back(positions_[indices_[i]]);
    if (!normals_.empty()) {
      new_normals.push_back(normals_[indices_[i]]);
    }
    if (!tangents_.empty()) {
      new_tangents.push_back(tangents_[indices_[i]]);
      new_signals.push_back(signals_[indices_[i]]);
    }
    if (!tex_coords_.empty()) {
      new_tex_coords.push_back(tex_coords_[indices_[i]]);
    }
    indices_[i] = i;
  }

  positions_ = new_positions;
  normals_ = new_normals;
  tangents_ = new_tangents;
  tex_coords_ = new_tex_coords;
  signals_ = new_signals;
  num_vertices_ = num_indices_;

  return 0;
}

template <typename Scalar>
int Mesh<Scalar>::GenerateNormals(Scalar merging_threshold) {
  std::vector<std::vector<Vector3<Scalar>>> vertex_normals(num_vertices_);
  std::vector<Vector3<Scalar>> weighted_normals(num_vertices_, Vector3<Scalar>::Zero());
  std::vector<Scalar> weights(num_vertices_, 0);
  std::vector<bool> merge(num_vertices_, true);

  for (size_t i = 0; i < num_indices_; i += 3) {
    Vector3<Scalar> v0 = positions_[indices_[i]];
    Vector3<Scalar> v1 = positions_[indices_[i + 1]];
    Vector3<Scalar> v2 = positions_[indices_[i + 2]];
    Vector3<Scalar> normal = (v1 - v0).cross(v2 - v0);
    if (normal.norm() == 0.0) {
      continue;
    }
    normal.normalize();
    Scalar angles[3];
    angles[0] = acos((v1 - v0).normalized().dot((v2 - v0).normalized()));
    angles[1] = acos((v2 - v1).normalized().dot((v0 - v1).normalized()));
    angles[2] = acos((v0 - v2).normalized().dot((v1 - v2).normalized()));
    for (size_t j = 0; j < 3; j++) {
      vertex_normals[indices_[i + j]].push_back(normal);
      weighted_normals[indices_[i + j]] += angles[j] * normal;
      weights[indices_[i + j]] += angles[j];
    }
  }

  std::vector<Vector3<Scalar>> new_positions = positions_;
  std::vector<Vector3<Scalar>> new_normals(num_vertices_);
  std::vector<Vector2<Scalar>> new_tex_coords = tex_coords_;
  std::vector<uint32_t> split_count(num_vertices_, 0);
  std::vector<uint32_t> new_indices;

  for (size_t i = 0; i < num_vertices_; i++) {
    for (size_t j = 0; j < vertex_normals[i].size(); j++) {
      for (size_t k = j + 1; k < vertex_normals[i].size(); k++) {
        if (vertex_normals[i][j].dot(vertex_normals[i][k]) < merging_threshold) {
          merge[i] = false;
          break;
        }
      }
      if (!merge[i]) {
        break;
      }
    }

    if (merge[i]) {
      if (weights[i] > Eps<Scalar>()) {
        new_normals[i] = (weighted_normals[i] / weights[i]).normalized();
      }
    }
  }

  for (size_t i = 0; i < num_indices_; i++) {
    uint32_t vi = indices_[i];
    if (merge[vi]) {
      new_indices.push_back(vi);
    } else {
      if (split_count[vi] == 0) {
        new_normals[vi] = vertex_normals[vi][0];
        new_indices.push_back(vi);
      } else {
        new_positions.push_back(positions_[vi]);
        new_normals.push_back(vertex_normals[vi][split_count[vi]]);
        if (!tex_coords_.empty()) {
          new_tex_coords.push_back(tex_coords_[vi]);
        }
        new_indices.push_back(new_positions.size() - 1);
      }
      split_count[vi]++;
    }
  }

  positions_ = new_positions;
  normals_ = new_normals;
  tex_coords_ = new_tex_coords;
  indices_ = new_indices;

  tangents_.clear();
  signals_.clear();
  num_vertices_ = positions_.size();
  num_indices_ = indices_.size();

  return 0;
}

template <typename Scalar>
int Mesh<Scalar>::InitializeTexCoords(const Vector2<Scalar> &tex_coord) {
  tex_coords_.resize(num_vertices_);
  std::fill(tex_coords_.begin(), tex_coords_.end(), tex_coord);
  return 0;
}

template <typename Scalar>
int Mesh<Scalar>::GenerateTangents() {
  SplitVertices();
  if (tex_coords_.empty()) {
    InitializeTexCoords();
  }
  if (normals_.empty()) {
    GenerateNormals();
  }

  SMikkTSpaceInterface mikkt_space_interface{};

  mikkt_space_interface.m_getNumFaces = [](const SMikkTSpaceContext *context) -> int {
    auto mesh = reinterpret_cast<const Mesh *>(context->m_pUserData);
    return mesh->NumIndices() / 3;
  };

  mikkt_space_interface.m_getNumVerticesOfFace = [](const SMikkTSpaceContext *context, const int face) { return 3; };

  mikkt_space_interface.m_getPosition = [](const SMikkTSpaceContext *context, float position[], const int face,
                                           const int vertex) {
    auto mesh = reinterpret_cast<const Mesh *>(context->m_pUserData);
    auto positions = mesh->positions_[face * 3 + vertex];
    position[0] = positions[0];
    position[1] = positions[1];
    position[2] = positions[2];
  };

  mikkt_space_interface.m_getNormal = [](const SMikkTSpaceContext *context, float normal[], const int face,
                                         const int vertex) {
    auto mesh = reinterpret_cast<const Mesh *>(context->m_pUserData);
    auto normals = mesh->normals_[face * 3 + vertex];
    normal[0] = normals[0];
    normal[1] = normals[1];
    normal[2] = normals[2];
  };

  mikkt_space_interface.m_getTexCoord = [](const SMikkTSpaceContext *context, float tex_coord[], const int face,
                                           const int vertex) {
    auto mesh = reinterpret_cast<const Mesh *>(context->m_pUserData);
    auto tex_coords = mesh->tex_coords_[face * 3 + vertex];
    tex_coord[0] = tex_coords[0];
    tex_coord[1] = tex_coords[1];
  };

  mikkt_space_interface.m_setTSpaceBasic = [](const SMikkTSpaceContext *context, const float tangent[],
                                              const float sign, const int face, const int vertex) {
    auto mesh = reinterpret_cast<Mesh *>(context->m_pUserData);
    auto &tangents = mesh->tangents_[face * 3 + vertex];
    tangents[0] = tangent[0];
    tangents[1] = tangent[1];
    tangents[2] = tangent[2];
    mesh->signals_[face * 3 + vertex] = sign;
  };

  SMikkTSpaceContext context{};
  context.m_pInterface = &mikkt_space_interface;
  context.m_pUserData = reinterpret_cast<void *>(this);

  tangents_.resize(num_vertices_);
  signals_.resize(num_vertices_);

  if (genTangSpaceDefault(&context) == false) {
    return -1;
  }

  return MergeVertices();
}

template <typename Scalar>
Mesh<Scalar> Mesh<Scalar>::Sphere(int precision_lon, int precision_lat) {
  if (precision_lat == -1) {
    precision_lat = precision_lon * 2;
  }
  std::vector<glm::vec<3, Scalar>> sphere_positions;
  std::vector<uint32_t> sphere_indices;
  sphere_positions.push_back(glm::vec3{0.0, 1.0, 0.0});
  for (int j = 1; j < precision_lon; ++j) {
    for (int i = 0; i < precision_lat; ++i) {
      Scalar theta = glm::radians(Scalar(360.0) * (i + (j & 1) * Scalar(0.5)) / precision_lat);
      Scalar phi = glm::radians(Scalar(180.0) * j / precision_lon);
      sphere_positions.push_back(
          glm::vec<3, Scalar>{glm::sin(phi) * glm::cos(theta), glm::cos(phi), glm::sin(phi) * glm::sin(theta)});
    }
  }
  sphere_positions.push_back(glm::vec<3, Scalar>{0.0, -1.0, 0.0});
  for (int j = 0; j < precision_lon; ++j) {
    for (int i = 0; i < precision_lat; ++i) {
      int i_1 = (i - 1 + precision_lat) % precision_lat;
      int i0 = i;
      int i1 = (i + 1) % precision_lat;
      int v00 = (j - 1) * precision_lat + i0 + 1;
      int v01 = (j - 1) * precision_lat + i1 + 1;
      int v10 = j * precision_lat + i0 + 1;
      int v11 = j * precision_lat + i1 + 1;
      if (j & 1) {
        v00 = (j - 1) * precision_lat + i_1 + 1;
        v01 = (j - 1) * precision_lat + i0 + 1;
      }
      if (j == 0) {
        v00 = 0;
        v01 = 0;
      }
      if (j == precision_lon - 1) {
        v10 = sphere_positions.size() - 1;
        v11 = sphere_positions.size() - 1;
      } else {
        sphere_indices.push_back(v11);
        sphere_indices.push_back(v10);
        sphere_indices.push_back(v01);
      }

      if (j) {
        sphere_indices.push_back(v00);
        sphere_indices.push_back(v01);
        sphere_indices.push_back(v10);
      }
    }
  }
  Mesh sphere_mesh(sphere_positions.size(), sphere_indices.size(), sphere_indices.data(),
                   reinterpret_cast<Vector3<Scalar> *>(sphere_positions.data()),
                   reinterpret_cast<Vector3<Scalar> *>(sphere_positions.data()), nullptr, nullptr);
  return sphere_mesh;
}

template class Mesh<float>;
template class Mesh<double>;
}  // namespace grassland
