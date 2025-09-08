

#include "cub/cub.cuh"
#include "cuda_runtime.h"
#include "curand.h"
#include "long_march.h"

int main() {
  // Define a cube [-1, 1]^3
  std::vector<Eigen::Vector3f> positions = {
      {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1},
  };
  std::vector<uint32_t> indices = {
      0, 1, 2, 0, 2, 3, 1, 5, 6, 1, 6, 2, 5, 4, 7, 5, 7, 6, 4, 0, 3, 4, 3, 7, 3, 2, 6, 3, 6, 7, 0, 4, 5, 0, 5, 1,
  };

  for (size_t i = 0; i < indices.size(); i += 3) {
    std::swap(indices[i], indices[i + 1]);
  }

  std::ofstream outfile("cube.obj");
  for (const auto &pos : positions) {
    outfile << "v " << pos.x() << " " << pos.y() << " " << pos.z() << std::endl;
  }
  for (size_t i = 0; i < indices.size(); i += 3) {
    outfile << "f " << indices[i] + 1 << " " << indices[i + 1] + 1 << " " << indices[i + 2] + 1 << std::endl;
  }
  outfile.close();

  CD::VertexBufferView vbv = {positions.data()};
  CD::MeshSDF mesh_sdf(vbv, positions.size(), indices.data(), indices.size());
  CD::MeshSDFRef mesh_ref = mesh_sdf;
  std::cout << "num_points:" << mesh_ref.num_points << std::endl;
  std::cout << "num_triangles:" << mesh_ref.num_triangles << std::endl;
  std::cout << "num_edges:" << mesh_ref.num_edges << std::endl;

  positions = {
      {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, -1, 0}, {0, 0, -1}, {-1, 0, 0},
  };
  indices = {
      0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 1, 4, 5, 1, 5, 2, 2, 5, 3, 3, 5, 4,
  };
  outfile.open("octahedron.obj");
  for (const auto &pos : positions) {
    outfile << "v " << pos.x() << " " << pos.y() << " " << pos.z() << std::endl;
  }
  for (size_t i = 0; i < indices.size(); i += 3) {
    outfile << "f " << indices[i] + 1 << " " << indices[i + 1] + 1 << " " << indices[i + 2] + 1 << std::endl;
  }
  outfile.close();

  return 0;
}
