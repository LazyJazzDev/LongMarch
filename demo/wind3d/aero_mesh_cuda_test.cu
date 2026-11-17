#include <cuda_runtime.h>

#include <iostream>

#include "aero_mesh.h"

using namespace wind3d::aero;

namespace {
__global__ void QueryMesh(MeshView view, const Vector *points, double *distance, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    distance[i] = view.Distance(points[i]);
}

struct Buffers {
  MeshTriangle *triangles{};
  MeshNode *nodes{};
  Vector *points{};
  double *distance{};

  ~Buffers() {
    cudaFree(triangles);
    cudaFree(nodes);
    cudaFree(points);
    cudaFree(distance);
  }
};
}  // namespace

bool CheckCudaMesh(const MeshGeometry &mesh, const std::vector<Vector> &points, const std::vector<double> &expected) {
  auto check = [](cudaError_t status) {
    if (status == cudaSuccess)
      return true;
    std::cerr << cudaGetErrorString(status) << '\n';
    return false;
  };
  Buffers b;
  size_t triangles = mesh.Triangles().size() * sizeof(MeshTriangle), nodes = mesh.Nodes().size() * sizeof(MeshNode);
  if (!check(cudaMalloc(&b.triangles, triangles)) || !check(cudaMalloc(&b.nodes, nodes)) ||
      !check(cudaMalloc(&b.points, points.size() * sizeof(Vector))) ||
      !check(cudaMalloc(&b.distance, points.size() * sizeof(double))))
    return false;
  if (!check(cudaMemcpy(b.triangles, mesh.Triangles().data(), triangles, cudaMemcpyHostToDevice)) ||
      !check(cudaMemcpy(b.nodes, mesh.Nodes().data(), nodes, cudaMemcpyHostToDevice)) ||
      !check(cudaMemcpy(b.points, points.data(), points.size() * sizeof(Vector), cudaMemcpyHostToDevice)))
    return false;
  QueryMesh<<<(points.size() + 255) / 256, 256>>>({b.triangles, b.nodes, int(mesh.Nodes().size())}, b.points,
                                                  b.distance, int(points.size()));
  std::vector<double> result(points.size());
  if (!check(cudaGetLastError()) ||
      !check(cudaMemcpy(result.data(), b.distance, result.size() * sizeof(double), cudaMemcpyDeviceToHost)))
    return false;
  for (size_t i = 0; i < result.size(); ++i)
    if (!std::isfinite(result[i]) || std::abs(result[i] - expected[i]) > 1e-10 ||
        (expected[i] == 0 && result[i] != 0)) {
      std::cerr << "CUDA signed distance mismatch at " << i;
      return false;
    }
  std::cout << "PASS CUDA mesh queries: " << points.size() << '\n';
  return true;
}
