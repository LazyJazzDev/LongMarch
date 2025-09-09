#include "chang_zheng.h"
#include "gtest/gtest.h"
#include "thrust/device_vector.h"

struct SDFResult {
  float sdf;
  Eigen::Vector3<float> jacobian;
  Eigen::Matrix3<float> hessian;
};

LM_DEVICE_FUNC bool AnyHit(const Eigen::Vector3<float> &query, const SDFResult *result, const CD::AABB &aabb) {
  Eigen::Vector3<float> p;
  for (int i = 0; i < 3; i++) {
    if (query[i] < aabb.lower_bound[i]) {
      p[i] = aabb.lower_bound[i] - query[i];
    } else if (query[i] > aabb.upper_bound[i]) {
      p[i] = query[i] - aabb.upper_bound[i];
    } else {
      p[i] = 0;
    }
  }
  return p.norm() < result->sdf;
}

struct AttachedInfo {
  Eigen::Vector3<float> *vertices;
  uint32_t *indices;
  int num_vertices;
  int num_indices;
};

LM_DEVICE_FUNC bool InstanceHit(const Eigen::Vector3<float> &position,
                                SDFResult *result,
                                int instance_index,
                                const AttachedInfo *attached) {
  float &dist = result->sdf;
  bool hit = false;

  for (int i = 0; i < 3; i++) {
    Eigen::Vector3<float> p = attached->vertices[attached->indices[instance_index * 3 + i]];
    float d = (p - position).norm();
    if (d < dist) {
      if (d == 0.0) {
        printf("point: %d\n", i);
      }
      dist = d;
      hit = true;
      CD::PointSDF<float> point_sdf;
      point_sdf.position = p;
      result->jacobian = point_sdf.Jacobian(position).transpose();
      result->hessian = point_sdf.Hessian(position).m[0];
    }
  }

  while (true) {
    Eigen::Vector3<float> p0 = attached->vertices[attached->indices[instance_index * 3 + 0]];
    Eigen::Vector3<float> p1 = attached->vertices[attached->indices[instance_index * 3 + 1]];
    Eigen::Vector3<float> p2 = attached->vertices[attached->indices[instance_index * 3 + 2]];
    Eigen::Vector3<float> n = (p2 - p0).cross(p1 - p0);

    // p0, p1
    Eigen::Vector3<float> wn;
    wn = (p1 - p0).cross(n);
    if (wn.dot(position) < wn.dot(p0)) {
      break;
    }
    wn = (p2 - p1).cross(n);
    if (wn.dot(position) < wn.dot(p1)) {
      break;
    }
    wn = (p0 - p2).cross(n);
    if (wn.dot(position) < wn.dot(p2)) {
      break;
    }
    if (n.squaredNorm() == 0.0) {
      break;
    }
    n.normalize();
    float d = n.dot(position - p0);
    float signal = 1.0;
    if (d < 0.0) {
      signal = -1.0;
      d = -d;
    }
    if (d < dist) {
      dist = d;
      hit = true;
      CD::PlaneSDF<float> plane_sdf;
      plane_sdf.normal = n;
      plane_sdf.d = -n.dot(p0);
      result->jacobian = plane_sdf.Jacobian(position).transpose() * signal;
      result->hessian = plane_sdf.Hessian(position).m[0] * signal;
    }
    break;
  }

  for (int i = 0; i < 3; i++) {
    int j = (i + 1) % 3;
    Eigen::Vector3<float> p0 = attached->vertices[attached->indices[instance_index * 3 + i]];
    Eigen::Vector3<float> p1 = attached->vertices[attached->indices[instance_index * 3 + j]];

    Eigen::Vector3<float> n = p1 - p0;
    n.normalize();
    float dp = n.dot(position);
    float dp0 = n.dot(p0);
    float dp1 = n.dot(p1);
    if (dp < dp0 || dp > dp1) {
      continue;
    }
    p0 += n * (dp - dp0);
    float d = (p0 - position).norm();
    if (d < dist) {
      if (d == 0.0) {
        printf("edge: %d\n", i);
      }
      dist = d;
      hit = true;
      CD::LineSDF<float> line_sdf;
      line_sdf.direction = n;
      line_sdf.origin = p0;
      result->jacobian = line_sdf.Jacobian(position).transpose();
      result->hessian = line_sdf.Hessian(position).m[0];
    }
  }

  return hit;
}

void BatchQuery(const CD::BVHRef &bvh,
                const Eigen::Vector3<float> *queries,
                SDFResult *result,
                const AttachedInfo &attached_info,
                int num_tasks) {
  for (int i = 0; i < num_tasks; i++) {
    bvh.Traversal(queries[i], result + i, &attached_info, AnyHit, InstanceHit);
  }
}

__global__ void BatchQueryDevKernel(const CD::BVHRef bvh,
                                    const Eigen::Vector3<float> *queries,
                                    SDFResult *result,
                                    const AttachedInfo attached_info,
                                    int num_tasks) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_tasks) {
    return;
  }
  SDFResult result_dev = result[tid];
  auto query = queries[tid];
  bvh.Traversal(query, &result_dev, &attached_info, AnyHit, InstanceHit);
  result[tid] = result_dev;
}

void BatchQueryDev(const CD::BVHHost &bvh,
                   const Eigen::Vector3<float> *queries,
                   SDFResult *result,
                   const AttachedInfo &attached_info,
                   int num_tasks) {
  thrust::device_vector<Eigen::Vector3<float>> queries_dev(queries, queries + num_tasks);
  thrust::device_vector<SDFResult> result_dev(result, result + num_tasks);
  thrust::device_vector<Eigen::Vector3<float>> vertices_dev(attached_info.vertices,
                                                            attached_info.vertices + attached_info.num_vertices);
  thrust::device_vector<uint32_t> indices_dev(attached_info.indices, attached_info.indices + attached_info.num_indices);
  AttachedInfo attached_dev;
  attached_dev.vertices = thrust::raw_pointer_cast(vertices_dev.data());
  attached_dev.indices = thrust::raw_pointer_cast(indices_dev.data());
  attached_dev.num_vertices = attached_info.num_vertices;
  attached_dev.num_indices = attached_info.num_indices;

  CD::BVHCuda bvh_dev(bvh);

  CD::DeviceClock clock;
  BatchQueryDevKernel<<<(num_tasks + 255) / 256, 256>>>(bvh_dev, thrust::raw_pointer_cast(queries_dev.data()),
                                                        thrust::raw_pointer_cast(result_dev.data()), attached_dev,
                                                        num_tasks);
  clock.Record("GPU");
  clock.Finish();

  thrust::copy(result_dev.begin(), result_dev.end(), result);
}

TEST(BVH, SphereSDF) {
  std::vector<Eigen::Vector3<float>> vertices;
  std::vector<uint32_t> indices;
  // vertices = {{-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {-1.0, 1.0, -1.0}, {1.0, 1.0, -1.0},
  //             {-1.0, -1.0, 1.0},  {1.0, -1.0, 1.0},  {-1.0, 1.0, 1.0},  {1.0, 1.0, 1.0}};
  // indices = {0, 1, 2, 1, 3, 2, 4, 6, 5, 5, 6, 7, 0, 2, 4, 4, 2, 6,
  //              1, 5, 3, 5, 7, 3, 0, 4, 1, 4, 5, 1, 2, 3, 6, 6, 3, 7};

  int precision = 50;
  float inv_precision = 1.0 / precision;
  const float pi = 3.14159265358979323846264338327950288419716939937510f;
  for (int i = 0; i <= precision; i++) {
    float theta = pi * i * inv_precision;
    for (int j = 0; j < precision; j++) {
      float phi = 2 * pi * j * inv_precision;
      vertices.emplace_back(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta));
      if (i < precision) {
        int i1 = i + 1;
        int j1 = (j + 1) % precision;
        indices.push_back(i * precision + j);
        indices.push_back(i1 * precision + j);
        indices.push_back(i * precision + j1);
        indices.push_back(i * precision + j1);
        indices.push_back(i1 * precision + j);
        indices.push_back(i1 * precision + j1);
      }
    }
  }

  std::vector<CD::AABB> aabbs;
  std::vector<int> instance_indices;
  for (size_t i = 0; i < indices.size(); i += 3) {
    CD::AABB aabb;
    aabb.lower_bound = vertices[indices[i]];
    aabb.upper_bound = vertices[indices[i]];
    for (size_t j = 1; j < 3; ++j) {
      aabb.lower_bound = aabb.lower_bound.cwiseMin(vertices[indices[i + j]]);
      aabb.upper_bound = aabb.upper_bound.cwiseMax(vertices[indices[i + j]]);
    }
    aabbs.push_back(aabb);
    instance_indices.push_back(i / 3);
  }

  CD::BVHHost bvh{aabbs.data(), instance_indices.data(), static_cast<int>(aabbs.size())};

  AttachedInfo attached;
  attached.vertices = vertices.data();
  attached.indices = indices.data();
  attached.num_vertices = vertices.size();
  attached.num_indices = indices.size();
  std::vector<Eigen::Vector3<float>> queries;

  int num_queries = 1048576;

  for (int i = 0; i < num_queries; i++) {
    Eigen::Vector3<float> vec = Eigen::Vector3<float>::Random() * 10;
    queries.push_back(vec);
  }
  std::vector<SDFResult> results(queries.size());
  std::vector<SDFResult> results_dev;
  for (auto &result : results) {
    result.sdf = 1e10;
  }
  results_dev = results;
  auto tp = std::chrono::system_clock::now();
  BatchQuery(bvh, queries.data(), results.data(), attached, queries.size());
  auto tp1 = std::chrono::system_clock::now();
  BatchQueryDev(bvh, queries.data(), results_dev.data(), attached, queries.size());

  std::cout << "CPU: " << std::chrono::duration_cast<std::chrono::milliseconds>(tp1 - tp).count() << "ms" << std::endl;

  for (int i = 0; i < queries.size(); i++) {
    EXPECT_NEAR(results[i].sdf, results_dev[i].sdf, 1e-4);
  }
}
