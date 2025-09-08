
#if defined(__CUDACC__)
#include <thrust/host_vector.h>
#endif

#include "gtest/gtest.h"
#include "long_march.h"

TEST(Math, MeshSDFCorrectness) {
  std::vector<Eigen::Vector3f> positions = {
      {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1},
  };
  std::vector<uint32_t> indices = {
      0, 1, 2, 0, 2, 3, 1, 5, 6, 1, 6, 2, 5, 4, 7, 5, 7, 6, 4, 0, 3, 4, 3, 7, 3, 2, 6, 3, 6, 7, 0, 4, 5, 0, 5, 1,
  };

  for (size_t i = 0; i < indices.size(); i += 3) {
    std::swap(indices[i], indices[i + 1]);
  }

  CD::VertexBufferView vbv = {positions.data()};
  CD::MeshSDF mesh_sdf(vbv, positions.size(), indices.data(), indices.size());
  CD::MeshSDFRef mesh_ref = mesh_sdf;

  for (int task = 0; task < 10000; task++) {
    Eigen::Vector3<float> t = Eigen::Vector3<float>::Random();
    float s = Eigen::Matrix<float, 1, 1>::Random().value() + 1.1;
    Eigen::Matrix<float, 3, 3> R = Eigen::Matrix<float, 3, 3>::Identity() * s;
    CD::CubeSDF<float> cube_sdf;
    cube_sdf.center = t;
    cube_sdf.size = s;
    for (int test = 0; test < 100; test++) {
      Eigen::Vector3<float> p;
      Eigen::Vector3<float> dp;
      do {
        p = Eigen::Vector3<float>::Random() * 4.0 * s + t;
        dp = (p - t).cwiseAbs();
      } while (fabs(dp[0] - dp[1]) < CD::Eps<float>() || fabs(dp[1] - dp[2]) < CD::Eps<float>() ||
               fabs(dp[0] - dp[2]) < CD::Eps<float>());

      Eigen::Vector3<float> mesh_jacobian;
      Eigen::Matrix<float, 3, 3> mesh_hessian;
      float mesh_t;
      mesh_ref.SDF(p, R, t, &mesh_t, &mesh_jacobian, &mesh_hessian);
      Eigen::Vector3<float> cube_jacobian;
      Eigen::Matrix<float, 3, 3> cube_hessian;
      float cube_t;
      cube_t = cube_sdf(p).value();
      cube_jacobian = cube_sdf.Jacobian(p);
      cube_hessian = cube_sdf.Hessian(p).m[0];
      bool error = false;
      EXPECT_NEAR(mesh_t, cube_t, 1e-4f), error = true;
      EXPECT_NEAR((mesh_jacobian - cube_jacobian).norm() / fmax(mesh_jacobian.norm() * cube_jacobian.norm(), 1), 0,
                  1e-4f),
          error = true;
      EXPECT_NEAR((mesh_hessian - cube_hessian).norm() / fmax(mesh_hessian.norm() * cube_hessian.norm(), 1), 0, 1e-4f),
          error = true;
      if (error) {
        std::cout << "test: " << test << std::endl,
            std::cout << "mesh_t: " << mesh_t << " "
                      << "cube_t: " << cube_t << std::endl,
            std::cout << "mesh_jacobian: " << mesh_jacobian.transpose() << std::endl,
            std::cout << "cube_jacobian: " << cube_jacobian.transpose() << std::endl,
            std::cout << "mesh_hessian: \n"
                      << mesh_hessian << std::endl
                      << "cube_hessian: \n"
                      << cube_hessian << std::endl,
            std::cout << "p: " << p.transpose() << std::endl,
            std::cout << "s: " << s << " "
                      << "t: " << t.transpose() << std::endl,
            std::cout << "p - t: " << (p - t).cwiseAbs().transpose() << std::endl,
            std::cout << "abs(p - t) - s: " << ((p - t).cwiseAbs() - (Eigen::Vector3<float>::Ones() * s)).transpose()
                      << std::endl;
      }
    }
  }
}

#if defined(__CUDACC__)

__global__ void MeshSDFDeviceKernel(CD::MeshSDFRef mesh_sdf,
                                    const Eigen::Vector4<float> *mesh_refs,
                                    const Eigen::Vector3<float> *task_positions,
                                    float *results_t,
                                    Eigen::Vector3<float> *results_jacobian,
                                    Eigen::Matrix3<float> *results_hessian) {
  int task = blockIdx.x;
  int test = threadIdx.x;
  int idx = task * blockDim.x + test;

  Eigen::Vector3<float> t = mesh_refs[task].head<3>();
  float s = mesh_refs[task][3];
  Eigen::Matrix<float, 3, 3> R = Eigen::Matrix<float, 3, 3>::Identity() * s;
  Eigen::Vector3<float> position = task_positions[idx];
  mesh_sdf.SDF(position, R, t, &results_t[idx], &results_jacobian[idx], &results_hessian[idx]);
}

TEST(Math, MeshSDFDevice) {
  std::vector<Eigen::Vector3f> positions = {
      {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1},
  };
  std::vector<uint32_t> indices = {
      0, 1, 2, 0, 2, 3, 1, 5, 6, 1, 6, 2, 5, 4, 7, 5, 7, 6, 4, 0, 3, 4, 3, 7, 3, 2, 6, 3, 6, 7, 0, 4, 5, 0, 5, 1,
  };

  for (size_t i = 0; i < indices.size(); i += 3) {
    std::swap(indices[i], indices[i + 1]);
  }

  CD::VertexBufferView vbv = {positions.data()};
  CD::MeshSDF mesh_sdf(vbv, positions.size(), indices.data(), indices.size());
  CD::MeshSDFDevice mesh_sdf_device = mesh_sdf;
  CD::MeshSDFRef mesh_ref = mesh_sdf;
  CD::MeshSDFRef mesh_ref_device = mesh_sdf_device;

  std::vector<Eigen::Vector4<float>> mesh_refs;
  thrust::device_vector<Eigen::Vector4<float>> mesh_refs_device;
  std::vector<Eigen::Vector3<float>> task_positions;
  thrust::device_vector<Eigen::Vector3<float>> task_positions_device;
  thrust::device_vector<float> results_t;
  thrust::device_vector<Eigen::Vector3<float>> results_jacobian;
  thrust::device_vector<Eigen::Matrix<float, 3, 3>> results_hessian;
  thrust::host_vector<float> results_t_host;
  thrust::host_vector<Eigen::Vector3<float>> results_jacobian_host;
  thrust::host_vector<Eigen::Matrix<float, 3, 3>> results_hessian_host;

  const int num_task = 100000;
  const int num_test = 100;
  for (int task = 0; task < num_task; task++) {
    Eigen::Vector3<float> t = Eigen::Vector3<float>::Random();
    float s = Eigen::Matrix<float, 1, 1>::Random().value() + 1.1;

    mesh_refs.push_back({t[0], t[1], t[2], s});

    for (int test = 0; test < num_test; test++) {
      Eigen::Vector3<float> p;
      Eigen::Vector3<float> dp;
      do {
        p = Eigen::Vector3<float>::Random() * 4.0 * s + t;
        dp = (p - t).cwiseAbs();
      } while (fabs(dp[0] - dp[1]) < CD::Eps<float>() || fabs(dp[1] - dp[2]) < CD::Eps<float>() ||
               fabs(dp[0] - dp[2]) < CD::Eps<float>());
      task_positions.push_back(p);
    }
  }

  mesh_refs_device = mesh_refs;
  task_positions_device = task_positions;
  results_t.resize(task_positions.size());
  results_jacobian.resize(task_positions.size());
  results_hessian.resize(task_positions.size());

  MeshSDFDeviceKernel<<<num_task, num_test>>>(mesh_ref_device, mesh_refs_device.data().get(),
                                              task_positions_device.data().get(), results_t.data().get(),
                                              results_jacobian.data().get(), results_hessian.data().get());

  results_t_host = results_t;
  results_jacobian_host = results_jacobian;
  results_hessian_host = results_hessian;

  for (int task = 0; task < num_task; task++) {
    Eigen::Vector3<float> t = mesh_refs[task].head<3>();
    float s = mesh_refs[task][3];

    Eigen::Matrix<float, 3, 3> R = Eigen::Matrix<float, 3, 3>::Identity() * s;

    for (int test = 0; test < num_test; test++) {
      int idx = task * num_test + test;
      Eigen::Vector3<float> p = task_positions[idx];
      float mesh_t;
      Eigen::Vector3<float> mesh_jacobian;
      Eigen::Matrix3<float> mesh_hessian;

      float mesh_t_dev;
      Eigen::Vector3<float> mesh_jacobian_dev;
      Eigen::Matrix3<float> mesh_hessian_dev;

      mesh_ref.SDF(p, R, t, &mesh_t, &mesh_jacobian, &mesh_hessian);
      mesh_t_dev = results_t_host[idx];
      mesh_jacobian_dev = results_jacobian_host[idx];
      mesh_hessian_dev = results_hessian_host[idx];

      bool error = false;
      EXPECT_NEAR(mesh_t, mesh_t_dev, 1e-4f), error = true;
      EXPECT_NEAR((mesh_jacobian - mesh_jacobian_dev).norm() / fmax(mesh_jacobian.norm() * mesh_jacobian_dev.norm(), 1),
                  0, 1e-4f),
          error = true;
      EXPECT_NEAR((mesh_hessian - mesh_hessian_dev).norm() / fmax(mesh_hessian.norm() * mesh_hessian_dev.norm(), 1), 0,
                  1e-4f),
          error = true;
      if (error) {
        std::cout << "p: " << p.transpose() << std::endl;
        std::cout << "t: " << t.transpose() << std::endl << "s: " << s << std::endl;
        std::cout << "p - t: " << (p - t).transpose() << std::endl;
        std::cout << "sdf: " << mesh_t << " "
                  << "sdf_dev: " << mesh_t_dev << std::endl;
        std::cout << "jacobian: " << mesh_jacobian.transpose() << std::endl
                  << "jacobian_dev: " << mesh_jacobian_dev.transpose() << std::endl;
        std::cout << "hessian: " << std::endl
                  << mesh_hessian << std::endl
                  << "hessian_dev: " << std::endl
                  << mesh_hessian_dev << std::endl;
      }
    }
  }
}
#endif
