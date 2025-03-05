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

  grassland::VertexBufferView vbv = {positions.data()};
  grassland::MeshSDF mesh_sdf(vbv, positions.size(), indices.data(), indices.size());
  grassland::MeshSDFRef mesh_ref = mesh_sdf;

  for (int task = 0; task < 10000; task++) {
    Eigen::Vector3<float> t = Eigen::Vector3<float>::Random();
    float s = Eigen::Matrix<float, 1, 1>::Random().value() + 1.1;
    // s = 1.65211;
    // t << -0.561205, 0.785272, 0.777642;
    mesh_ref.translation = t;
    mesh_ref.rotation = Eigen::Matrix<float, 3, 3>::Identity() * s;
    grassland::CubeSDF<float> cube_sdf;
    cube_sdf.center = t;
    cube_sdf.size = s;
    for (int test = 0; test < 100; test++) {
      Eigen::Vector3<float> p;
      Eigen::Vector3<float> dp;
      do {
        Eigen::Vector3<float>::Random() * 4.0 * s + t;
        dp = p - t;
      } while (fabs(dp[0] - dp[1]) < grassland::Eps<float>() || fabs(dp[1] - dp[2]) < grassland::Eps<float>() ||
               fabs(dp[0] - dp[2]) < grassland::Eps<float>());
      // p << -2.09659, 5.56972, -0.757746;
      Eigen::Vector3<float> mesh_jacobian;
      Eigen::Matrix<float, 3, 3> mesh_hessian;
      float mesh_t;
      mesh_ref.SDF(p, &mesh_t, &mesh_jacobian, &mesh_hessian);
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
