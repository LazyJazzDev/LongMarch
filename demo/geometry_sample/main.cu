#include "cmath"
#include "grassland/physics/physics.h"
#include "iostream"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

using namespace CD;

using real = float;

int main() {
  Eigen::Vector3<real> x0(0, 1, 0);
  Eigen::Vector3<real> x1(0, 0, 1);
  Eigen::Vector3<real> x2(0, 0, -1);
  Eigen::Vector3<real> x3(-1, 0, 0);
  Eigen::Matrix<real, 3, 4> X;
  X << x0, x1, x2, x3;

  DihedralAngle<real> dihedral_angle;
  std::cout << dihedral_angle(X).value() << std::endl;
  std::cout << dihedral_angle.Hessian(X) << std::endl;
}
