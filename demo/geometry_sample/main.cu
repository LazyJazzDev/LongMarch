#include "cmath"
#include "iostream"
#include "long_march.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

using namespace long_march;

using real = float;

int main() {
  geometry::Vector3<real> v1(1, 0, 0);
  geometry::Vector3<real> v2(0, 1, 0);
  geometry::Vector3<real> v3(0, 0, 1);
  geometry::Vector3<real> v = (v1 + v2 + v3) / 3;
  std::cout << geometry::FacePointIntersection(v1, v2, v3, v) << std::endl;
}
