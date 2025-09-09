#include "xue_shan/solver/solver_util.h"

namespace XS::solver {

LM_DEVICE_FUNC RigidObjectState RigidObjectState::NextState(float dt) const {
  RigidObjectState new_state = *this;
  new_state.t += v * dt;
  new_state.R = Eigen::AngleAxis<float>(omega.norm() * dt, omega.normalized()).toRotationMatrix() * R;
  Matrix3<float> I = R * inertia * R.transpose();
  Vector3<float> angular_momentum = I * omega;
  I = new_state.R * inertia * new_state.R.transpose();
  new_state.omega = I.inverse() * angular_momentum;
  return new_state;
}

Directory::Directory(const std::vector<int> &contents, int num_bucket) {
  first.resize(num_bucket, 0);
  count.resize(num_bucket, 0);
  for (int content : contents) {
    count[content]++;
  }
  for (int i = 0, accum = 0; i < first.size(); i++) {
    first[i] = accum;
    accum += count[i];
  }
  positions.resize(contents.size());
  for (int i = 0; i < contents.size(); i++) {
    positions[first[contents[i]]++] = i;
  }
  for (int i = 0, accum = 0; i < first.size(); i++) {
    first[i] = accum;
    accum += count[i];
  }
}

Directory::operator DirectoryRef() const {
  DirectoryRef directory_ref;
  directory_ref.first = first.data();
  directory_ref.count = count.data();
  directory_ref.positions = positions.data();
  return directory_ref;
}

#if defined(__CUDACC__)

DirectoryDevice::DirectoryDevice(const Directory &directory) {
  first = directory.first;
  count = directory.count;
  positions = directory.positions;
}

DirectoryDevice::operator DirectoryRef() const {
  DirectoryRef directory_ref;
  directory_ref.first = thrust::raw_pointer_cast(first.data());
  directory_ref.count = thrust::raw_pointer_cast(count.data());
  directory_ref.positions = thrust::raw_pointer_cast(positions.data());
  return directory_ref;
}

#endif
}  // namespace XS::solver
