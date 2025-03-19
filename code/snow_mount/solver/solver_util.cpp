#include "snow_mount/solver/solver_util.h"

namespace snow_mount::solver {
Directory::Directory(const std::vector<int> &contents) {
  int max_content = 0;
  for (int content : contents) {
    max_content = std::max(max_content, content);
  }
  first.resize(max_content + 1, 0);
  count.resize(max_content + 1, 0);
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
}  // namespace snow_mount::solver
