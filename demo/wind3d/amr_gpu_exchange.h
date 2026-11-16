#pragma once
#include <cuda_runtime_api.h>

#include <memory>
#include <string>

#include "amr_partition.h"

namespace wind3d::amr {
struct PressureLocalLayout {
  size_t owned_cells{}, owned_faces{};
  // Owned entries first, followed by imports. Values use these local indices.
  std::vector<int> cells, faces;
};

// Persistent peer-copy buffers for a single-node pressure partition. Transport
// halos and a distributed solver are separate consumers, not implemented here.
class GpuPressureExchange {
 public:
  GpuPressureExchange();
  ~GpuPressureExchange();
  GpuPressureExchange(const GpuPressureExchange &) = delete;
  GpuPressureExchange &operator=(const GpuPressureExchange &) = delete;
  // Enables required peer access for the CUDA contexts' lifetime; never disables
  // access shared with other solver instances.
  // Optional global-face mask applies only to EnqueueFaces; empty means all.
  // Full layouts and synchronous exchanges are unaffected. Failure preserves
  // the previous plan. Nonempty masks must have one entry per global face.
  bool Build(const PressurePartition &partition,
             const std::vector<int> &devices,
             const std::vector<unsigned char> &queued_face_mask = {});
  const std::vector<PressureLocalLayout> &Layouts() const;
  // One caller-owned device allocation per rank, sized to its complete layout.
  // Producers must finish before calling. Returns after all copies finish;
  // owned entries are unchanged. CUDA failure may leave imports incomplete.
  bool ExchangeCells(const std::vector<double *> &values);
  bool ExchangeFaces(const std::vector<double *> &values);
  // Queue capturable peer-store face imports between producer/consumer work on
  // one stream per rank, restricted to Build's mask. Unselected imports stay unchanged.
  // Streams belong to Build's devices. Repeated enqueues on these same streams
  // are ordered by events. Caller must drain all streams, including on failure,
  // before destroying/rebuilding this object or using its synchronous methods.
  bool EnqueueFaces(const std::vector<double *> &values, const std::vector<cudaStream_t> &streams);

  const std::string &Error() const {
    return error_;
  }

 private:
  bool Exchange(const std::vector<double *> &values, bool faces);
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string error_;
};
}  // namespace wind3d::amr
