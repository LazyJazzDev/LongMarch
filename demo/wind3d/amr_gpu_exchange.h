#pragma once
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
  bool Build(const PressurePartition &partition, const std::vector<int> &devices);
  const std::vector<PressureLocalLayout> &Layouts() const;
  // One caller-owned device allocation per rank, sized to its complete layout.
  // Producers must finish before calling. Returns after all copies finish;
  // owned entries are unchanged. CUDA failure may leave imports incomplete.
  bool ExchangeCells(const std::vector<double *> &values);
  bool ExchangeFaces(const std::vector<double *> &values);

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
