#pragma once
// Partial Sobol generator used by the offline backends.
//
// The online backends upload the full 65536 x 1024 table produced by
// grassland::SobolTableGen and index it as `sobol_table[samp * 1024 + dim]`.
// The offline backends only ever read the rows they actually consume
// (`sample_ind < accumulated_samples + samples_per_dispatch`), so they generate
// exactly those rows with the same recurrence and the same direction-number
// file. GenerateSobolRows is verified against grassland::SobolTableGen in
// test/sparkium/offline_backend_test.cpp.

#include <cstdint>
#include <string>
#include <vector>

namespace sparkium::backends {

// Number of dimensions of the online table, i.e. the stride of one sample row.
inline constexpr uint32_t kSobolDimensions = 1024;
// Number of rows of the online table; RandomUint falls back to WangHash beyond it.
inline constexpr uint32_t kSobolMaxRows = 65536;

// Returns `rows * kSobolDimensions` entries; row `s`, dimension `d` is at
// `s * kSobolDimensions + d`, which is what random.hlsli loads.
std::vector<uint32_t> GenerateSobolRows(uint32_t rows, const std::string &direction_file);

}  // namespace sparkium::backends
