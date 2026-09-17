#include "sparkium/backends/sobol_rows.h"

#include <fstream>
#include <stdexcept>

namespace sparkium::backends {

// Mirrors grassland::SobolTableGen (code/grassland/util/sobol.cpp) with the row
// count as a parameter.
std::vector<uint32_t> GenerateSobolRows(uint32_t rows, const std::string &direction_file) {
  if (rows == 0)
    rows = 1;
  if (rows > kSobolMaxRows)
    rows = kSobolMaxRows;
  const uint32_t N = rows;
  const uint32_t D = kSobolDimensions;

  std::ifstream infile(direction_file, std::ios::in);
  if (!infile)
    throw std::runtime_error("cannot open Sobol direction numbers: " + direction_file);
  char buffer[1000];
  infile.getline(buffer, 1000, '\n');

  const uint32_t L = 32;

  // C[i] = index from the right of the first zero bit of i
  std::vector<uint32_t> C(N);
  C[0] = 1;
  for (uint32_t i = 1; i <= N - 1; i++) {
    C[i] = 1;
    uint32_t value = i;
    while (value & 1) {
      value >>= 1;
      C[i]++;
    }
  }

  std::vector<uint32_t> V(L + 1);
  for (uint32_t i = 1; i <= L; i++)
    V[i] = uint32_t(1) << (32 - i);

  std::vector<uint32_t> X(N);
  X[0] = 0;
  for (uint32_t i = 1; i <= N - 1; i++)
    X[i] = X[i - 1] ^ V[C[i - 1]];

  std::vector<uint32_t> sobol_table(size_t(N) * D);
  for (uint32_t i = 0; i < N; i++)
    sobol_table[size_t(i) * D] = X[i];

  for (uint32_t j = 1; j < D; j++) {
    uint32_t d, s, a;
    infile >> d >> s >> a;
    std::vector<uint32_t> m(s + 1);
    for (uint32_t i = 1; i <= s; i++)
      infile >> m[i];

    if (L <= s) {
      for (uint32_t i = 1; i <= L; i++)
        V[i] = m[i] << (32 - i);
    } else {
      for (uint32_t i = 1; i <= s; i++)
        V[i] = m[i] << (32 - i);
      for (uint32_t i = s + 1; i <= L; i++) {
        V[i] = V[i - s] ^ (V[i - s] >> s);
        for (uint32_t k = 1; k <= s - 1; k++)
          V[i] ^= (((a >> (s - 1 - k)) & 1) * V[i - k]);
      }
    }

    X[0] = 0;
    for (uint32_t i = 1; i <= N - 1; i++)
      X[i] = X[i - 1] ^ V[C[i - 1]];
    for (uint32_t i = 0; i < N; i++)
      sobol_table[size_t(i) * D + j] = X[i];
  }

  return sobol_table;
}

}  // namespace sparkium::backends
