#include "grassland/util/sobol.h"

#include "fstream"
#include "grassland/util/log.h"

namespace grassland {
std::vector<uint32_t> SobolTableGen(unsigned int N, unsigned int D, const std::string &dir_file) {
  std::ifstream infile(dir_file, std::ios::in);
  if (!infile) {
    LogError("Input file containing direction numbers cannot be found!\n");
  }
  char buffer[1000];
  infile.getline(buffer, 1000, '\n');

  // L = max number of bits needed
  unsigned L = 32;

  // C[i] = index from the right of the first zero bit of i
  std::vector<uint32_t> C(N);
  C[0] = 1;
  for (unsigned i = 1; i <= N - 1; i++) {
    C[i] = 1;
    unsigned value = i;
    while (value & 1) {
      value >>= 1;
      C[i]++;
    }
  }

  // ----- Compute the first dimension -----

  // Compute direction numbers V[1] to V[L], scaled by pow(2,32)
  std::vector<uint32_t> V(L + 1);
  for (unsigned i = 1; i <= L; i++)
    V[i] = 1 << (32 - i);  // all m's = 1

  // Evalulate X[0] to X[N-1], scaled by pow(2,32)
  std::vector<uint32_t> X(N);
  X[0] = 0;
  for (unsigned i = 1; i <= N - 1; i++) {
    X[i] = X[i - 1] ^ V[C[i - 1]];
  }

  std::vector<uint32_t> sobol_table(N * D);

  for (int i = 0; i < N; i++) {
    sobol_table[i * D] = X[i];
  }

  // ----- Compute the remaining dimensions -----
  for (unsigned j = 1; j < D; j++) {
    // Read in parameters from file
    unsigned d, s;
    unsigned a;
    infile >> d >> s >> a;
    std::vector<uint32_t> m(s + 1);
    for (unsigned i = 1; i <= s; i++)
      infile >> m[i];

    // Compute direction numbers V[1] to V[L], scaled by pow(2,32)
    if (L <= s) {
      for (unsigned i = 1; i <= L; i++)
        V[i] = m[i] << (32 - i);
    } else {
      for (unsigned i = 1; i <= s; i++)
        V[i] = m[i] << (32 - i);
      for (unsigned i = s + 1; i <= L; i++) {
        V[i] = V[i - s] ^ (V[i - s] >> s);
        for (unsigned k = 1; k <= s - 1; k++)
          V[i] ^= (((a >> (s - 1 - k)) & 1) * V[i - k]);
      }
    }

    // Evalulate X[0] to X[N-1], scaled by pow(2,32)
    X[0] = 0;
    for (unsigned i = 1; i <= N - 1; i++) {
      X[i] = X[i - 1] ^ V[C[i - 1]];
    }
    for (int i = 0; i < N; i++) {
      sobol_table[i * D + j] = X[i];
    }
  }

  return sobol_table;
}

}  // namespace grassland
