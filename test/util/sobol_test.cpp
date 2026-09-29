#include "grassland/util/sobol.h"

#include <stdexcept>

#include "gtest/gtest.h"

// A missing direction-number table must fail loudly: continuing would build a
// sampler table from an unreadable stream.
TEST(Sobol, MissingDirectionNumbersThrow) {
  EXPECT_THROW(grassland::SobolTableGen(16, 2, "/nonexistent/new-joe-kuo-7.21201"), std::runtime_error);
}
