#include "grassland/util/sha256.h"

#include <gtest/gtest.h>

TEST(SHA256, KnownDigest) {
  grassland::SHA256 digest;
  digest.Update("abc");
  EXPECT_EQ(digest.Finish(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}
