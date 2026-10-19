#include <gtest/gtest.h>

#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
namespace {

TEST(RhiEnumeration, NativeFailureReturnsFalseAndDiscardsPartialResults) {
  std::vector<uint32_t> values{123};
  EXPECT_FALSE(EnumerateList([](uint32_t *, uint32_t *) { return VK_ERROR_INITIALIZATION_FAILED; }, values));
  EXPECT_TRUE(values.empty());

  EXPECT_FALSE(EnumerateList(
      [](uint32_t *count, uint32_t *data) {
        if (!data) {
          *count = 2;
          return VK_SUCCESS;
        }
        data[0] = 42;
        return VK_ERROR_DEVICE_LOST;
      },
      values));
  EXPECT_TRUE(values.empty());
}

TEST(RhiEnumeration, EmptySuccessIsDistinctFromFailure) {
  std::vector<uint32_t> values{123};
  EXPECT_TRUE(EnumerateList(
      [](uint32_t *count, uint32_t *) {
        *count = 0;
        return VK_SUCCESS;
      },
      values));
  EXPECT_TRUE(values.empty());
}

TEST(RhiEnumeration, IncompleteResultsAreRetriedAndLaterFailureIsNotSuccess) {
  std::vector<uint32_t> values;
  int calls = 0;
  EXPECT_TRUE(EnumerateList(
      [&](uint32_t *count, uint32_t *data) {
        ++calls;
        if (!data) {
          *count = calls == 1 ? 1 : 2;
          return VK_SUCCESS;
        }
        data[0] = 7;
        if (calls == 2)
          return VK_INCOMPLETE;
        data[1] = 9;
        return VK_SUCCESS;
      },
      values));
  EXPECT_EQ(values, (std::vector<uint32_t>{7, 9}));
  EXPECT_EQ(calls, 4);

  calls = 0;
  EXPECT_FALSE(EnumerateList(
      [&](uint32_t *count, uint32_t *data) {
        ++calls;
        if (calls == 3)
          return VK_ERROR_OUT_OF_HOST_MEMORY;
        *count = 1;
        if (!data)
          return VK_SUCCESS;
        data[0] = 7;
        return VK_INCOMPLETE;
      },
      values));
  EXPECT_TRUE(values.empty());
}

}  // namespace
}  // namespace grassland::rhi::backend::vulkan
