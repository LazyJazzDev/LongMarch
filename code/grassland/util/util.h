#pragma once

#define NOMINMAX

#include "grassland/util/double_ptr.h"
#include "grassland/util/event_manager.h"
#include "grassland/util/log.h"
#include "grassland/util/string_convert.h"

namespace grassland {
#if defined(__CUDACC__)
#define LM_DEVICE_FUNC __device__ __host__
#else
#define LM_DEVICE_FUNC
#endif
}  // namespace grassland
