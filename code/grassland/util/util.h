#pragma once

#define NOMINMAX

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "grassland/util/double_ptr.h"
#include "grassland/util/event_manager.h"
#include "grassland/util/log.h"
#include "grassland/util/string_convert.h"
#include "grassland/util/vendor_id.h"

namespace grassland {
#if defined(__CUDACC__)
#define LM_DEVICE_FUNC __device__ __host__
#else
#define LM_DEVICE_FUNC
#endif
}  // namespace grassland
