#pragma once
#include <stdexcept>
#include <string>

namespace sparkium::backend::cpu {

[[noreturn]] inline void CpuUnsupported() {
  throw std::runtime_error(
      "operation unavailable on this headless compute backend; use a graphics backend for rasterization or presentation");
}

}  // namespace sparkium::backend::cpu
