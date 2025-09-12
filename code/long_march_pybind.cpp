#include "long_march.h"

PYBIND11_MODULE(long_march, m) {
  long_march::PybindModuleRegistration(m);
}
