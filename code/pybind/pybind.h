#pragma once
// Python bindings are compiled only into the long_march extension module, so the
// C++ libraries and executables do not depend on Python.
#include "pybind11/chrono.h"
#include "pybind11/eigen.h"
#include "pybind11/functional.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
