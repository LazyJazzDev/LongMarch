#pragma once
// Python bindings are compiled only into the long_march extension module, so the
// C++ libraries and executables do not depend on Python.
#include "grassland/graphics/frame_profile.h"
#include "grassland/graphics/graphics.h"
#include "grassland/graphics/shader.h"
#include "pybind11/chrono.h"
#include "pybind11/eigen.h"
#include "pybind11/functional.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace grassland::graphics::pybind {
void RegisterAccelerationStructure(py::classh<AccelerationStructure> &c);
void RegisterBuffer(py::classh<Buffer> &c);
void RegisterCommandContext(py::classh<CommandContext> &c);
void RegisterCoreSettings(py::classh<Core::Settings> &c);
void RegisterCore(py::classh<Core> &c);
void RegisterGraphics(py::module_ &m);
void RegisterImage(py::classh<Image> &c);
void RegisterImGui(py::module_ &m);
void RegisterProgram(py::classh<Program> &c);
void RegisterComputeProgram(py::classh<ComputeProgram> &c);
void RegisterRayTracingProgram(py::classh<RayTracingProgram> &c);
void RegisterSampler(py::classh<Sampler> &c);
void RegisterShader(py::classh<Shader> &c);
void RegisterTypes(py::module_ &m);
void RegisterWindow(py::classh<Window> &c);
}  // namespace grassland::graphics::pybind
