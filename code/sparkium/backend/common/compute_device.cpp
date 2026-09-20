#include "sparkium/backend/common/compute_device.h"

#include <iostream>

#include "sparkium/backend/common/native_buffer.h"
#include "sparkium/backend/common/native_command_context.h"
#include "sparkium/backend/common/native_image.h"
#include "sparkium/backend/common/native_program.h"
#include "sparkium/backend/common/native_sampler.h"
#include "sparkium/backend/common/native_util.h"

namespace sparkium::backend {
int ComputeDevice::CreateBuffer(size_t s, BufferType t, double_ptr<Buffer> p) {
  p.construct<NativeBuffer>(UsesCUDA(), s, t);
  return 0;
}

int ComputeDevice::CreateImage(int w, int h, ImageFormat f, double_ptr<Image> p) {
  p.construct<NativeImage>(UsesCUDA(), w, h, f);
  return 0;
}

int ComputeDevice::CreateSampler(const SamplerInfo &i, double_ptr<Sampler> p) {
  p.construct<NativeSampler>(i);
  return 0;
}

int ComputeDevice::CreateShader(const std::string &s,
                                const std::string &e,
                                const std::string &t,
                                double_ptr<Shader> p) {
  VirtualFileSystem v;
  v.WriteFile("input.hlsl", s);
  return CreateShader(v, "input.hlsl", e, t, {}, p);
}

int ComputeDevice::CreateShader(const VirtualFileSystem &v,
                                const std::string &s,
                                const std::string &e,
                                const std::string &t,
                                double_ptr<Shader> p) {
  return CreateShader(v, s, e, t, {}, p);
}

int ComputeDevice::CreateShader(const VirtualFileSystem &v,
                                const std::string &s,
                                const std::string &e,
                                const std::string &t,
                                const std::vector<std::string> &a,
                                double_ptr<Shader> p) {
  if (t.rfind("cs_", 0) != 0)
    NativeUnsupported();
  OptixDevice *optix = Optix();
  p.construct<NativeShader>(UsesCUDA(), v, s, e, a, optix);
  return 0;
}

int ComputeDevice::CreateComputeProgram(Shader *s, double_ptr<ComputeProgram> p) {
  auto *n = dynamic_cast<NativeShader *>(s);
  if (!n)
    throw std::runtime_error("foreign native shader");
  p.construct<NativeProgram>(n);
  return 0;
}

int ComputeDevice::CreateCommandContext(double_ptr<CommandContext> p) {
  p.construct<NativeCommandContext>(this);
  return 0;
}

int ComputeDevice::SubmitCommandContext(CommandContext *p) {
  auto *n = dynamic_cast<NativeCommandContext *>(p);
  if (!n)
    throw std::runtime_error("foreign native command context");
  for (auto &f : n->commands)
    f();
  WaitGPU();
  for (auto &f : n->GetPostExecutionCallbacks())
    f();
  return 0;
}

int ComputeDevice::CreateProgram(const std::vector<ImageFormat> &, ImageFormat, double_ptr<Program>) {
  NativeUnsupported();
}

int ComputeDevice::CreateBottomLevelAccelerationStructure(BufferRange,
                                                          uint32_t,
                                                          uint32_t,
                                                          RayTracingGeometryFlag,
                                                          double_ptr<AccelerationStructure>) {
  NativeUnsupported();
}

int ComputeDevice::CreateBottomLevelAccelerationStructure(BufferRange vertices,
                                                          BufferRange indices,
                                                          uint32_t vertex_count,
                                                          uint32_t stride,
                                                          uint32_t primitive_count,
                                                          RayTracingGeometryFlag flags,
                                                          double_ptr<AccelerationStructure> output) {
  NativeUnsupported();
}

int ComputeDevice::CreateBottomLevelAccelerationStructure(Buffer *vertices,
                                                          Buffer *indices,
                                                          uint32_t stride,
                                                          double_ptr<AccelerationStructure> output) {
  NativeUnsupported();
}

int ComputeDevice::CreateTopLevelAccelerationStructure(const std::vector<RayTracingInstance> &instances,
                                                       double_ptr<AccelerationStructure> output) {
  NativeUnsupported();
}

int ComputeDevice::CreateRayTracingProgram(double_ptr<RayTracingProgram>) {
  NativeUnsupported();
}

}  // namespace sparkium::backend
