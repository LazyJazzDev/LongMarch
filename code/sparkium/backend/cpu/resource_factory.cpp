#include <iostream>

#include "sparkium/backend/common/native_buffer.h"
#include "sparkium/backend/common/native_command_context.h"
#include "sparkium/backend/common/native_image.h"
#include "sparkium/backend/common/native_program.h"
#include "sparkium/backend/common/native_sampler.h"
#include "sparkium/backend/common/native_util.h"
#include "sparkium/backend/cpu/cpu_device.h"

namespace sparkium::backend {
int CpuDevice::CreateBuffer(size_t s, BufferType t, double_ptr<Buffer> p) {
  p.construct<NativeBuffer>(false, s, t);
  return 0;
}

int CpuDevice::CreateImage(int w, int h, ImageFormat f, double_ptr<Image> p) {
  p.construct<NativeImage>(false, w, h, f);
  return 0;
}

int CpuDevice::CreateSampler(const SamplerInfo &i, double_ptr<Sampler> p) {
  p.construct<NativeSampler>(i);
  return 0;
}

int CpuDevice::CreateShader(const std::string &s, const std::string &e, const std::string &t, double_ptr<Shader> p) {
  VirtualFileSystem v;
  v.WriteFile("input.hlsl", s);
  return CreateShader(v, "input.hlsl", e, t, {}, p);
}

int CpuDevice::CreateShader(const VirtualFileSystem &v,
                            const std::string &s,
                            const std::string &e,
                            const std::string &t,
                            double_ptr<Shader> p) {
  return CreateShader(v, s, e, t, {}, p);
}

int CpuDevice::CreateShader(const VirtualFileSystem &v,
                            const std::string &s,
                            const std::string &e,
                            const std::string &t,
                            const std::vector<std::string> &a,
                            double_ptr<Shader> p) {
  if (t.rfind("cs_", 0) != 0)
    NativeUnsupported();
  OptixDevice *optix = nullptr;
  p.construct<NativeShader>(false, v, s, e, a, optix);
  return 0;
}

int CpuDevice::CreateComputeProgram(Shader *s, double_ptr<ComputeProgram> p) {
  auto *n = dynamic_cast<NativeShader *>(s);
  if (!n)
    throw std::runtime_error("foreign native shader");
  p.construct<NativeProgram>(n);
  return 0;
}

int CpuDevice::CreateCommandContext(double_ptr<CommandContext> p) {
  p.construct<NativeCommandContext>(this);
  return 0;
}

int CpuDevice::SubmitCommandContext(CommandContext *p) {
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

int CpuDevice::CreateProgram(const std::vector<ImageFormat> &, ImageFormat, double_ptr<Program>) {
  NativeUnsupported();
}

int CpuDevice::CreateBottomLevelAccelerationStructure(BufferRange,
                                                      uint32_t,
                                                      uint32_t,
                                                      RayTracingGeometryFlag,
                                                      double_ptr<AccelerationStructure>) {
  NativeUnsupported();
}

int CpuDevice::CreateBottomLevelAccelerationStructure(BufferRange vertices,
                                                      BufferRange indices,
                                                      uint32_t vertex_count,
                                                      uint32_t stride,
                                                      uint32_t primitive_count,
                                                      RayTracingGeometryFlag flags,
                                                      double_ptr<AccelerationStructure> output) {
  NativeUnsupported();
}

int CpuDevice::CreateBottomLevelAccelerationStructure(Buffer *vertices,
                                                      Buffer *indices,
                                                      uint32_t stride,
                                                      double_ptr<AccelerationStructure> output) {
  NativeUnsupported();
}

int CpuDevice::CreateTopLevelAccelerationStructure(const std::vector<RayTracingInstance> &instances,
                                                   double_ptr<AccelerationStructure> output) {
  NativeUnsupported();
}

int CpuDevice::CreateRayTracingProgram(double_ptr<RayTracingProgram>) {
  NativeUnsupported();
}

}  // namespace sparkium::backend
