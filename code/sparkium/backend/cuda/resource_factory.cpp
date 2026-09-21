#include <iostream>

#include "sparkium/backend/common/native_buffer.h"
#include "sparkium/backend/common/native_command_context.h"
#include "sparkium/backend/common/native_image.h"
#include "sparkium/backend/common/native_program.h"
#include "sparkium/backend/common/native_sampler.h"
#include "sparkium/backend/common/native_util.h"
#include "sparkium/backend/cuda/cuda_device.h"

namespace sparkium::backend {
int CudaDevice::CreateBuffer(size_t s, BufferType t, double_ptr<Buffer> p) {
  p.construct<NativeBuffer>(true, s, t);
  return 0;
}

int CudaDevice::CreateImage(int w, int h, ImageFormat f, double_ptr<Image> p) {
  p.construct<NativeImage>(true, w, h, f);
  return 0;
}

int CudaDevice::CreateSampler(const SamplerInfo &i, double_ptr<Sampler> p) {
  p.construct<NativeSampler>(i);
  return 0;
}

int CudaDevice::CreateShader(const std::string &s, const std::string &e, const std::string &t, double_ptr<Shader> p) {
  VirtualFileSystem v;
  v.WriteFile("input.hlsl", s);
  return CreateShader(v, "input.hlsl", e, t, {}, p);
}

int CudaDevice::CreateShader(const VirtualFileSystem &v,
                             const std::string &s,
                             const std::string &e,
                             const std::string &t,
                             double_ptr<Shader> p) {
  return CreateShader(v, s, e, t, {}, p);
}

int CudaDevice::CreateShader(const VirtualFileSystem &v,
                             const std::string &s,
                             const std::string &e,
                             const std::string &t,
                             const std::vector<std::string> &a,
                             double_ptr<Shader> p) {
  if (t.rfind("cs_", 0) != 0)
    NativeUnsupported();
  OptixDevice *optix = Optix();
  p.construct<NativeShader>(true, v, s, e, a, optix);
  return 0;
}

int CudaDevice::CreateComputeProgram(Shader *s, double_ptr<ComputeProgram> p) {
  auto *n = dynamic_cast<NativeShader *>(s);
  if (!n)
    throw std::runtime_error("foreign native shader");
  p.construct<NativeProgram>(n);
  return 0;
}

int CudaDevice::CreateCommandContext(double_ptr<CommandContext> p) {
  p.construct<NativeCommandContext>(this);
  return 0;
}

int CudaDevice::SubmitCommandContext(CommandContext *p) {
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

int CudaDevice::CreateProgram(const std::vector<ImageFormat> &, ImageFormat, double_ptr<Program>) {
  NativeUnsupported();
}

int CudaDevice::CreateRayTracingProgram(double_ptr<RayTracingProgram>) {
  NativeUnsupported();
}

}  // namespace sparkium::backend
