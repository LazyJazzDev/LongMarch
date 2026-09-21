#include <iostream>

#include "sparkium/backend/cuda/cuda_buffer.h"
#include "sparkium/backend/cuda/cuda_command_context.h"
#include "sparkium/backend/cuda/cuda_device.h"
#include "sparkium/backend/cuda/cuda_image.h"
#include "sparkium/backend/cuda/cuda_program.h"
#include "sparkium/backend/cuda/cuda_sampler.h"
#include "sparkium/backend/cuda/cuda_util.h"

namespace sparkium::backend {
using namespace cuda;

int CudaDevice::CreateBuffer(size_t s, BufferType t, double_ptr<Buffer> p) {
  p.construct<CudaBuffer>(s, t);
  return 0;
}

int CudaDevice::CreateImage(int w, int h, ImageFormat f, double_ptr<Image> p) {
  p.construct<CudaImage>(w, h, f);
  return 0;
}

int CudaDevice::CreateSampler(const SamplerInfo &i, double_ptr<Sampler> p) {
  p.construct<CudaSampler>(i);
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
    CudaUnsupported();
  OptixDevice *optix = Optix();
  p.construct<CudaShader>(v, s, e, a, optix);
  return 0;
}

int CudaDevice::CreateComputeProgram(Shader *s, double_ptr<ComputeProgram> p) {
  auto *n = dynamic_cast<CudaShader *>(s);
  if (!n)
    throw std::runtime_error("foreign compute shader");
  p.construct<CudaProgram>(n);
  return 0;
}

int CudaDevice::CreateCommandContext(double_ptr<CommandContext> p) {
  p.construct<CudaCommandContext>(this);
  return 0;
}

int CudaDevice::SubmitCommandContext(CommandContext *p) {
  auto *n = dynamic_cast<CudaCommandContext *>(p);
  if (!n)
    throw std::runtime_error("foreign compute command context");
  for (auto &f : n->commands)
    f();
  WaitGPU();
  for (auto &f : n->GetPostExecutionCallbacks())
    f();
  return 0;
}

int CudaDevice::CreateProgram(const std::vector<ImageFormat> &, ImageFormat, double_ptr<Program>) {
  CudaUnsupported();
}

int CudaDevice::CreateRayTracingProgram(double_ptr<RayTracingProgram>) {
  CudaUnsupported();
}

}  // namespace sparkium::backend
