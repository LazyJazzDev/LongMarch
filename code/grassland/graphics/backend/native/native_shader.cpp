#include <slang-com-ptr.h>
#include <slang.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>

#include "native_internal.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace grassland::graphics::backend {
namespace {
std::string Read(const std::filesystem::path &p) {
  std::ifstream f(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}
void Write(const std::filesystem::path &p, const std::string &s) {
  std::ofstream f(p, std::ios::binary);
  f << s;
  if (!f)
    throw std::runtime_error("cannot write native shader temporary file");
}
void Replace(std::string &s, const std::string &a, const std::string &b) {
  size_t p = 0;
  while ((p = s.find(a, p)) != std::string::npos) {
    s.replace(p, a.size(), b);
    p += b.size();
  }
}
struct TempDirectory {
  std::filesystem::path path;
  TempDirectory() {
    std::random_device random;
    for (int attempt = 0; attempt < 100; ++attempt) {
      path = std::filesystem::temp_directory_path() / ("longmarch-native-" + std::to_string(random()));
      if (std::filesystem::create_directory(path))
        return;
    }
    throw std::runtime_error("cannot create private native shader directory");
  }
  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};
// DXC's single buffer-type templates predate Slang generics. Both native targets
// represent read-only and writable byte buffers by the same pointer/size ABI.
// Erase only these known templates, not their function bodies or algorithms.
// The original sources still compile unchanged through DXC on graphics devices.
std::string LowerSource(std::string s, const std::string &filename) {
  if (filename == "buffer_helper.hlsli") {
    auto a = s.find("template <class BufferType>\nclass BufferReference");
    auto b = s.find("template <>", a);
    auto f = s.find("template <class BufferType>\nBufferReference", a);
    if (a == std::string::npos || b == std::string::npos || f == std::string::npos)
      throw std::runtime_error("buffer helper template layout changed; update native lowering");
    auto helpers = s.substr(0, a);
    helpers = std::regex_replace(helpers, std::regex(R"(template\s*<class BufferType>\s*)"), "");
    Replace(helpers, "BufferType", "BufferReference");
    auto factory = s.substr(f, b - f);
    s.erase(a, b - a);
    auto end = s.find("template <class BufferType>\nclass StreamedBufferReference");
    s.insert(end, helpers + factory);
    s = std::regex_replace(s, std::regex(R"((  (?:float|uint|int)[234]? Load\w+\(\)))"), "  [mutating] $1");
  }
  s = std::regex_replace(s, std::regex(R"(template\s*<\s*(?:class\s+(?:BufferType|GeometrySamplerType|B))?\s*>\s*)"),
                         "");
  if (s.find("template <") != std::string::npos)
    throw std::runtime_error("unsupported native shader template in " + filename);
  Replace(s, ".template ", ".");
  s = std::regex_replace(s, std::regex(R"(\bclass\b)"), "struct");
  s = std::regex_replace(s, std::regex(R"(\bBufferType\b)"), "RWByteAddressBuffer");
  if (filename == "layout.hlsli")
    s = std::regex_replace(s, std::regex(R"(\bB\b)"), "RWByteAddressBuffer");
  Replace(s, "} random_device;", "};");
  Replace(s, "GeometrySamplerType", "GeometrySampler");
  s = std::regex_replace(s, std::regex(R"(<(?:RWByteAddressBuffer|ByteAddressBuffer)>)"), "");
  s = std::regex_replace(s, std::regex(R"(\bByteAddressBuffer\b)"), "RWByteAddressBuffer");
  Replace(s, "void SetTransform(", "[mutating] void SetTransform(");
  if (filename == "principled_bsdf.hlsli") {
    Replace(s, "void CalculateClosureWeight(", "[mutating] void CalculateClosureWeight(");
    Replace(s, "float PrincipledThinReflectionProbability(", "[mutating] float PrincipledThinReflectionProbability(");
    Replace(s, "void SamplePrincipledThinReflection(", "[mutating] void SamplePrincipledThinReflection(");
    // These methods call CalculateClosureWeight and update the local BSDF state.
    s = std::regex_replace(
        s, std::regex(R"((^|\n)(Spectrum|float3|void) (EvalPrincipledBSDF\w*|SamplePrincipledBSDF\w*)\()"),
        "$1[mutating] $2 $3(");
  }
  // Keep register-space identity as a reflected attribute on native targets.
  // Unlike hardcoded name maps, this follows conditional shader bindings.
  s = std::regex_replace(
      s,
      std::regex(
          R"(((?:cbuffer|(?:RW)?ByteAddressBuffer|ConstantBuffer<\w+>|(?:RW)?Texture2D<\w+>|SamplerState)\s+\w+\s*(?:\[\s*\])?)\s*:\s*register\([but s][0-9]+,\s*space([0-9]+)\))"),
      "[NativeBinding($2)] $1");
  s = std::regex_replace(s, std::regex(R"(\[\[vk::image_format\("[^"]+"\)\]\])"), "");
  for (const char *type : {"float4", "float3", "float2", "float", "int", "uint"}) {
    Replace(s, std::string("RWTexture2D<") + type + ">", std::string("NativeTexture_") + type);
    Replace(s, std::string("Texture2D<") + type + ">", std::string("NativeTexture_") + type);
  }
  Replace(s, "SamplerState", "NativeSamplerState");
  s = std::regex_replace(s, std::regex(R"(\bRayDesc\b)"), "NativeRayDesc");
  // Use an explicit descriptor span. CUDA unsized resource arrays otherwise
  // receive a reflection layout that differs from the emitted C++ structure.
  s = std::regex_replace(s,
                         std::regex(R"((RWByteAddressBuffer|NativeTexture_\w+|NativeSamplerState)\s+(\w+)\s*\[\s*\])"),
                         "NativeArray<$1> $2");
  return s;
}
std::string ImagePrelude() {
  std::string result = R"(
[__AttributeUsage(_AttributeTargets.Var)]
struct NativeBindingAttribute { int slot; };
struct NativeSamplerState { int filter; int address_u; int address_v; };
struct NativeRayDesc { float3 Origin; float TMin; float3 Direction; float TMax; };
struct NativeArray<T> {
  StructuredBuffer<T> elements;
  __subscript(uint i) -> T { get { return elements[i]; } }
};
int NativeAddress(int x,int n,int mode) {
  if(mode==0) return ((x%n)+n)%n;
  if(mode==1) { int k=((x%(2*n))+2*n)%(2*n); return k<n?k:2*n-1-k; }
  if(mode==2) return clamp(x,0,n-1);
  return x;
}
)";
  for (const char *type : {"float4", "float3", "float2", "float", "int", "uint"}) {
    std::string body = R"(
struct NativeTexture_TYPE {
  RWByteAddressBuffer pixels;
  uint width;
  uint height;
  uint unorm;
  uint padding;
  void GetDimensions(out uint w,out uint h) { w=width;h=height; }
  TYPE Load(int3 p) {
    if(p.x<0 || p.y<0 || p.x>=int(width) || p.y>=int(height)) return TYPE(0);
    uint index=p.y*width+p.x;
    if(unorm!=0) {
      uint bits=pixels.Load(index*4);
      float4 rgba=float4(bits&255,(bits>>8)&255,(bits>>16)&255,bits>>24)/255.0f;
      return TYPE(UNPACK);
    }
    return pixels.Load<TYPE>(index*sizeof(TYPE));
  }
  TYPE Fetch(int2 p,NativeSamplerState s) {
    return Load(int3(NativeAddress(p.x,int(width),s.address_u),NativeAddress(p.y,int(height),s.address_v),0));
  }
  TYPE SampleLevel(NativeSamplerState s,float2 uv,float level) {
    float2 p=uv*float2(width,height);
    if(s.filter==0) return Fetch(int2(floor(p)),s);
    p-=0.5f;int2 i=int2(floor(p));float2 t=frac(p);
    return TYPE(lerp(lerp(Fetch(i,s),Fetch(i+int2(1,0),s),t.x),
                     lerp(Fetch(i+int2(0,1),s),Fetch(i+int2(1,1),s),t.x),t.y));
  }
  __subscript(uint2 p) -> TYPE {
    get { return Load(int3(p,0)); }
    [nonmutating] set {
      if(p.x>=width || p.y>=height) return;
      uint index=p.y*width+p.x;
      if(unorm!=0) {
        float4 rgba=PACK;
        rgba=float4(isfinite(rgba.x)?rgba.x:0,isfinite(rgba.y)?rgba.y:0,
                    isfinite(rgba.z)?rgba.z:0,isfinite(rgba.w)?rgba.w:0);
        float4 scaled=saturate(rgba)*255.0f;
        uint4 lo=uint4(floor(scaled));
        float4 remainder=scaled-float4(lo);
        // Round to nearest, ties to even, like the previous CPU conversion.
        uint4 q=lo+uint4(remainder.x>0.5f || (remainder.x==0.5f && (lo.x&1)!=0),
                        remainder.y>0.5f || (remainder.y==0.5f && (lo.y&1)!=0),
                        remainder.z>0.5f || (remainder.z==0.5f && (lo.z&1)!=0),
                        remainder.w>0.5f || (remainder.w==0.5f && (lo.w&1)!=0));
        pixels.Store(index*4,q.x|(q.y<<8)|(q.z<<16)|(q.w<<24));
      } else pixels.Store<TYPE>(index*sizeof(TYPE),newValue);
    }
  }
};
)";
    std::string t = type;
    Replace(body, "UNPACK", t == "float4" ? "rgba" : t == "float3" ? "rgba.xyz" : t == "float2" ? "rgba.xy" : "rgba.x");
    Replace(body, "PACK",
            t == "float3"   ? "float4(newValue,1)"
            : t == "float2" ? "float4(newValue,0,1)"
                            : "float4(newValue)");
    Replace(body, "TYPE", type);
    result += body;
  }
  return result;
}
void SlangCheck(SlangResult result, const std::string &message) {
  if (SLANG_FAILED(result))
    throw std::runtime_error(message);
}
struct SlangSessionOwner {
  SlangSession *session = spCreateSession();
  SlangSessionOwner() {
    if (!session)
      throw std::runtime_error("failed to create Slang session");
  }
  ~SlangSessionOwner() {
    spDestroySession(session);
  }
};
SlangSession *Session() {
  static SlangSessionOwner owner;
  return owner.session;
}
struct RequestOwner {
  SlangCompileRequest *request = spCreateCompileRequest(Session());
  RequestOwner() {
    if (!request)
      throw std::runtime_error("failed to create Slang compile request");
  }
  ~RequestOwner() {
    spDestroyCompileRequest(request);
  }
};
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
void CheckNVRTC(nvrtcResult result) {
  if (result != NVRTC_SUCCESS)
    throw std::runtime_error(std::string("native NVRTC: ") + nvrtcGetErrorString(result));
}
struct NVRTCProgramOwner {
  nvrtcProgram program{};
  ~NVRTCProgramOwner() {
    if (program)
      nvrtcDestroyProgram(&program);
  }
};
#endif
struct VaryingInput {
  uint32_t start[3], end[3];
};
using HostKernel = void (*)(VaryingInput *, void *, void *);
struct Parameter {
  int slot;
  size_t offset, size;
  bool array;
  SlangTypeKind kind;
  std::string name;
};
}  // namespace
struct NativeShader::Impl {
  bool cuda;
  std::string entry;
  std::vector<Parameter> parameters;
  size_t global_size{};
  uint32_t threads[3]{};
  Slang::ComPtr<ISlangSharedLibrary> library;
  HostKernel host{};
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
  CUmodule module{};
  CUfunction kernel{};
  CUdeviceptr global_device{};
  size_t global_device_size{};
#endif
  ~Impl() {
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
    if (module)
      cuModuleUnload(module);
#endif
  }
};
NativeShader::NativeShader(bool cuda,
                           const VirtualFileSystem &vfs,
                           const std::string &source,
                           const std::string &entry,
                           const std::vector<std::string> &args)
    : impl_(std::make_unique<Impl>()) {
  auto start = std::chrono::steady_clock::now();
  impl_->cuda = cuda;
  impl_->entry = entry;
  // Slang host-callable modules export *_Group and *_Thread entry helpers.
  // Unique names prevent ELF interposition between separate material kernels.
  static std::atomic<uint64_t> next_entry{0};
  const auto native_entry = "LongMarchNativeEntry" + std::to_string(next_entry++);
  TempDirectory directory;
  vfs.SaveToDirectory(directory.path);
  for (const auto &file : std::filesystem::recursive_directory_iterator(directory.path)) {
    if (file.is_regular_file()) {
      auto lowered = LowerSource(Read(file.path()), file.path().filename().string());
      lowered = std::regex_replace(lowered, std::regex("\\b" + entry + "\\b"), native_entry);
      Write(file.path(), lowered);
    }
  }
  Write(directory.path / "native_input.slang", ImagePrelude() + "\n#include \"" + source + "\"\n");
  RequestOwner owner;
  auto *request = owner.request;
  spSetCodeGenTarget(request, cuda ? SLANG_CUDA_SOURCE : SLANG_SHADER_HOST_CALLABLE);
  spSetMatrixLayoutMode(request, SLANG_MATRIX_LAYOUT_ROW_MAJOR);
  spSetOptimizationLevel(request, SLANG_OPTIMIZATION_LEVEL_MAXIMAL);
  spSetTargetFloatingPointMode(request, 0, SLANG_FLOATING_POINT_MODE_PRECISE);
  spAddSearchPath(request, directory.path.string().c_str());
  spAddPreprocessorDefine(request, "SPARKIUM_NATIVE", "1");
  // Slang 2026.8 emits the HLSL 'precise' token verbatim in C++. Enforce
  // non-contracting arithmetic downstream instead (NVRTC uses --fmad=false).
  spAddPreprocessorDefine(request, "precise", "");
  if (!cuda) {
    const char *options[] = {"-Xgcc", "-ffp-contract=off"};
    SlangCheck(spProcessCommandLineArguments(request, options, 2), "cannot set strict CPU arithmetic");
    if (std::getenv("SPARKIUM_NATIVE_ASAN")) {
      const char *asan[] = {"-Xgcc", "-fsanitize=address"};
      SlangCheck(spProcessCommandLineArguments(request, asan, 2), "cannot enable CPU shader ASan");
    }
  }
  if (!cuda)
    spAddPreprocessorDefine(request, "SPARKIUM_NATIVE_CPU", "1");
  for (const auto &arg : args) {
    if (arg.rfind("-D", 0) == 0) {
      auto equal = arg.find('=');
      auto key = arg.substr(2, equal == std::string::npos ? equal : equal - 2);
      auto value = equal == std::string::npos ? "1" : arg.substr(equal + 1);
      spAddPreprocessorDefine(request, key.c_str(), value.c_str());
    } else if (arg.rfind("-I", 0) != 0)
      throw std::runtime_error("unsupported native shader option: " + arg);
  }
  int unit = spAddTranslationUnit(request, SLANG_SOURCE_LANGUAGE_SLANG, "native_input");
  spAddTranslationUnitSourceFile(request, unit, (directory.path / "native_input.slang").string().c_str());
  spAddEntryPoint(request, unit, native_entry.c_str(), SLANG_STAGE_COMPUTE);
  if (const char *dump = std::getenv("SPARKIUM_NATIVE_DUMP")) {
    std::filesystem::create_directories(dump);
    const auto prefix =
        (std::filesystem::path(dump) / (std::filesystem::path(source).stem().string() + "-" + entry + "-")).string();
    spSetDumpIntermediates(request, 1);
    spSetDumpIntermediatePrefix(request, prefix.c_str());
  }
  auto status = spCompile(request);
  if (SLANG_FAILED(status))
    throw std::runtime_error("Slang " + source + ":" + entry + "\n" + spGetDiagnosticOutput(request));
  auto *reflection = reinterpret_cast<slang::ShaderReflection *>(spGetReflection(request));
  // The global parameter type is a ConstantBuffer on native targets: its
  // reported size is a pointer, not the size of the pointed-to parameter block.
  impl_->global_size = 0;
  SlangUInt threads[3];
  reflection->getEntryPointByIndex(0)->getComputeThreadGroupSize(3, threads);
  for (int i = 0; i < 3; ++i)
    impl_->threads[i] = uint32_t(threads[i]);
  for (unsigned i = 0; i < reflection->getParameterCount(); ++i) {
    auto *p = reflection->getParameterByIndex(i);
    auto *attribute = p->getVariable()->findUserAttributeByName(Session(), "NativeBinding");
    if (!attribute)
      throw std::runtime_error(std::string("native shader parameter lacks register space: ") + p->getName());
    int slot = -1;
    SlangCheck(attribute->getArgumentValueInt(0, &slot), "invalid native binding attribute");
    auto *type = p->getTypeLayout();
    bool array = type->getType()->getName() && std::string(type->getType()->getName()) == "NativeArray";
    auto kind = (SlangTypeKind)type->getKind();
    impl_->parameters.push_back({slot, p->getOffset(), type->getSize(), array, kind, p->getName()});
    impl_->global_size = std::max(impl_->global_size, p->getOffset() + type->getSize());
  }
  if (cuda) {
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
    Slang::ComPtr<ISlangBlob> blob;
    SlangCheck(spGetEntryPointCodeBlob(request, 0, 0, blob.writeRef()), "cannot obtain native CUDA source");
    std::string code(static_cast<const char *>(blob->getBufferPointer()), blob->getBufferSize());
    if (const char *dump = std::getenv("SPARKIUM_NATIVE_DUMP")) {
      std::filesystem::create_directories(dump);
      Write(std::filesystem::path(dump) / (std::filesystem::path(source).stem().string() + "-" + entry + ".cu"), code);
    }
    NVRTCProgramOwner nvrtc;
    CheckNVRTC(nvrtcCreateProgram(&nvrtc.program, code.c_str(), "sparkium.cu", 0, nullptr, nullptr));
    auto program = nvrtc.program;
    CUdevice device;
    CheckCUDA(cuCtxGetDevice(&device));
    int major, minor;
    CheckCUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CheckCUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    std::string arch = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
    const char *options[] = {arch.c_str(), "--std=c++17", "--fmad=false"};
    auto result = nvrtcCompileProgram(program, 3, options);
    size_t log_size = 0;
    CheckNVRTC(nvrtcGetProgramLogSize(program, &log_size));
    std::string log(log_size, '\0');
    CheckNVRTC(nvrtcGetProgramLog(program, log.data()));
    if (result != NVRTC_SUCCESS)
      throw std::runtime_error("NVRTC " + source + ":" + entry + "\n" + log);
    size_t size;
    CheckNVRTC(nvrtcGetPTXSize(program, &size));
    std::string ptx(size, '\0');
    CheckNVRTC(nvrtcGetPTX(program, ptx.data()));
    CheckCUDA(cuModuleLoadData(&impl_->module, ptx.c_str()));
    CheckCUDA(cuModuleGetFunction(&impl_->kernel, impl_->module, native_entry.c_str()));
    CheckCUDA(
        cuModuleGetGlobal(&impl_->global_device, &impl_->global_device_size, impl_->module, "SLANG_globalParams"));
    if (impl_->global_device_size < impl_->global_size)
      throw std::runtime_error("CUDA global parameter ABI mismatch");
#else
    throw std::runtime_error("native CUDA backend was not built");
#endif
  } else {
    SlangCheck(spGetEntryPointHostCallable(request, 0, 0, impl_->library.writeRef()), "cannot load native CPU kernel");
    impl_->host = reinterpret_cast<HostKernel>(impl_->library->findSymbolAddressByName(native_entry.c_str()));
    if (!impl_->host)
      throw std::runtime_error("native CPU entry point missing");
  }
  std::cout << "Compiled " << (cuda ? "CUDA" : "CPU") << " kernel " << source << ":" << entry << " ("
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s)\n";
}
NativeShader::~NativeShader() = default;
std::string NativeShader::EntryPoint() const {
  return impl_->entry;
}
void NativeShader::Dispatch(const NativeBindings &bindings, uint32_t x, uint32_t y, uint32_t z) {
  if (!x || !y || !z)
    return;
  const bool trace = std::getenv("SPARKIUM_NATIVE_TRACE");
  if (trace)
    std::cerr << "Dispatch " << impl_->entry << " " << x << "," << y << "," << z << " globals=" << impl_->global_size
              << "\n";
  std::vector<uint8_t> globals(impl_->global_size);
  std::vector<std::unique_ptr<NativeMemory>> allocations;
  auto allocate = [&](const void *data, size_t size) {
    auto memory = std::make_unique<NativeMemory>(impl_->cuda, size);
    memory->Upload(data, size);
    void *ptr = memory->Data();
    allocations.push_back(std::move(memory));
    return ptr;
  };
  for (const auto &p : impl_->parameters) {
    if (trace)
      std::cerr << "  " << p.name << " offset=" << p.offset << " size=" << p.size << " slot=" << p.slot
                << " array=" << p.array << "\n";
    std::vector<uint8_t> elements;
    size_t count = 0;
    auto append = [&](const auto &v) {
      auto old = elements.size();
      elements.resize(old + sizeof(v));
      std::memcpy(elements.data() + old, &v, sizeof(v));
      ++count;
    };
    if (auto it = bindings.buffers.find(p.slot); it != bindings.buffers.end()) {
      for (const auto &range : it->second) {
        auto *b = dynamic_cast<NativeBuffer *>(range.buffer);
        if (!b)
          throw std::runtime_error("foreign native buffer");
        if (range.offset > b->Size() || range.size > b->Size() - range.offset)
          throw std::out_of_range("native buffer binding range");
        void *ptr = static_cast<char *>(b->memory->Data()) + range.offset;
        if (p.kind == SLANG_TYPE_KIND_CONSTANT_BUFFER)
          append(ptr);
        else
          append(NativeSpan{ptr, range.size});
      }
    } else if (auto it = bindings.images.find(p.slot); it != bindings.images.end()) {
      for (auto *image : it->second) {
        auto *n = dynamic_cast<NativeImage *>(image);
        if (!n)
          throw std::runtime_error("foreign native image");
        append(n->Binding());
      }
    } else if (auto it = bindings.samplers.find(p.slot); it != bindings.samplers.end()) {
      for (auto *sampler : it->second) {
        auto *n = dynamic_cast<NativeSampler *>(sampler);
        if (!n)
          throw std::runtime_error("foreign native sampler");
        struct Data {
          int filter, u, v;
        };
        append(Data{int(n->info.min_filter), int(n->info.address_mode_u), int(n->info.address_mode_v)});
      }
    } else
      throw std::runtime_error("missing native binding: " + p.name);
    if (p.array) {
      NativeSpan array{allocate(elements.data(), elements.size()), count};
      if (p.size != sizeof(array))
        throw std::runtime_error("native array ABI mismatch: " + p.name);
      std::memcpy(globals.data() + p.offset, &array, sizeof(array));
    } else {
      if (elements.size() != p.size)
        throw std::runtime_error("native parameter ABI mismatch: " + p.name + " expected " + std::to_string(p.size) +
                                 " got " + std::to_string(elements.size()));
      std::memcpy(globals.data() + p.offset, elements.data(), elements.size());
    }
  }
  if (impl_->cuda) {
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
    // Slang 2026 uses CUDA module constant memory for global parameters.
    CheckCUDA(cuMemcpyHtoD(impl_->global_device, globals.data(), globals.size()));
    CheckCUDA(cuLaunchKernel(impl_->kernel, x, y, z, impl_->threads[0], impl_->threads[1], impl_->threads[2], 0,
                             nullptr, nullptr, nullptr));
    // Resource descriptors are launch-local. Keep them alive until completion;
    // errors are attributed to this launch instead of a later image download.
    CheckCUDA(cuCtxSynchronize());
#endif
  } else {
    const uint64_t groups = uint64_t(x) * y * z;
    // Row/group independent kernels. CPU scan variants explicitly avoid barriers.
    // Cap default CPU workers at six; OMP_NUM_THREADS can request fewer workers.
#ifdef _OPENMP
    const int workers = std::min(6, omp_get_max_threads());
#pragma omp parallel for schedule(dynamic, 8) num_threads(workers) if (groups > 16)
#endif
    for (int64_t group = 0; group < static_cast<int64_t>(groups); ++group) {
      VaryingInput input{{uint32_t(group % x), uint32_t(group / x % y), uint32_t(group / (uint64_t(x) * y))}, {}};
      for (int i = 0; i < 3; ++i)
        input.end[i] = input.start[i] + 1;
      impl_->host(&input, nullptr, globals.data());
    }
  }
}
}  // namespace grassland::graphics::backend
