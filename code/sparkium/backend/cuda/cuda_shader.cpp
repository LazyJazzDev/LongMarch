#include "sparkium/backend/cuda/cuda_shader.h"

#include <slang-com-ptr.h>
#include <slang.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>

#include "sparkium/backend/cuda/cuda_buffer.h"
#include "sparkium/backend/cuda/cuda_image.h"
#include "sparkium/backend/cuda/cuda_sampler.h"
#include "sparkium/backend/cuda/cuda_shader_internal.h"
#include "sparkium/backend/cuda/cuda_util.h"
#ifdef SPARKIUM_OPTIX_ENABLED
#include "sparkium/backend/cuda/optix_acceleration_structure.h"
#include "sparkium/backend/cuda/optix_launch.h"
#endif

namespace sparkium::backend::cuda {
namespace {
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
      path = std::filesystem::temp_directory_path() / ("longmarch-compute-" + std::to_string(random()));
      if (std::filesystem::create_directory(path))
        return;
    }
    throw std::runtime_error("cannot create private compute shader directory");
  }

  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

std::string ImagePrelude() {
  std::string result = R"(
[__AttributeUsage(_AttributeTargets.Var)]
struct ComputeBindingAttribute { int slot; };
struct ComputeContext { uint64_t slots[256]; };
T ComputeResource<T>(ComputeContext* context, uint slot) {
  return *((T*)&context->slots[slot * 4]);
}
struct ComputeSamplerState { int filter; int address_u; int address_v; };
struct ComputeRayDesc { float3 Origin; float TMin; float3 Direction; float TMax; };
struct ComputeArray<T> {
  StructuredBuffer<T> elements;
  __subscript(uint i) -> T { get { return elements[i]; } }
};
int ComputeAddress(int x,int n,int mode) {
  if(mode==0) return ((x%n)+n)%n;
  if(mode==1) { int k=((x%(2*n))+2*n)%(2*n); return k<n?k:2*n-1-k; }
  if(mode==2) return clamp(x,0,n-1);
  return x;
}
)";
  for (const char *type : {"float4", "float3", "float2", "float", "int", "uint"}) {
    std::string body = R"(
struct ComputeTexture_TYPE {
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
  TYPE Fetch(int2 p,ComputeSamplerState s) {
    return Load(int3(ComputeAddress(p.x,int(width),s.address_u),ComputeAddress(p.y,int(height),s.address_v),0));
  }
  TYPE SampleLevel(ComputeSamplerState s,float2 uv,float level) {
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

}  // namespace

CudaShader::Impl::~Impl() {
#ifdef SPARKIUM_CUDA_ENABLED
  ReleaseCUDA();
#endif
}

CudaShader::CudaShader(const VirtualFileSystem &vfs,
                       const std::string &source,
                       const std::string &entry,
                       const std::vector<std::string> &args,
                       OptixDevice *optix)
    : impl_(std::make_shared<Impl>()) {
  // The compiler session is shared; compilation is serialized, execution is not.
  Session();  // Outlives cached JIT libraries during static destruction.
  static std::mutex compile_mutex;
  std::lock_guard<std::mutex> compile_lock(compile_mutex);
  auto start = std::chrono::steady_clock::now();
  impl_->entry = entry;
  const bool optix_shader = std::find(args.begin(), args.end(), "-DSPARKIUM_OPTIX") != args.end();
  if (optix_shader && !optix)
    throw std::runtime_error("OptiX shaders require available CUDA hardware ray tracing support");
  // Unique function and resource names prevent interposition between JIT modules.
  static std::atomic<uint64_t> next_entry{0};
  const auto compute_entry = "LongMarchComputeEntry" + std::to_string(next_entry++);
  TempDirectory directory;
  vfs.SaveToDirectory(directory.path);
  if (!std::filesystem::exists(directory.path / "compute_contract.hlsli"))
    throw std::invalid_argument(
        "compute shaders require compute_contract.hlsli and the SP_RESOURCE/SP_CONTEXT contract");

  Write(directory.path / "compute_input.slang", ImagePrelude() + "\n#include \"" + source + "\"\n");
  RequestOwner owner;
  auto *request = owner.request;
  spSetCodeGenTarget(request, SLANG_CUDA_SOURCE);
  if (optix_shader)
    spSetTargetFlags(request, 0, SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM);
  spSetMatrixLayoutMode(request, SLANG_MATRIX_LAYOUT_ROW_MAJOR);
  spSetOptimizationLevel(request, SLANG_OPTIMIZATION_LEVEL_MAXIMAL);
  spSetTargetFloatingPointMode(request, 0, SLANG_FLOATING_POINT_MODE_PRECISE);
  spAddSearchPath(request, directory.path.string().c_str());
  spAddPreprocessorDefine(request, "SPARKIUM_COMPUTE", "1");
  spAddPreprocessorDefine(request, entry.c_str(), compute_entry.c_str());
  // NVRTC uses --fmad=false; LLVM receives the precise target mode above.
  spAddPreprocessorDefine(request, "precise", "");

  for (const auto &arg : args) {
    if (arg.rfind("-D", 0) == 0) {
      auto equal = arg.find('=');
      auto key = arg.substr(2, equal == std::string::npos ? equal : equal - 2);
      auto value = equal == std::string::npos ? "1" : arg.substr(equal + 1);
      spAddPreprocessorDefine(request, key.c_str(), value.c_str());
    } else if (arg.rfind("-I", 0) != 0)
      throw std::runtime_error("unsupported compute shader option: " + arg);
  }

  int unit = spAddTranslationUnit(request, SLANG_SOURCE_LANGUAGE_SLANG, "compute_input");
  spAddTranslationUnitSourceFile(request, unit, (directory.path / "compute_input.slang").string().c_str());
  spAddEntryPoint(request, unit, compute_entry.c_str(),
                  optix_shader ? SLANG_STAGE_RAY_GENERATION : SLANG_STAGE_COMPUTE);
  if (optix_shader) {
    spAddEntryPoint(request, unit, "LongMarchOptixClosest", SLANG_STAGE_CLOSEST_HIT);
    spAddEntryPoint(request, unit, "LongMarchOptixMiss", SLANG_STAGE_MISS);
  }

  if (const char *dump = std::getenv("SPARKIUM_COMPUTE_DUMP")) {
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
  // The global parameter type is a ConstantBuffer on compute targets: its
  // reported size is a pointer, not the size of the pointed-to parameter block.
  impl_->global_size = 0;
  // The shared tracer records 8x8 compute groups; OptiX launches individual pixels.
  SlangUInt threads[3]{8, 8, 1};
  if (!optix_shader)
    reflection->getEntryPointByIndex(0)->getComputeThreadGroupSize(3, threads);
  for (int i = 0; i < 3; ++i)
    impl_->threads[i] = uint32_t(threads[i]);

  for (unsigned i = 0; i < reflection->getParameterCount(); ++i) {
    auto *p = reflection->getParameterByIndex(i);
    auto *attribute = p->getVariable()->findUserAttributeByName(Session(), "ComputeBinding");
    if (!attribute)
      throw std::runtime_error(std::string("compute shader parameter lacks register space: ") + p->getName());
    int slot = -1;
    SlangCheck(attribute->getArgumentValueInt(0, &slot), "invalid compute binding attribute");
    if (slot < 0 || slot > 63)
      throw std::runtime_error("compute binding slot out of range");
    auto *type = p->getTypeLayout();
    bool array = type->getType()->getName() && std::string(type->getType()->getName()) == "ComputeArray";
    auto kind = (SlangTypeKind)type->getKind();
    impl_->parameters.push_back({slot, p->getOffset(), type->getSize(), array, kind, p->getName()});
    impl_->global_size = std::max(impl_->global_size, p->getOffset() + type->getSize());
    if (kind == SLANG_TYPE_KIND_CONSTANT_BUFFER) {
      auto *layout = type->getElementTypeLayout();
      for (unsigned f = 0; f < layout->getFieldCount(); ++f) {
        auto *field = layout->getFieldByIndex(f);
        impl_->parameters.back().constant_size =
            std::max(impl_->parameters.back().constant_size, field->getOffset() + field->getTypeLayout()->getSize());
      }
    }
  }

#ifdef SPARKIUM_CUDA_ENABLED
  Slang::ComPtr<ISlangBlob> blob;
  const auto code_status = optix_shader ? spGetTargetCodeBlob(request, 0, blob.writeRef())
                                        : spGetEntryPointCodeBlob(request, 0, 0, blob.writeRef());
  if (SLANG_FAILED(code_status)) {
    const char *diagnostic = spGetDiagnosticOutput(request);
    throw std::runtime_error(std::string("cannot obtain compute CUDA source: ") + (diagnostic ? diagnostic : ""));
  }
  std::string code(static_cast<const char *>(blob->getBufferPointer()), blob->getBufferSize());
  if (const char *dump = std::getenv("SPARKIUM_COMPUTE_DUMP")) {
    std::filesystem::create_directories(dump);
    Write(std::filesystem::path(dump) / (std::filesystem::path(source).stem().string() + "-" + entry + ".cu"), code);
  }
  impl_->CompileCUDA(code, source, compute_entry, optix_shader, optix);
#else
  throw std::runtime_error("compute CUDA backend was not built");
#endif

  std::cout << "Compiled " << (optix_shader ? "OptiX kernel " : "CUDA kernel ") << source << ":" << entry << " ("
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s)\n";
}

CudaShader::~CudaShader() = default;

std::string CudaShader::EntryPoint() const {
  return impl_->entry;
}

void CudaShader::Dispatch(const CudaBindings &bindings, uint32_t x, uint32_t y, uint32_t z) {
  if (!x || !y || !z)
    return;
  const bool trace = std::getenv("SPARKIUM_COMPUTE_TRACE");
  if (trace)
    std::cerr << "Dispatch " << impl_->entry << " " << x << "," << y << "," << z << " globals=" << impl_->global_size
              << "\n";
  std::vector<uint8_t> globals(impl_->global_size);
  std::vector<std::unique_ptr<CudaMemory>> allocations;
  auto allocate = [&](const void *data, size_t size) {
    auto memory = std::make_unique<CudaMemory>(size);
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
    if (auto it = bindings.acceleration_structures.find(p.slot); it != bindings.acceleration_structures.end()) {
#ifdef SPARKIUM_OPTIX_ENABLED
      auto *as = dynamic_cast<OptixAccelerationStructure *>(it->second);
      if (!as || as->Device() != impl_->optix_device || !as->IsTopLevel())
        throw std::runtime_error("foreign OptiX acceleration structure");
      append(as->Handle());
#else
      CudaUnsupported();
#endif
    } else if (auto it = bindings.buffers.find(p.slot); it != bindings.buffers.end()) {
      for (const auto &range : it->second) {
        auto *b = dynamic_cast<CudaBuffer *>(range.buffer);
        if (!b)
          throw std::runtime_error("foreign compute buffer");
        if (range.offset > b->Size() || range.size > b->Size() - range.offset)
          throw std::out_of_range("compute buffer binding range");
        void *ptr = static_cast<char *>(b->memory->Data()) + range.offset;
        if (p.kind == SLANG_TYPE_KIND_CONSTANT_BUFFER) {
          if (range.size < p.constant_size)
            throw std::out_of_range("compute constant buffer range");
          append(ptr);
        } else
          append(CudaSpan{ptr, range.size});
      }
    } else if (auto it = bindings.images.find(p.slot); it != bindings.images.end()) {
      for (auto *image : it->second) {
        auto *n = dynamic_cast<CudaImage *>(image);
        if (!n)
          throw std::runtime_error("foreign compute image");
        append(n->Binding());
      }
    } else if (auto it = bindings.samplers.find(p.slot); it != bindings.samplers.end()) {
      for (auto *sampler : it->second) {
        auto *n = dynamic_cast<CudaSampler *>(sampler);
        if (!n)
          throw std::runtime_error("foreign compute sampler");

        struct Data {
          int filter, u, v;
        };

        append(Data{int(n->info.min_filter), int(n->info.address_mode_u), int(n->info.address_mode_v)});
      }
    } else
      throw std::runtime_error("missing compute binding: " + p.name);
    if (p.array) {
      CudaSpan array{allocate(elements.data(), elements.size()), count};
      if (p.size != sizeof(array))
        throw std::runtime_error("compute array ABI mismatch: " + p.name);
      std::memcpy(globals.data() + p.offset, &array, sizeof(array));
    } else {
      if (elements.size() != p.size)
        throw std::runtime_error("compute parameter ABI mismatch: " + p.name + " expected " + std::to_string(p.size) +
                                 " got " + std::to_string(elements.size()));
      std::memcpy(globals.data() + p.offset, elements.data(), elements.size());
    }
  }

#ifdef SPARKIUM_CUDA_ENABLED
  impl_->DispatchCUDA(globals, x, y, z);
#endif
}
}  // namespace sparkium::backend::cuda
