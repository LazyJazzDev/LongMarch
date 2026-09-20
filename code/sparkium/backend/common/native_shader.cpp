#include "sparkium/backend/common/native_shader.h"

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
#include <set>
#include <sstream>

#include "sparkium/backend/common/native_buffer.h"
#include "sparkium/backend/common/native_image.h"
#include "sparkium/backend/common/native_sampler.h"
#include "sparkium/backend/common/native_shader_compat.h"
#include "sparkium/backend/common/native_shader_internal.h"
#include "sparkium/backend/common/native_util.h"
#ifdef SPARKIUM_OPTIX_ENABLED
#include "sparkium/backend/cuda/native_acceleration_structure.h"
#include "sparkium/backend/cuda/optix_launch.h"
#endif

namespace sparkium::backend {
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

std::string ImagePrelude() {
  std::string result = R"(
[__AttributeUsage(_AttributeTargets.Var)]
struct NativeBindingAttribute { int slot; };
struct NativeContext { uint64_t slots[256]; };
T NativeResource<T>(NativeContext* context, uint slot) {
  return *((T*)&context->slots[slot * 4]);
}
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

}  // namespace

NativeShader::Impl::~Impl() {
#ifdef SPARKIUM_NATIVE_CUDA_ENABLED
  ReleaseCUDA();
#endif
}

NativeShader::NativeShader(bool cuda,
                           const VirtualFileSystem &vfs,
                           const std::string &source,
                           const std::string &entry,
                           const std::vector<std::string> &args,
                           OptixDevice *optix)
    : impl_(std::make_shared<Impl>()) {
  // The compiler session is shared; compilation is serialized, execution is not.
  Session();  // Outlives cached JIT libraries during static destruction.
  static std::mutex compile_mutex;
  std::lock_guard<std::mutex> compile_lock(compile_mutex);
  static std::vector<std::pair<std::string, std::shared_ptr<Impl>>> cache;
  auto start = std::chrono::steady_clock::now();
  impl_->cuda = cuda;
  impl_->entry = entry;
  const bool optix_shader = std::find(args.begin(), args.end(), "-DSPARKIUM_OPTIX") != args.end();
  if (optix_shader && !optix)
    throw std::runtime_error("OptiX shaders require available CUDA hardware ray tracing support");
  // Unique function and resource names prevent interposition between JIT modules.
  static std::atomic<uint64_t> next_entry{0};
  const auto native_entry = "LongMarchNativeEntry" + std::to_string(next_entry++);
  std::set<std::string> constant_blocks;
  TempDirectory directory;
  vfs.SaveToDirectory(directory.path);
  const bool explicit_context = std::filesystem::exists(directory.path / "native_contract.hlsli");
  impl_->explicit_context = explicit_context;
  std::string cache_key;
  if (explicit_context && !cuda) {
    // Exact content keys: no stale modules after edits, no hash collision aliasing.
    std::vector<std::filesystem::path> files;
    for (const auto &file : std::filesystem::recursive_directory_iterator(directory.path))
      if (file.is_regular_file())
        files.push_back(file.path());
    std::sort(files.begin(), files.end());
    auto append = [&](const std::string &value) { cache_key += std::to_string(value.size()) + ":" + value; };
    append(source);
    append(entry);
    append(std::to_string(args.size()));
    for (const auto &arg : args)
      append(arg);
    append(std::to_string(files.size()));
    for (const auto &file : files) {
      append(std::filesystem::relative(file, directory.path).generic_string());
      append(Read(file));
    }
    auto found = std::find_if(cache.begin(), cache.end(), [&](const auto &item) { return item.first == cache_key; });
    if (found != cache.end()) {
      impl_ = found->second;
      auto hit = std::move(*found);
      cache.erase(found);
      cache.push_back(std::move(hit));
      std::cout << "Reused CPU functions " << source << ":" << entry << "\n";
      return;
    }
  }
  for (const auto &file : std::filesystem::recursive_directory_iterator(directory.path)) {
    if (!explicit_context && file.is_regular_file()) {
      auto lowered = LowerLegacySource(Read(file.path()), file.path().filename().string(), optix_shader);
      lowered = RenameLegacyEntry(std::move(lowered), entry, native_entry);
      if (!cuda) {
        auto blocks = LegacyConstantBlocks(lowered);
        constant_blocks.insert(blocks.begin(), blocks.end());
      }
      Write(file.path(), lowered);
    }
  }

  Write(directory.path / "native_input.slang", ImagePrelude() + "\n#include \"" + source + "\"\n");
  RequestOwner owner;
  auto *request = owner.request;
  spSetCodeGenTarget(request, cuda ? SLANG_CUDA_SOURCE : SLANG_SHADER_HOST_CALLABLE);
  if (optix_shader)
    spSetTargetFlags(request, 0, SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM);
  spSetMatrixLayoutMode(request, SLANG_MATRIX_LAYOUT_ROW_MAJOR);
  spSetOptimizationLevel(request, SLANG_OPTIMIZATION_LEVEL_MAXIMAL);
  spSetTargetFloatingPointMode(request, 0, SLANG_FLOATING_POINT_MODE_PRECISE);
  spAddSearchPath(request, directory.path.string().c_str());
  spAddPreprocessorDefine(request, "SPARKIUM_NATIVE", "1");
  if (explicit_context)
    spAddPreprocessorDefine(request, entry.c_str(), native_entry.c_str());
  // NVRTC uses --fmad=false; LLVM receives the precise target mode above.
  spAddPreprocessorDefine(request, "precise", "");
  if (!cuda) {
    // Select the direct Slang IR -> LLVM IR -> in-memory JIT route explicitly.
    // Merely requesting host-callable permits external compiler fallback when
    // slang-llvm is absent, which must never happen for this backend.
    SlangCheck(spSessionCheckPassThroughSupport(Session(), SLANG_PASS_THROUGH_LLVM),
               "native CPU requires the matching slang-llvm library; external compiler fallback is disabled");
    const char *options[] = {"-emit-cpu-via-llvm"};
    SlangCheck(spProcessCommandLineArguments(request, options, 1), "cannot enable embedded LLVM JIT");
    if (std::getenv("SPARKIUM_NATIVE_ASAN"))
      throw std::runtime_error("SPARKIUM_NATIVE_ASAN is unsupported by the embedded LLVM JIT");
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
  spAddEntryPoint(request, unit, native_entry.c_str(), optix_shader ? SLANG_STAGE_RAY_GENERATION : SLANG_STAGE_COMPUTE);
  if (optix_shader) {
    spAddEntryPoint(request, unit, "LongMarchOptixClosest", SLANG_STAGE_CLOSEST_HIT);
    spAddEntryPoint(request, unit, "LongMarchOptixMiss", SLANG_STAGE_MISS);
  }
  // CPU pass 1 only type-checks and reflects resource layout and invocation semantics.
  // It emits no shader executable; pass 2 below compiles ordinary exported functions.
  if (!cuda)
    spSetCompileFlags(request, SLANG_COMPILE_FLAG_NO_CODEGEN);
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
  // The shared tracer records 8x8 compute groups; OptiX launches individual pixels.
  SlangUInt threads[3]{8, 8, 1};
  if (!optix_shader)
    reflection->getEntryPointByIndex(0)->getComputeThreadGroupSize(3, threads);
  for (int i = 0; i < 3; ++i)
    impl_->threads[i] = uint32_t(threads[i]);
  std::string call_arguments;
  if (!cuda) {
    auto *entry_layout = reflection->getEntryPointByIndex(0);
    for (unsigned i = 0; i < entry_layout->getParameterCount(); ++i) {
      auto *parameter = entry_layout->getParameterByIndex(i);
      const char *semantic = parameter->getSemanticName();
      std::string normalized = semantic ? semantic : "";
      std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                     [](unsigned char c) { return char(std::tolower(c)); });
      std::string value;
      if (normalized == "sv_dispatchthreadid")
        value = "dispatch_id";
      else if (normalized == "sv_groupthreadid")
        value = "local_id";
      else if (normalized == "sv_groupid")
        value = "group_id";
      else if (normalized == "sv_groupindex")
        value = "local_index";
      else
        throw std::runtime_error("unsupported ordinary CPU function parameter semantic: " + normalized);
      if (i)
        call_arguments += ",";
      call_arguments += value;
    }
  }
  for (unsigned i = 0; i < reflection->getParameterCount(); ++i) {
    auto *p = reflection->getParameterByIndex(i);
    auto *attribute = p->getVariable()->findUserAttributeByName(Session(), "NativeBinding");
    if (!attribute)
      throw std::runtime_error(std::string("native shader parameter lacks register space: ") + p->getName());
    int slot = -1;
    SlangCheck(attribute->getArgumentValueInt(0, &slot), "invalid native binding attribute");
    if (slot < 0 || slot > (explicit_context ? 63 : 1024))
      throw std::runtime_error("native binding slot out of range");
    auto *type = p->getTypeLayout();
    bool array = type->getType()->getName() && std::string(type->getType()->getName()) == "NativeArray";
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
    if (!cuda && constant_blocks.count(p->getName())) {
      auto *fields = type->getElementTypeLayout();
      for (unsigned j = 0; j < fields->getFieldCount(); ++j) {
        auto *field = fields->getFieldByIndex(j);
        impl_->parameters.back().constants.push_back(
            {field->getName(), field->getOffset(), field->getTypeLayout()->getSize()});
      }
    }
  }
  if (cuda) {
#ifdef SPARKIUM_NATIVE_CUDA_ENABLED
    Slang::ComPtr<ISlangBlob> blob;
    const auto code_status = optix_shader ? spGetTargetCodeBlob(request, 0, blob.writeRef())
                                          : spGetEntryPointCodeBlob(request, 0, 0, blob.writeRef());
    if (SLANG_FAILED(code_status)) {
      const char *diagnostic = spGetDiagnosticOutput(request);
      throw std::runtime_error(std::string("cannot obtain native CUDA source: ") + (diagnostic ? diagnostic : ""));
    }
    std::string code(static_cast<const char *>(blob->getBufferPointer()), blob->getBufferSize());
    if (const char *dump = std::getenv("SPARKIUM_NATIVE_DUMP")) {
      std::filesystem::create_directories(dump);
      Write(std::filesystem::path(dump) / (std::filesystem::path(source).stem().string() + "-" + entry + ".cu"), code);
    }
    impl_->CompileCUDA(code, source, native_entry, optix_shader, optix);
#else
    throw std::runtime_error("native CUDA backend was not built");
#endif
  } else {
    impl_->CompileCPU(directory.path, source, native_entry, call_arguments, args);
  }
  if (explicit_context && !cuda) {
    if (cache.size() >= 16)
      cache.erase(cache.begin());
    cache.emplace_back(std::move(cache_key), impl_);
  }

  std::cout << "Compiled "
            << (optix_shader ? "OptiX kernel "
                : cuda       ? "CUDA kernel "
                             : "CPU functions ")
            << source << ":" << entry << " ("
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s)\n";
}

NativeShader::~NativeShader() = default;

std::string NativeShader::EntryPoint() const {
  return impl_->entry;
}

void NativeShader::Dispatch(const NativeBindings &bindings, uint32_t x, uint32_t y, uint32_t z) {
  if (!x || !y || !z)
    return;
  // Exported resource descriptors are immutable while workers execute this module.
  std::unique_lock<std::mutex> dispatch_lock(impl_->dispatch_mutex, std::defer_lock);
  if (!impl_->cuda && !impl_->explicit_context)
    dispatch_lock.lock();
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
    if (auto it = bindings.acceleration_structures.find(p.slot); it != bindings.acceleration_structures.end()) {
#ifdef SPARKIUM_OPTIX_ENABLED
      auto *as = dynamic_cast<OptixAccelerationStructure *>(it->second);
      if (!as || as->Device() != impl_->optix_device || !as->IsTopLevel())
        throw std::runtime_error("foreign OptiX acceleration structure");
      append(as->Handle());
#else
      NativeUnsupported();
#endif
    } else if (auto it = bindings.buffers.find(p.slot); it != bindings.buffers.end()) {
      for (const auto &range : it->second) {
        auto *b = dynamic_cast<NativeBuffer *>(range.buffer);
        if (!b)
          throw std::runtime_error("foreign native buffer");
        if (range.offset > b->Size() || range.size > b->Size() - range.offset)
          throw std::out_of_range("native buffer binding range");
        void *ptr = static_cast<char *>(b->memory->Data()) + range.offset;
        if (p.kind == SLANG_TYPE_KIND_CONSTANT_BUFFER) {
          if (range.size < p.constant_size)
            throw std::out_of_range("native constant buffer range");
          for (const auto &field : p.constants)
            if (field.offset > range.size || field.size > range.size - field.offset)
              throw std::out_of_range("native CPU constant buffer range");
          append(ptr);
        } else
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
#ifdef SPARKIUM_NATIVE_CUDA_ENABLED
    impl_->DispatchCUDA(globals, x, y, z);
#endif
  } else {
    impl_->DispatchCPU(globals, x, y, z);
  }
}
}  // namespace sparkium::backend
