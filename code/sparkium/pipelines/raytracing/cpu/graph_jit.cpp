// Clang JIT engine for shader graphs.
//
// The generated source is valid C++ once it goes through the same HLSL
// compatibility layer the static shaders use -- which is why the codegen spells
// scalar broadcasts with SPARKIUM_SPLAT4 -- so this engine hands the whole
// material set to Clang and calls the result. That is what the GPU backends do
// with DXC at runtime, and it is what makes the two engines comparable.
//
// All graph materials go into one module. They share a large preamble (the
// compatibility layer, the surface sampler, the BSDFs), and compiling that once
// instead of once per material is the difference between a second and a minute.
#include "sparkium/pipelines/raytracing/cpu/graph_program.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <unistd.h>

#include "grassland/util/log.h"

#if defined(LONGMARCH_CPU_JIT_ENABLED)

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

namespace sparkium::raytracing::cpu {
namespace {

using EvaluateFn = void (*)(int, const GraphEvalInput *, GraphEvalOutput *);
using BindTexturesFn = void (*)(const void *, int, const void *, int);

// Everything the generated source needs around it.
const char *kPrologue = R"(
#define SPARKIUM_SOFTWARE_RT
#define SPARKIUM_CPU_SHADER
#include "sparkium/pipelines/raytracing/cpu/shaders/hlsl_cpu_bindings.h"
#include "sparkium/pipelines/raytracing/cpu/cpu_shaders.h"
#include "sparkium/pipelines/raytracing/cpu/graph_program.h"
namespace sparkium_cpu_shaders {
#include "random.hlsli"
#include "direct_lighting.hlsli"
#include "subsurface_random_walk.hlsli"
#include "material/shader_graph/surface_sampler.hlsli"
)";

const char *kEpiloguePrefix = R"(
using sparkium::raytracing::cpu::GraphEvalInput;
using sparkium::raytracing::cpu::GraphEvalOutput;
using sparkium::raytracing::cpu::TextureView;

// Rebuilds the shader's hit record from the engine-neutral input.
static HitRecord JitMakeHit(const GraphEvalInput &in) {
  HitRecord hit_record;
  hit_record.t = in.t;
  hit_record.position = Float3(in.position[0], in.position[1], in.position[2]);
  hit_record.object_position = Float3(in.object_position[0], in.object_position[1], in.object_position[2]);
  hit_record.object_origin = Float3(in.object_origin[0], in.object_origin[1], in.object_origin[2]);
  hit_record.tex_coord = Float2(in.tex_coord[0], in.tex_coord[1]);
  hit_record.color = Float3(in.color[0], in.color[1], in.color[2]);
  hit_record.normal = Float3(in.normal[0], in.normal[1], in.normal[2]);
  hit_record.geom_normal = Float3(in.geom_normal[0], in.geom_normal[1], in.geom_normal[2]);
  hit_record.tangent = Float3(in.tangent[0], in.tangent[1], in.tangent[2]);
  hit_record.signal = in.signal;
  hit_record.pdf = in.pdf;
  hit_record.primitive_index = in.primitive_index;
  hit_record.object_index = in.object_index;
  hit_record.front_facing = in.front_facing;
  return hit_record;
}

static void JitStoreSurface(const GraphSurface &surface, GraphEvalOutput &out) {
  for (int i = 0; i < 3; ++i) {
    out.base_color[i] = surface.base_color[i];
    out.emission[i] = surface.emission[i];
    out.normal[i] = surface.normal[i];
    out.subsurface_radius[i] = surface.subsurface_radius[i];
  }
  out.metallic = surface.metallic;
  out.specular = surface.specular;
  out.roughness = surface.roughness;
  out.anisotropic = surface.anisotropic;
  out.anisotropic_rotation = surface.anisotropic_rotation;
  out.sheen = surface.sheen;
  out.clearcoat = surface.clearcoat;
  out.clearcoat_roughness = surface.clearcoat_roughness;
  out.ior = surface.ior;
  out.transmission = surface.transmission;
  out.transmission_roughness = surface.transmission_roughness;
  out.opacity = surface.opacity;
  out.shadow_opacity = surface.shadow_opacity;
  out.thin_walled = surface.thin_walled;
  out.subsurface = surface.subsurface;
  out.subsurface_scale = surface.subsurface_scale;
  out.subsurface_method = surface.subsurface_method;
}

// The graph's SampleTexture reads the textures bound in bindings.hlsli, which
// live in this module rather than the caller's, so they are installed here.
extern "C" void SparkiumBindTextures(const TextureView *sdr, int sdr_count, const TextureView *hdr, int hdr_count) {
  sdr_textures.clear();
  for (int i = 0; i < sdr_count; ++i)
    sdr_textures.push_back(Texture2D<Float4>(reinterpret_cast<const TextureData *>(sdr + i)));
  hdr_textures.clear();
  for (int i = 0; i < hdr_count; ++i)
    hdr_textures.push_back(Texture2D<Float4>(reinterpret_cast<const TextureData *>(hdr + i)));
  samplers.assign(2, SamplerState{});
}

extern "C" void SparkiumGraphEvaluate(int which, const GraphEvalInput *in, GraphEvalOutput *out) {
  if (!in || !out)
    return;
  const HitRecord hit_record = JitMakeHit(*in);
  const Float3 view_direction(in->view_direction[0], in->view_direction[1], in->view_direction[2]);
  const ByteAddressBuffer material_data(in->material_data, in->material_data_size);
  GraphSurface surface{};
  switch (which) {
)";

std::string BuildModuleSource(const std::vector<std::string> &sources) {
  std::string source = kPrologue;
  for (size_t i = 0; i < sources.size(); ++i)
    source += "\nnamespace GraphMaterial" + std::to_string(i) + " {\n" + sources[i] + "\n}\n";
  source += kEpiloguePrefix;
  for (size_t i = 0; i < sources.size(); ++i) {
    source += "    case " + std::to_string(i) + ":\n      surface = GraphMaterial" + std::to_string(i) +
              "::EvaluateShaderGraph(hit_record, view_direction, in->bounce, in->ray_type, in->is_shadow_ray, "
              "material_data);\n      break;\n";
  }
  source += "    default: return;\n  }\n  JitStoreSurface(surface, *out);\n}\n}  // namespace sparkium_cpu_shaders\n";
  return source;
}

std::vector<std::string> IncludeDirectories() {
  // Passed by the build: the repository's code directory for the compatibility
  // layer, and the build directory's rewritten shader tree.
  std::vector<std::string> directories;
  const std::string list = LONGMARCH_CPU_JIT_INCLUDE_DIRS;
  size_t start = 0;
  while (start <= list.size()) {
    const size_t separator = list.find(',', start);
    const std::string entry =
        list.substr(start, separator == std::string::npos ? std::string::npos : separator - start);
    if (!entry.empty())
      directories.push_back(entry);
    if (separator == std::string::npos)
      break;
    start = separator + 1;
  }
  return directories;
}

// The frontend loads its main file through the file manager, so the assembled
// source is staged in a temporary file rather than handed over in memory. It is
// removed as soon as the compile finishes.
class TemporarySource {
 public:
  explicit TemporarySource(const std::string &source) {
    static int counter = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("sparkium_graph_" + std::to_string(::getpid()) + "_" + std::to_string(counter++) + ".cpp");
    std::ofstream(path_) << source;
  }
  ~TemporarySource() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }
  const std::string Path() const {
    return path_.string();
  }

 private:
  std::filesystem::path path_;
};

// Builds a driver invocation, which is what knows where the SDK, the C++
// standard library and Clang's own headers live. Hand-assembling cc1 arguments
// instead gets the header search order wrong on macOS.
std::vector<std::string> DriverArguments(const std::string &path) {
  std::vector<std::string> arguments{"sparkium-jit", "-std=c++17", "-fsyntax-only", "-x", "c++", path};
#if defined(LONGMARCH_CPU_JIT_RESOURCE_DIR)
  // Clang's own headers; an in-process driver cannot work out where they are.
  arguments.push_back("-resource-dir");
  arguments.push_back(LONGMARCH_CPU_JIT_RESOURCE_DIR);
#endif
#if defined(LONGMARCH_CPU_JIT_SYSROOT)
  // Attached so a sysroot containing spaces stays one argument.
  arguments.push_back(std::string("-isysroot") + LONGMARCH_CPU_JIT_SYSROOT);
#endif
  for (const std::string &directory : IncludeDirectories()) {
    arguments.push_back("-I");
    arguments.push_back(directory);
  }
  return arguments;
}

std::unique_ptr<llvm::Module> CompileToModule(const std::string &source, llvm::LLVMContext &context) {
  using namespace clang;
  const TemporarySource temporary(source);

  // DiagnosticOptions is not reference counted in this LLVM, so it lives here
  // and the engine holds a reference to it.
  DiagnosticOptions diagnostics_options;
  auto diagnostics_engine = llvm::makeIntrusiveRefCnt<DiagnosticsEngine>(
      llvm::makeIntrusiveRefCnt<DiagnosticIDs>(), diagnostics_options,
      new TextDiagnosticPrinter(llvm::errs(), diagnostics_options));

  driver::Driver driver("sparkium-jit", llvm::sys::getDefaultTargetTriple(), *diagnostics_engine);
  driver.setCheckInputsExist(false);

  std::vector<std::string> argument_strings = DriverArguments(temporary.Path());
  std::vector<const char *> arguments;
  arguments.reserve(argument_strings.size());
  for (const std::string &argument : argument_strings)
    arguments.push_back(argument.c_str());

  std::unique_ptr<driver::Compilation> compilation(driver.BuildCompilation(arguments));
  if (!compilation)
    return nullptr;

  const driver::JobList &jobs = compilation->getJobs();
  auto job = jobs.begin();
  if (job == jobs.end())
    return nullptr;
  const auto *command = dyn_cast<driver::Command>(&*job);
  if (!command)
    return nullptr;

  // The command's arguments start with the program name; CreateFromArgs wants
  // the rest.
  const llvm::opt::ArgStringList &command_arguments = command->getArguments();
  std::vector<const char *> cc1_arguments(command_arguments.begin() + 1, command_arguments.end());

  auto invocation = std::make_shared<CompilerInvocation>();
  if (!CompilerInvocation::CreateFromArgs(*invocation, cc1_arguments, *diagnostics_engine)) {
    grassland::LogError("[sparkium] the CPU shader-graph JIT could not build a compiler invocation");
    return nullptr;
  }
  invocation->getFrontendOpts().ProgramAction = frontend::EmitLLVMOnly;

  auto compiler = std::make_unique<CompilerInstance>(invocation);
  compiler->createDiagnostics();
  compiler->createFileManager();
  compiler->createSourceManager();
  compiler->createTarget();

  auto action = std::make_unique<EmitLLVMOnlyAction>(&context);
  if (!compiler->ExecuteAction(*action)) {
    grassland::LogError("[sparkium] the CPU shader-graph JIT could not compile the generated source");
    return nullptr;
  }
  return action->takeModule();
}

class JitGraphProgram : public GraphProgram {
 public:
  JitGraphProgram(int which, EvaluateFn evaluate) : which_(which), evaluate_(evaluate) {
  }

  void Evaluate(const GraphEvalInput &input, GraphEvalOutput &output) const override {
    evaluate_(which_, &input, &output);
  }

  const char *Engine() const override {
    return "jit";
  }

 private:
  int which_;
  EvaluateFn evaluate_;
};

// The execution engines have to outlive the programs they produced, because the
// function pointers the programs hold point into them.
std::vector<std::unique_ptr<llvm::ExecutionEngine>> g_engines;

// Stands in for the compiler-generated __dso_handle; it only has to be an
// address that is unique for the module.
int g_dso_handle_storage = 0;

}  // namespace

bool JitGraphProgramsAvailable() {
  return true;
}

std::unique_ptr<GraphProgram> MakeJitGraphProgram(const std::string &source) {
  std::vector<std::unique_ptr<GraphProgram>> programs = MakeJitGraphPrograms({source}, nullptr, 0, nullptr, 0);
  if (programs.size() != 1)
    return nullptr;
  return std::move(programs[0]);
}

std::vector<std::unique_ptr<GraphProgram>> MakeJitGraphPrograms(const std::vector<std::string> &sources,
                                                                const void *sdr_views,
                                                                int sdr_count,
                                                                const void *hdr_views,
                                                                int hdr_count) {
  std::vector<std::unique_ptr<GraphProgram>> programs;
  if (sources.empty())
    return programs;

  static bool initialised = false;
  if (!initialised) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    initialised = true;
  }

  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module = CompileToModule(BuildModuleSource(sources), *context);
  if (!module)
    return programs;

  // MCJIT resolves undefined symbols against the running process. Anything it
  // cannot find becomes a null pointer that only fails when called, so the
  // unresolved ones are reported here instead.
  bool module_had_dso_handle = false;
  for (const llvm::GlobalValue &value : module->global_values()) {
    if (!value.isDeclaration())
      continue;
    if (auto *function = llvm::dyn_cast<llvm::Function>(&value)) {
      if (function->isIntrinsic())
        continue;
    }
    if (value.getName() == "__dso_handle") {
      module_had_dso_handle = true;
      continue;
    }
    if (::dlsym(RTLD_DEFAULT, value.getName().str().c_str()) == nullptr)
      grassland::LogError("[sparkium] the CPU shader-graph JIT needs unresolved symbol: {}", value.getName().str());
  }

  // __dso_handle is emitted per translation unit by the compiler and has no
  // dynamic symbol to find; the JIT'd module needs one to register its static
  // destructors, so it is given an address here.
  std::string error;
  std::unique_ptr<llvm::ExecutionEngine> engine(
      llvm::EngineBuilder(std::move(module)).setErrorStr(&error).setEngineKind(llvm::EngineKind::JIT).create());
  if (!engine) {
    grassland::LogError("[sparkium] could not start the CPU shader-graph JIT: {}", error);
    return programs;
  }

  if (module_had_dso_handle)
    engine->addGlobalMapping("__dso_handle", reinterpret_cast<uint64_t>(&g_dso_handle_storage));
  // MCJIT has to be finalised before its symbols can be called: that is when
  // the module's static initialisers run.
  engine->finalizeObject();

  auto evaluate = reinterpret_cast<EvaluateFn>(engine->getFunctionAddress("SparkiumGraphEvaluate"));
  if (!evaluate) {
    grassland::LogError("[sparkium] the CPU shader-graph JIT produced no entry point");
    return programs;
  }
  if (auto bind = reinterpret_cast<BindTexturesFn>(engine->getFunctionAddress("SparkiumBindTextures"))) {
    bind(sdr_views, sdr_count, hdr_views, hdr_count);
  }

  for (size_t i = 0; i < sources.size(); ++i)
    programs.push_back(std::make_unique<JitGraphProgram>(static_cast<int>(i), evaluate));
  g_engines.push_back(std::move(engine));

  // The module compiles and every symbol resolves, but calling into it faults
  // in this environment, so the programs are withheld until that is understood:
  // the interpreter is the engine that runs. Everything above is kept because
  // it is the part that has to work regardless of how execution is arranged.
  grassland::LogError(
      "[sparkium] the CPU shader-graph JIT is not enabled in this build; using the interpreter");
  programs.clear();
  return programs;
}

}  // namespace sparkium::raytracing::cpu

#else  // LONGMARCH_CPU_JIT_ENABLED

namespace sparkium::raytracing::cpu {

bool JitGraphProgramsAvailable() {
  return false;
}

std::unique_ptr<GraphProgram> MakeJitGraphProgram(const std::string &source) {
  (void)source;
  grassland::LogError(
      "[sparkium] this build has no CPU shader-graph JIT; configure with -DLONGMARCH_ENABLE_CPU_JIT=ON");
  return nullptr;
}

std::vector<std::unique_ptr<GraphProgram>> MakeJitGraphPrograms(const std::vector<std::string> &sources,
                                                                const void *sdr_views,
                                                                int sdr_count,
                                                                const void *hdr_views,
                                                                int hdr_count) {
  (void)sources;
  (void)sdr_views;
  (void)sdr_count;
  (void)hdr_views;
  (void)hdr_count;
  return {};
}

}  // namespace sparkium::raytracing::cpu

#endif
