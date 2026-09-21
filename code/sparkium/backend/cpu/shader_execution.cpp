#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>

#include "sparkium/backend/cpu/cpu_shader_internal.h"
#include "sparkium/backend/cpu/cpu_thread_pool.h"

namespace sparkium::backend::cpu {
void CpuShader::Impl::CompileCPU(const std::filesystem::path &directory,
                                 const std::string &source,
                                 const std::string &compute_entry,
                                 const std::string &call_arguments,
                                 const std::vector<std::string> &args) {
  std::ostringstream wrapper;
  wrapper << Read(directory / "compute_input.slang") << "\nexport __extern_cpp void " << compute_entry << "_Run("
          << "ComputeContext* compute_context, "
          << "uint64_t begin, uint64_t end, uint nx, uint ny) {\n"
          << "for(uint64_t group=begin;group<end;++group) {\n"
          << "uint3 group_id=uint3(uint(group%nx),uint((group/nx)%ny),uint(group/(uint64_t(nx)*ny)));\n"
          << "for(uint z=0;z<" << threads[2] << ";++z) for(uint y=0;y<" << threads[1] << ";++y) for(uint x=0;x<"
          << threads[0] << ";++x) {\n"
          << "uint3 local_id=uint3(x,y,z); uint3 dispatch_id=group_id*uint3(" << threads[0] << "," << threads[1] << ","
          << threads[2] << ")+local_id;\n"
          << "uint local_index=(z*" << threads[1] << "+y)*" << threads[0] << "+x;\n"
          << compute_entry << "(" << (call_arguments.empty() ? "compute_context" : "compute_context,") << call_arguments
          << "); } } }\n";
  {
    wrapper << "\nexport __extern_cpp uint64_t " << compute_entry
            << "_ABI() { return "
               "uint64_t(sizeof(ComputeContext)) | (uint64_t(sizeof(ComputeSamplerState)) << 32); }\n";
  }
  Write(directory / "compute_input.slang", wrapper.str());
  RequestOwner host_owner;
  auto *host_request = host_owner.request;
  spSetCodeGenTarget(host_request, SLANG_HOST_HOST_CALLABLE);
  spSetTargetFlags(host_request, 0, SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM);
  spSetMatrixLayoutMode(host_request, SLANG_MATRIX_LAYOUT_ROW_MAJOR);
  spSetOptimizationLevel(host_request, SLANG_OPTIMIZATION_LEVEL_MAXIMAL);
  spSetTargetFloatingPointMode(host_request, 0, SLANG_FLOATING_POINT_MODE_PRECISE);
  const char *options[] = {"-emit-cpu-via-llvm"};
  SlangCheck(spProcessCommandLineArguments(host_request, options, 1), "cannot enable function LLVM JIT");
  spAddSearchPath(host_request, directory.string().c_str());
  spAddPreprocessorDefine(host_request, "SPARKIUM_COMPUTE", "1");
  {
    spAddPreprocessorDefine(host_request, "SPARKIUM_CPU_FUNCTIONS", "1");
    spAddPreprocessorDefine(host_request, entry.c_str(), compute_entry.c_str());
  }
  spAddPreprocessorDefine(host_request, "SPARKIUM_CPU", "1");
  spAddPreprocessorDefine(host_request, "precise", "");
  for (const auto &arg : args) {
    if (arg.rfind("-D", 0) != 0)
      continue;
    auto equal = arg.find('=');
    auto key = arg.substr(2, equal == std::string::npos ? equal : equal - 2);
    auto value = equal == std::string::npos ? "1" : arg.substr(equal + 1);
    spAddPreprocessorDefine(host_request, key.c_str(), value.c_str());
  }
  int host_unit = spAddTranslationUnit(host_request, SLANG_SOURCE_LANGUAGE_SLANG, "compute_functions");
  spAddTranslationUnitSourceFile(host_request, host_unit, (directory / "compute_input.slang").string().c_str());
  if (const char *dump = std::getenv("SPARKIUM_COMPUTE_DUMP")) {
    const auto destination = std::filesystem::path(dump) / compute_entry;
    std::filesystem::create_directories(destination);
    std::filesystem::copy(directory, destination,
                          std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);
    spSetDumpIntermediates(host_request, 1);
    spSetDumpIntermediatePrefix(host_request, (destination / "host-").string().c_str());
  }
  auto host_status = spCompile(host_request);
  if (SLANG_FAILED(host_status))
    throw std::runtime_error("Slang CPU functions " + source + ":" + entry + "\n" +
                             spGetDiagnosticOutput(host_request));
  SlangCheck(spGetTargetHostCallable(host_request, 0, library.writeRef()), "cannot load ordinary CPU functions");
  context_host = reinterpret_cast<ContextFunction>(library->findSymbolAddressByName((compute_entry + "_Run").c_str()));
  if (!context_host)
    throw std::runtime_error("ordinary CPU function missing");
  {
    auto abi = reinterpret_cast<uint64_t (*)()>(library->findSymbolAddressByName((compute_entry + "_ABI").c_str()));
    // Slang sizeof(resource) is a logical HLSL size (zero), not its compute
    // descriptor size. Descriptor layouts are checked against reflection when
    // binding; only ordinary pointer/scalar structures use sizeof here.
    const uint64_t expected = sizeof(ComputeContext) | (uint64_t(12) << 32);
    if (!abi || abi() != expected)
      throw std::runtime_error("CPU context/descriptor ABI mismatch: " + std::to_string(abi ? abi() : 0) +
                               " expected " + std::to_string(expected));
  }
}

void CpuShader::Impl::DispatchCPU(const std::vector<uint8_t> &globals, uint32_t x, uint32_t y, uint32_t z) {
  const uint64_t plane = uint64_t(x) * y;
  if (plane > std::numeric_limits<uint64_t>::max() / z)
    throw std::overflow_error("compute CPU dispatch grid too large");
  const uint64_t groups = plane * z;
  ComputeContext context;
  for (const auto &p : parameters) {
    if (p.size > 32)
      throw std::runtime_error("CPU descriptor exceeds context slot");
    std::memcpy(context.slots + size_t(p.slot) * 4, globals.data() + p.offset, p.size);
  }
  CpuThreadPool::Shared().Run(groups, [&](uint64_t begin, uint64_t end) { context_host(&context, begin, end, x, y); });
}
}  // namespace sparkium::backend::cpu
