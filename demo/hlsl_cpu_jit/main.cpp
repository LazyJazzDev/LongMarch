#include <slang-com-ptr.h>
#include <slang.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
void Check(SlangResult result, const char *message) {
  if (SLANG_FAILED(result))
    throw std::runtime_error(message);
}

// Slang's CPU ABI: a structured buffer is a pointer followed by an element count.
struct Globals {
  float *data;
  size_t count;
};

// The exported compute entry executes [start, end) workgroups on the calling CPU thread.
struct DispatchRange {
  uint32_t start[3];
  uint32_t end[3];
};

using CpuKernel = void (*)(DispatchRange *, void *entryPointParams, void *globalParams);
}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc > 2)
      throw std::runtime_error("Usage: demo_hlsl_cpu_jit [kernel.hlsl]");
    const char *source = argc == 2 ? argv[1] : DEMO_HLSL_PATH;
    std::unique_ptr<SlangSession, decltype(&spDestroySession)> session(spCreateSession(), spDestroySession);
    if (!session)
      throw std::runtime_error("Cannot create Slang session");

    // Require the embedded compiler. Never fall back to launching a C++ compiler.
    Check(spSessionCheckPassThroughSupport(session.get(), SLANG_PASS_THROUGH_LLVM),
          "The matching slang-llvm shared library is required");
    std::unique_ptr<SlangCompileRequest, decltype(&spDestroyCompileRequest)> request(
        spCreateCompileRequest(session.get()), spDestroyCompileRequest);
    if (!request)
      throw std::runtime_error("Cannot create Slang compile request");

    // HLSL -> Slang IR -> LLVM IR -> in-process JIT machine code.
    spSetCodeGenTarget(request.get(), SLANG_SHADER_HOST_CALLABLE);
    const char *options[] = {"-emit-cpu-via-llvm"};
    Check(spProcessCommandLineArguments(request.get(), options, 1), "Cannot enable LLVM JIT");
    spSetOptimizationLevel(request.get(), SLANG_OPTIMIZATION_LEVEL_MAXIMAL);
    int unit = spAddTranslationUnit(request.get(), SLANG_SOURCE_LANGUAGE_HLSL, "example");
    spAddTranslationUnitSourceFile(request.get(), unit, source);
    spAddEntryPoint(request.get(), unit, "computeMain", SLANG_STAGE_COMPUTE);
    const auto status = spCompile(request.get());
    Check(status, spGetDiagnosticOutput(request.get()));

    // Check the one-buffer layout before passing C++ memory across the ABI.
    auto *layout = reinterpret_cast<slang::ShaderReflection *>(spGetReflection(request.get()));
    if (layout->getParameterCount() != 1 || layout->getParameterByIndex(0)->getOffset() != 0 ||
        layout->getParameterByIndex(0)->getTypeLayout()->getSize() != sizeof(Globals))
      throw std::runtime_error("Expected one CPU structured-buffer parameter");
    SlangUInt groupSize[3];
    layout->getEntryPointByIndex(0)->getComputeThreadGroupSize(3, groupSize);
    if (groupSize[0] != 1 || groupSize[1] != 1 || groupSize[2] != 1)
      throw std::runtime_error("This example requires numthreads(1, 1, 1)");

    Slang::ComPtr<ISlangSharedLibrary> module;
    Check(spGetEntryPointHostCallable(request.get(), 0, 0, module.writeRef()), "Cannot obtain JIT module");
    auto kernel = reinterpret_cast<CpuKernel>(module->findSymbolAddressByName("computeMain"));
    if (!kernel)
      throw std::runtime_error("CPU entry point is missing");

    float values[] = {0, 1, 2, 3};
    Globals globals{values, 4};
    DispatchRange range{{0, 0, 0}, {4, 1, 1}};
    kernel(&range, nullptr, &globals);  // Ordinary function call; no GPU or graphics API.

    for (size_t i = 0; i < 4; ++i) {
      if (values[i] != float(i * 2 + 1))
        throw std::runtime_error("Unexpected CPU result");
      std::cout << (i ? ", " : "CPU result: ") << values[i];
    }
    std::cout << "\nVerified: HLSL executed through embedded LLVM JIT.\n";
    // Keep module alive until all calls through kernel have finished.
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hlsl_cpu_jit: " << error.what() << '\n';
    return 1;
  }
}
