#include <slang-com-ptr.h>
#include <slang.h>

#include <iostream>
#include <memory>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace {
void Check(SlangResult result, const char *message) {
  if (SLANG_FAILED(result))
    throw std::runtime_error(message ? message : "Slang operation failed");
}
}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc > 2)
      throw std::runtime_error("Usage: demo_hlsl_cpu_function_jit [functions.hlsl]");
#ifdef _WIN32
    // Experiment safeguard: prove compilation works with all child processes prohibited.
    PROCESS_MITIGATION_CHILD_PROCESS_POLICY policy{};
    policy.NoChildProcessCreation = 1;
    if (!SetProcessMitigationPolicy(ProcessChildProcessPolicy, &policy, sizeof(policy)))
      throw std::runtime_error("Cannot prohibit child processes");
#endif
    std::unique_ptr<SlangSession, decltype(&spDestroySession)> session(spCreateSession(), spDestroySession);
    if (!session)
      throw std::runtime_error("Cannot create Slang session");
    Check(spSessionCheckPassThroughSupport(session.get(), SLANG_PASS_THROUGH_LLVM),
          "The matching slang-llvm shared library is required");
    std::unique_ptr<SlangCompileRequest, decltype(&spDestroyCompileRequest)> request(
        spCreateCompileRequest(session.get()), spDestroyCompileRequest);
    if (!request)
      throw std::runtime_error("Cannot create Slang compile request");

    // HOST_HOST_CALLABLE exports ordinary functions, not compute dispatch entry points.
    spSetCodeGenTarget(request.get(), SLANG_HOST_HOST_CALLABLE);
    // With no shader entry points, explicitly request code for the entire exported module.
    spSetTargetFlags(request.get(), 0, SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM);
    const char *options[] = {"-emit-cpu-via-llvm"};
    Check(spProcessCommandLineArguments(request.get(), options, 1), "Cannot enable LLVM JIT");
    spSetOptimizationLevel(request.get(), SLANG_OPTIMIZATION_LEVEL_MAXIMAL);
    int unit = spAddTranslationUnit(request.get(), SLANG_SOURCE_LANGUAGE_HLSL, "functions");
    spAddTranslationUnitSourceFile(request.get(), unit, argc == 2 ? argv[1] : DEMO_HLSL_PATH);
    // No spAddEntryPoint, shader stage, numthreads, SV_DispatchThreadID or global buffers.
    const auto status = spCompile(request.get());
    Check(status, spGetDiagnosticOutput(request.get()));

    Slang::ComPtr<ISlangSharedLibrary> module;
    const auto moduleStatus = spGetTargetHostCallable(request.get(), 0, module.writeRef());
    if (SLANG_FAILED(moduleStatus)) {
      if (const char *diagnostics = spGetDiagnosticOutput(request.get()))
        std::cerr << diagnostics;
      Check(moduleStatus, "Cannot obtain host JIT module");
    }
    using Evaluate = float (*)(float);
    using MultiplyAdd = float (*)(float, float, float);
    auto evaluate = reinterpret_cast<Evaluate>(module->findSymbolAddressByName("evaluate"));
    auto multiplyAdd = reinterpret_cast<MultiplyAdd>(module->findSymbolAddressByName("multiplyAdd"));
    if (!evaluate || !multiplyAdd)
      throw std::runtime_error("Exported function is missing");

    for (float x : {-2.0f, 0.0f, 2.0f, 3.0f}) {
      const float result = evaluate(x);
      if (result != x * x + 1.0f)
        throw std::runtime_error("Unexpected evaluate result");
      std::cout << "evaluate(" << x << ") = " << result << '\n';
    }
    const float result = multiplyAdd(2.0f, 3.0f, 4.0f);
    if (result != 10.0f)
      throw std::runtime_error("Unexpected multiplyAdd result");
    std::cout << "multiplyAdd(2, 3, 4) = " << result << '\n';
    std::cout << "Verified: ordinary CPU functions, no compute kernel or dispatch wrapper.\n";
    // module owns the executable memory; keep it alive throughout all calls.
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hlsl_cpu_function_jit: " << error.what() << '\n';
    return 1;
  }
}
