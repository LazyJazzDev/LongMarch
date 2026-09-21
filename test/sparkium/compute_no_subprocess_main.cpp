#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <gtest/gtest.h>
#include <windows.h>

#include <iostream>

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  // This executable exercises the CPU suite with OS-enforced subprocess
  // denial. Compiler discovery through the registry/PATH cannot bypass it.
  GTEST_FLAG_SET(filter, "*/CPU");
  PROCESS_MITIGATION_CHILD_PROCESS_POLICY policy{};
  policy.NoChildProcessCreation = 1;
  if (!SetProcessMitigationPolicy(ProcessChildProcessPolicy, &policy, sizeof(policy))) {
    std::cerr << "Cannot restrict child processes: " << GetLastError() << '\n';
    return 1;
  }

  PROCESS_MITIGATION_CHILD_PROCESS_POLICY actual{};
  if (!GetProcessMitigationPolicy(GetCurrentProcess(), ProcessChildProcessPolicy, &actual, sizeof(actual)) ||
      !actual.NoChildProcessCreation) {
    std::cerr << "Child process restriction is not active\n";
    return 1;
  }

  std::cout << "Child process creation is disabled by Windows\n";
  return RUN_ALL_TESTS();
}
