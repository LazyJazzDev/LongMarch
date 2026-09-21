# 普通 HLSL 函数 → LLVM JIT → C++ 函数指针

本实验已在 Windows x64、Release、vcpkg Slang 2026.7.1#1 上验证。
与相邻的 [`hlsl_cpu_jit`](../hlsl_cpu_jit/README.md) 不同，这里不创建 compute kernel，
没有 `[numthreads]`、shader entry point、dispatch range 或全局参数块。

`functions.hlsl` 使用 HLSL 函数体，加上 Slang 扩展的导出标记：

```hlsl
export __extern_cpp float evaluate(float x)
{
    return x * x + 1.0f;
}
```

`export` 保留并导出函数；`__extern_cpp` 保持可查找的原始符号名。
这两个标记是 **Slang 扩展，不是标准 HLSL**。本例仍以
`SLANG_SOURCE_LANGUAGE_HLSL` 加载源文件，由 Slang 解析这些扩展。

C++ 直接查找并调用普通标量函数：

```cpp
using Evaluate = float (*)(float);
auto evaluate = reinterpret_cast<Evaluate>(module->findSymbolAddressByName("evaluate"));
float result = evaluate(2.0f); // 5
```

关键编译配置：

```cpp
spSetCodeGenTarget(request, SLANG_HOST_HOST_CALLABLE);
spSetTargetFlags(request, 0, SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM);
const char *options[] = {"-emit-cpu-via-llvm"};
spProcessCommandLineArguments(request, options, 1);
// 添加源文件，spCompile；不调用 spAddEntryPoint。
spGetTargetHostCallable(request, 0, module.writeRef());
```

`GENERATE_WHOLE_PROGRAM` 在这里表示为整个目标模块生成代码，保留导出的普通函数；
不表示生成 compute kernel。本次试验中，不设置这个标志时，编译步骤可以成功，
但没有 shader entry point 的请求无法取得可调用模块。

在已配置 Slang 依赖的仓库中构建并运行：

```powershell
cmake -S . -B cmake-build-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-ninja --target demo_hlsl_cpu_function_jit
./cmake-build-ninja/demo/hlsl_cpu_function_jit/demo_hlsl_cpu_function_jit.exe
```

预期输出：

```text
evaluate(-2) = 5
evaluate(0) = 1
evaluate(2) = 5
evaluate(3) = 10
multiplyAdd(2, 3, 4) = 10
Verified: ordinary CPU functions, no compute kernel or dispatch wrapper.
```

Windows 实验进程在初始化 Slang 前启用 `NoChildProcessCreation`，禁止创建任何子进程。
成功运行验证了这条内嵌 LLVM JIT 路径不需要外部编译器或链接器子进程。
模块持有 JIT 机器码，必须保持存活直到所有函数调用结束。

本实验验证的是无全局状态的标量 `float` 参数和返回值，以及多个函数从同一模块导出。
不要据此直接将 `float3` 或按值传递的结构体强转为任意 C++ 函数签名：
Slang LLVM 对向量、结构体参数和返回值有特定 ABI，需另行匹配。
这也不测量 JIT 函数调用开销或跨模块内联能力。

参考：[Slang 2026.7.1 LLVM targets / ABI](https://github.com/shader-slang/slang/blob/v2026.7.1/docs/llvm-target.md)。
