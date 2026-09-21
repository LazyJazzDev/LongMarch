# HLSL → LLVM JIT → CPU

最小的 CPU compute 示例，只依赖 Slang，不使用 LongMarch 渲染器、GPU 或图形 API。

若只需导出普通标量计算函数，参见相邻的
[`hlsl_cpu_function_jit`](../hlsl_cpu_function_jit/README.md) 实验，无需 compute kernel。

- `kernel.hlsl`：对数组中每个元素计算 `x * 2 + 1`，使用标准 HLSL 的
  `RWStructuredBuffer<float>` 和 `SV_DispatchThreadID`。
- `main.cpp`：加载 HLSL、在进程内编译、取得函数指针，然后把 C++ 数组传给它执行。
- `CMakeLists.txt`：链接项目 vcpkg 提供的 Slang，并部署匹配的 LLVM 插件。

在已配置好的仓库构建目录中执行：

```powershell
cmake -S . -B cmake-build-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-ninja --target demo_hlsl_cpu_jit
./cmake-build-ninja/demo/hlsl_cpu_jit/demo_hlsl_cpu_jit.exe
```

Linux/macOS 运行时省略 `.exe`。需要项目的 `shader-slang` 包及其 LLVM 插件；
`LONGMARCH_ENABLE_COMPUTE_RENDER` 保持默认的 `ON`，以启用根目录的 Slang 包发现。
默认读取本示例目录中的 `kernel.hlsl`，也可以传入另一个遵守同样接口的 HLSL 路径。

预期输出：

```text
CPU result: 1, 3, 5, 7
Verified: HLSL executed through embedded LLVM JIT.
```

核心流程如下：

```text
kernel.hlsl
  → spCompile（Slang IR → LLVM IR → JIT 机器码）
  → spGetEntryPointHostCallable（进程内模块）
  → findSymbolAddressByName("computeMain")（CPU 函数指针）
  → kernel(&range, nullptr, &globals)
  → C++ 数组变为 {1, 3, 5, 7}
```

`SLANG_SHADER_HOST_CALLABLE` 选择可调用的 CPU 模块，`-emit-cpu-via-llvm`
显式选择内嵌 LLVM 路径。程序先检查 LLVM 插件是否可用；缺失时直接报错。
运行时不会先生成 C++ 再调用外部编译器；构建这个 C++ 宿主程序仍然需要常规 C++ 编译器。

示例中的 `Globals` 遵循 Slang CPU ABI：structured buffer 表示为数据指针和
`size_t` 元素数量。代码用反射检查参数大小和偏移。`DispatchRange` 表示工作组的
起点和终点（不含终点）；这里是 4 个工作组，每组 1 个 invocation，依次修改 4 个元素。
入口参数块为空，所以第二个实参是 `nullptr`。

这是单 CPU 线程的调用，不创建线程池。JIT 模块必须在函数调用结束后才能释放。
该示例演示当前 Slang CPU ABI 和直接 LLVM 编译路径，不表示任意 GPU HLSL 功能都受支持；
直接 LLVM emitter 仍是 Slang 的实验性功能。
