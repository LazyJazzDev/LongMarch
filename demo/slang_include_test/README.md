# Slang include preprocessing experiment

使用 Slang 官方 `slangc -E` 展开 include，保留未定义的 `RHI_INPUT`、`RHI_UNIFORM`，随后补上宏定义编译。实验不实现自定义源码解析器，也不改变 RHI 编译流程。

```sh
python3 demo/slang_include_test/run.py
# 指定其他 Slang 安装：
python3 demo/slang_include_test/run.py --slangc /path/to/slangc
```

依赖 Python 3 和支持 SPIR-V / Metal 输出的 slangc。脚本从任意工作目录均可运行；生成的源码和编译产物位于临时目录，结束后自动清理。CMake 不添加编译目标。

## 本机实验结果

macOS，`/usr/local/bin/slangc`，版本 `2026.1-52-gc8ddf20bb`。这是工具版本，未据此推断仓库链接的 Slang SDK 行为。

| 实验 | 结果 |
| --- | --- |
| 多层 include | 展开为完整源码，最终编译无需原始 include 目录 |
| 重复 include | include guard 和 `#pragma once` 均生效，声明只出现一次 |
| 未定义的 RHI 宏 | `RHI_INPUT(...)`、`RHI_UNIFORM(...)` 原样保留 token，空白格式会改变 |
| 延迟定义宏 | 为展开结果添加宏定义后，顶点入口成功编译为 SPIR-V 和 Metal 源码 |
| 条件编译 | `#if defined(RHI_INPUT)` 在第一次预处理时就选定分支，无法在第二次编译时恢复 |
| include 路径宏 | 已定义时可展开；未定义时报错 |
| RHI 宏参数中的其他宏 | 已定义的参数宏仍会展开，例如 `FIELD` 变成 `position`；不是整个调用逐字保留 |
| 源码位置 | 即使指定 `-line-directive-mode standard`，本版本 `-E` 也未输出 `#line`，结果压成一行；后续错误定位到展开文件而非原始 include 文件 |
| 预处理错误状态 | 未定义的 include 路径产生 error 诊断，但本版本 `-E` 退出码为 0，不能仅按退出码判成功 |

脚本用断言验证关键结果，并打印错误诊断和行号行为；这是可复现的探索实验，不是所有 Slang 版本的行为保证。Metal 验证止于生成 Metal 源码，SPIR-V 验证止于生成二进制，不包含 GPU 执行。

## 对 RHI 的意义

该方案能实现“先加载并展开 include，再注入布局宏”。第一阶段必须不定义需要延迟的 RHI 宏；用于选择 include、平台分支及其他条件编译的宏则必须提前确定。已定义的普通宏会在第一阶段展开，定义本身不会作为宏定义保留在输出中。

目前最明显的集成障碍是原始文件行号丢失和 `-E` 的错误退出状态。下一步可以验证链接库的官方预处理接口及诊断 writer；不能直接假设 `loadModuleFromSourceString()` 能返回未经语法分析的预处理文本。此实验尚未验证 SDK 路径、Slang `import` 模块、DXIL 或其他平台。

## RHI 集成

RHI / RHIX 现在提供三个入口：

```cpp
// 系统文件系统；相对 include 相对于源文件所在目录解析。
auto shader = device.CreateShader("shaders/main.slang", "VSMain", SHADER_STAGE_VERTEX);

// 自定义文件系统。VirtualFileSystem 实现 grassland::FileSystem。
auto shader = device.CreateShader("shaders/main.slang", "VSMain", SHADER_STAGE_VERTEX, &files);

// 直接提供源码；默认虚拟文件名为 main.slang。
auto shader = device.CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
```

仍支持 `CreateShader(ShaderDesc)`，原有 `{source, entry_point, stage}` 调用兼容。
`ShaderDesc` 新增 `filename`、`file_system`、`include_paths`、`defines`：

- `source` 非空时使用内存源码，`filename` 指定其诊断路径和相对 include 基准；否则读取 `filename`。
- `file_system == nullptr` 时使用 Slang 的系统文件系统；自定义 FileSystem 的 `ReadFile` 返回 0 表示成功。
- `defines` 是预处理阶段的名称/值对；布局宏由 Program 创建阶段注入，不应在这里定义。
- `GetDesc()` 保留原始参数，包括借用的 FileSystem 指针。该指针只在创建期间使用；调用者必须自行确认其仍然存活，不能通过 GetDesc 延长其生命周期。
- Shader 内部保存独立的预处理结果，FileSystem 可以在 CreateShader 返回后销毁，Program 创建不会再次读取 include。

预处理通过链接的 Slang SDK 的官方 `ICompileRequest`、`-E` 和输出 writer 完成，不启动外部 slangc。该接口已被 Slang 标为 deprecated，目前仍是本实现使用的官方预处理输出路径。Metal 与 Vulkan 使用 SPIR-V 预处理目标，D3D12 使用 DXIL；执行平台宏在预处理阶段提供。

语法、入口点和布局检查仍在 Program 创建时执行；缺失文件、include 错误及 `#error` 则在 Shader 创建时返回空句柄。由于当前 SDK 的预处理请求可能在报错后返回成功，实现额外识别旧式和当前格式的错误诊断头；升级 Slang 后需继续运行错误路径测试。警告不会导致创建失败。

限制仍包括：预处理输出不能保留原始 include 行号；条件分支已经确定；本功能封装 `#include` 源码，不封装 Slang `import` 模块及其编译会话依赖。不要把依赖已销毁自定义文件系统的 import 当作受支持输入。

本机验证：文件系统生命周期、磁盘/内存输入、相对 include、搜索路径和宏定义、预处理失败、警告以及延迟布局宏均覆盖 Metal / Vulkan。两个 hello demo 的 triangle 模块展示嵌入式 VirtualFileSystem 文件加载。D3D12 已接入同一路径，但未在本机执行。

前端默认入口文件名为 `main.slang`；Program 阶段后端合成源码使用 `shader.slang`，包括宏前缀后的 `#line` 和 Slang 模块源码路径，以区分两个阶段的诊断。

## VirtualFileSystem 路径约定

内部使用规范化 UTF-8 路径到文件内容的映射，目录由文件路径隐式表示；公开读写、复制移动及目录导入导出接口保持不变。

- 每个平台都把 `/` 和 `\\` 识别为分隔符，允许混用及重复分隔符。
- 前导分隔符表示 VFS 根目录，`/shaders/main.slang` 与 `shaders/main.slang` 等价。它不表示宿主磁盘根目录或 UNC 网络共享。
- `.`、`..` 交给 `std::filesystem::path::lexically_normal()` 处理，不要求被消去的中间目录存在；规范化后越过虚拟根目录的路径失败。
- 文件路径不能以分隔符、`.` 或 `..` 结尾；空路径、NUL、冒号（包括 Windows 盘符和流语法）无效。VFS 不解释宿主盘符。
- 路径大小写敏感；文件不能覆盖隐式目录，也不能作为其他文件的父目录。无效路径及冲突写入不会建立部分目录；读取失败保持调用者的输出内容不变。
- 虚拟路径按 UTF-8 转换，导入导出使用 `u8path` / `generic_u8string`，避免 Windows 系统代码页参与虚拟文件名转换。

验证入口：`test_virtual_file_system`，以及 `test_rhi` 中的 `FileShadersOwnPreprocessedIncludesAndDeferLayoutMacros` 等 Shader 文件加载测试。反斜杠行为在 macOS 上也实际测试；不等同于已经执行原生 Windows 测试。
