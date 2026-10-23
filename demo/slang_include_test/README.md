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
