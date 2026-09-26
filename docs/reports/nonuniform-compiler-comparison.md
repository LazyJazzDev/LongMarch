# DXC / Slang：全部受测写法的 NonUniform 编译对比

日期：2026-09-26。范围：HLSL → SPIR-V 的 storage-buffer 描述符非均匀索引。

本报告完整列出本轮 **17 种写法 × 5 个版本 × 2 个优化等级 = 170 个原始模块**，
包括所有成功、失败和常量索引对照，没有筛选案例。17 种源码在各版本和优化等级间
逐字节一致，已通过源码 SHA-256 核对。“全部”指本报告定义的受测集合，不代表穷尽
HLSL 的所有语法组合、资源类型或控制流。

## 判定标准

- **正确**：本样例实际读取 buffers 的 OpLoad 指针带 NonUniform 装饰，且模块声明
  StorageBufferArrayNonUniformIndexing（5308）capability。
- **错：能力**：最终指针标记存在，但缺少上述 capability。
- **错：标记+能力**：两者都缺少。只给整数索引或中间资源指针标记不算正确。
- **正确（对照）**：使用 buffers[0] 常量索引，无需这两项；实测也均未生成这两项。

“正确/错误”仅评价这些非均匀访问要求，不是编译器整体正确性、完整 Vulkan 合法性或
GPU 渲染正确性的证明。非均匀样例采用同一工作组中的 id.x % 2，资源选择可能在子组内
分歧。资源数组来自描述符集，不是同一个 buffer 内的元素索引。

全部 170 次编译命令成功，且全部通过 spirv-val v2026.1 的 Vulkan 1.2 检查。
因此若仅将“编译成功”或“验证器通过”作为判据，会漏掉本报告中的失败情况。
没有运行 LongMarch 修补函数、SPIRV-Cross 或额外优化器；Slang 使用直接 SPIR-V 后端。
本轮没有在手机或其他 GPU 上执行这 170 个模块。

## 版本与总计

| 编号 | 编译器版本 | 精确来源 | 非均匀正确 / 32 | 非均匀错误 / 32 | 常量对照正确 / 2 |
| --- | --- | --- | ---: | ---: | ---: |
| D1 | DXC 本机 1.10(5180-e3554182) | e35541826046479d9787ea0368b274ed2447f913 | 0 | 32 | 2 |
| D2 | DXC 稳定版 v1.9.2607 | 0d3ee6b551b8fa768fbf825300ebab81047ef6a8 | 0 | 32 | 2 |
| D3 | DXC 预览版 v1.10.2605.37 | c4d8f4f99aa221da58cd540bf2099a2889632ab6 | 0 | 32 | 2 |
| S1 | Slang 本机 2026.1-52-gc8ddf20bb | 本机版本字符串 | 0 | 32 | 2 |
| S2 | Slang 正式版 2026.18.3 | 官方 macOS ARM64 发布包 | 28 | 4 | 2 |

D2/D3 从官方 tag 在本机编译，数字 build count 与官方二进制可能不同，以 tag 和完整
提交标识为准。Slang 最新版于 2026-09-25 发布。本轮使用独立路径，没有替换系统工具。
D1/D2/D3 各有 12 个模块仅缺 capability、20 个同时缺标记与 capability。
S1 的 32 个非均匀模块均缺两项；S2 的四个失败模块对应两种调用者标记整数实参的写法。

## 完整结果矩阵

下表保留三个 DXC 版本的独立列，即使结果一致也不合并。禁用优化与 O3 分别展示，
不假设 DXC 的 Od 与 Slang 的 O0 内部 pass 完全等价。

### 禁用优化：DXC -Od / Slang -O0

| 编号 / 写法 | D1 | D2 | D3 | S1 | S2 |
| --- | --- | --- | --- | --- | --- |
| 01 ByteAddressBuffer 直接访问 | 错：能力 | 错：能力 | 错：能力 | 错：标记+能力 | 正确 |
| 02 ByteAddressBuffer 局部资源别名 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 正确 |
| 03 标记后保存整数局部变量 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 正确 |
| 04 缓冲区对象作为函数参数 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 正确 |
| 05 函数返回缓冲区对象 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 正确 |
| 06 辅助函数内部标记整数索引 | 错：能力 | 错：能力 | 错：能力 | 错：标记+能力 | 正确 |
| 07 局部资源别名 + `Load<uint>` | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 正确 |
| 08 StructuredBuffer 直接访问 | 错：能力 | 错：能力 | 错：能力 | 错：标记+能力 | 正确 |
| 09 StructuredBuffer 局部资源别名 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 正确 |
| 10 RWByteAddressBuffer 局部资源别名（读取） | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 正确 |
| 11 仅调用者标记整数实参 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 |
| 12 整数局部变量在使用处再次标记 | 错：能力 | 错：能力 | 错：能力 | 错：标记+能力 | 正确 |
| 13 固定长度数组 + 局部资源别名 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 正确 |
| 14 固定长度数组直接访问 | 错：能力 | 错：能力 | 错：能力 | 错：标记+能力 | 正确 |
| 15 常量索引对照 | 正确（对照） | 正确（对照） | 正确（对照） | 正确（对照） | 正确（对照） |
| 16 noinline + 仅调用者标记整数实参 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 |
| 17 noinline + 函数内部标记索引 | 错：能力 | 错：能力 | 错：能力 | 错：标记+能力 | 正确 |

### 优化：DXC / Slang -O3

| 编号 / 写法 | D1 | D2 | D3 | S1 | S2 |
| --- | --- | --- | --- | --- | --- |
| 01 ByteAddressBuffer 直接访问 | 错：能力 | 错：能力 | 错：能力 | 错：标记+能力 | 正确 |
| 02 ByteAddressBuffer 局部资源别名 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 正确 |
| 03 标记后保存整数局部变量 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 正确 |
| 04 缓冲区对象作为函数参数 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 正确 |
| 05 函数返回缓冲区对象 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 正确 |
| 06 辅助函数内部标记整数索引 | 错：能力 | 错：能力 | 错：能力 | 错：标记+能力 | 正确 |
| 07 局部资源别名 + `Load<uint>` | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 正确 |
| 08 StructuredBuffer 直接访问 | 错：能力 | 错：能力 | 错：能力 | 错：标记+能力 | 正确 |
| 09 StructuredBuffer 局部资源别名 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 正确 |
| 10 RWByteAddressBuffer 局部资源别名（读取） | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 正确 |
| 11 仅调用者标记整数实参 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 |
| 12 整数局部变量在使用处再次标记 | 错：能力 | 错：能力 | 错：能力 | 错：标记+能力 | 正确 |
| 13 固定长度数组 + 局部资源别名 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 正确 |
| 14 固定长度数组直接访问 | 错：能力 | 错：能力 | 错：能力 | 错：标记+能力 | 正确 |
| 15 常量索引对照 | 正确（对照） | 正确（对照） | 正确（对照） | 正确（对照） | 正确（对照） |
| 16 noinline + 仅调用者标记整数实参 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 | 错：标记+能力 |
| 17 noinline + 函数内部标记索引 | 错：能力 | 错：能力 | 错：能力 | 错：标记+能力 | 正确 |

## 全部样例源码

每段均为独立、完整的 compute shader；编号对应上面的两张矩阵。RWByteAddressBuffer
样例测试读取，没有将结果推广到 Store、原子操作或所有 RW 访问。

### 01 ByteAddressBuffer 直接访问

测试 ID：`byte_direct`。

```hlsl
ByteAddressBuffer buffers[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);

[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) { output[id.x] = buffers[NonUniformResourceIndex(id.x%2)].Load(0); }
```

### 02 ByteAddressBuffer 局部资源别名

测试 ID：`byte_local`。

```hlsl
ByteAddressBuffer buffers[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);

[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) { ByteAddressBuffer b = buffers[NonUniformResourceIndex(id.x%2)]; output[id.x] = b.Load(0); }
```

### 03 标记后保存整数局部变量

测试 ID：`byte_index_local`。

```hlsl
ByteAddressBuffer buffers[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);

[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) { uint i=NonUniformResourceIndex(id.x%2); output[id.x] = buffers[i].Load(0); }
```

### 04 缓冲区对象作为函数参数

测试 ID：`byte_param`。

```hlsl
ByteAddressBuffer buffers[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);
uint Read(ByteAddressBuffer b) { return b.Load(0); }
[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) { output[id.x] = Read(buffers[NonUniformResourceIndex(id.x%2)]); }
```

### 05 函数返回缓冲区对象

测试 ID：`byte_return`。

```hlsl
ByteAddressBuffer buffers[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);
ByteAddressBuffer Pick(uint i) { return buffers[NonUniformResourceIndex(i)]; }
[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) { output[id.x] = Pick(id.x%2).Load(0); }
```

### 06 辅助函数内部标记整数索引

测试 ID：`byte_helper_index`。

```hlsl
ByteAddressBuffer buffers[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);
uint Read(uint i) { return buffers[NonUniformResourceIndex(i)].Load(0); }
[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) { output[id.x] = Read(id.x%2); }
```

### 07 局部资源别名 + `Load<uint>`

测试 ID：`byte_templated_local`。

```hlsl
ByteAddressBuffer buffers[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);

[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) { ByteAddressBuffer b = buffers[NonUniformResourceIndex(id.x%2)]; output[id.x] = b.Load<uint>(0); }
```

### 08 StructuredBuffer 直接访问

测试 ID：`structured_direct`。

```hlsl
StructuredBuffer<uint> buffers[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);

[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) { output[id.x] = buffers[NonUniformResourceIndex(id.x%2)][0]; }
```

### 09 StructuredBuffer 局部资源别名

测试 ID：`structured_local`。

```hlsl
StructuredBuffer<uint> buffers[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);

[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) { StructuredBuffer<uint> b = buffers[NonUniformResourceIndex(id.x%2)]; output[id.x] = b[0]; }
```

### 10 RWByteAddressBuffer 局部资源别名（读取）

测试 ID：`rwbyte_local`。

```hlsl
RWByteAddressBuffer buffers[] : register(u0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);

[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) { RWByteAddressBuffer b = buffers[NonUniformResourceIndex(id.x%2)]; output[id.x] = b.Load(0); }
```

### 11 仅调用者标记整数实参

测试 ID：`byte_helper_annotated_arg`。

```hlsl
ByteAddressBuffer buffers[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);
uint Read(uint i) { return buffers[i].Load(0); }
[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) { output[id.x] = Read(NonUniformResourceIndex(id.x%2)); }
```

### 12 整数局部变量在使用处再次标记

测试 ID：`byte_index_rewrapped`。

```hlsl
ByteAddressBuffer buffers[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);

[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) { uint i=NonUniformResourceIndex(id.x%2); output[id.x] = buffers[NonUniformResourceIndex(i)].Load(0); }
```

### 13 固定长度数组 + 局部资源别名

测试 ID：`fixed_byte_local`。

```hlsl
ByteAddressBuffer buffers[2] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);

[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) { ByteAddressBuffer b = buffers[NonUniformResourceIndex(id.x%2)]; output[id.x] = b.Load(0); }
```

### 14 固定长度数组直接访问

测试 ID：`fixed_byte_direct`。

```hlsl
ByteAddressBuffer buffers[2] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);

[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) { output[id.x] = buffers[NonUniformResourceIndex(id.x%2)].Load(0); }
```

### 15 常量索引对照

测试 ID：`byte_uniform`。

```hlsl
ByteAddressBuffer buffers[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);

[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) { ByteAddressBuffer b = buffers[0]; output[id.x] = b.Load(0); }
```

### 16 noinline + 仅调用者标记整数实参

测试 ID：`byte_noinline_arg`。

```hlsl
ByteAddressBuffer buffers[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);
[noinline] uint Read(uint i) { return buffers[i].Load(0); }
[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) { output[id.x] = Read(NonUniformResourceIndex(id.x%2)); }
```

### 17 noinline + 函数内部标记索引

测试 ID：`byte_noinline_inside`。

```hlsl
ByteAddressBuffer buffers[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);
[noinline] uint Read(uint i) { return buffers[NonUniformResourceIndex(i)].Load(0); }
[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) { output[id.x] = Read(id.x%2); }
```

## 复现与机器可读结果

- [完整测试脚本](../../test/graphics/compare_nonuniform_compilers.py)：内含上述全部 17 种源码，
  每次对指定编译器运行两个优化等级，保存命令、诊断、源码、原始 SPIR-V、反汇编和 JSON。
- [170 行完整结果 CSV](nonuniform-compiler-results.csv)：每行是一次编译，包含精确版本
  字符串、优化等级、实际加载指针数、装饰数、capability、编译及验证退出码，以及源码和
  SPIR-V SHA-256。版本编号与上表一致。
- 本机原始证据保存在 out/nonuniform-comparison 下对应的五个版本子目录；这些临时
  二进制没有加入 Git。源码与复现脚本已版本化，不依赖临时目录才能重建测试。

在仓库根目录运行，compiler 参数替换成对应版本的可执行文件路径：

```sh
python3 test/graphics/compare_nonuniform_compilers.py \
  --family dxc --compiler /path/to/dxc --output out/comparison/dxc
python3 test/graphics/compare_nonuniform_compilers.py \
  --family slang --compiler /path/to/slangc --output out/comparison/slang
```

需要 Python 3、spirv-dis 和 spirv-val。实际编译参数：

```sh
dxc -spirv -T cs_6_0 -E Main -fspv-target-env=vulkan1.2 \
  -Od case.hlsl -Fo case.spv
slangc case.hlsl -entry Main -stage compute -target spirv \
  -profile spirv_1_5 -emit-spirv-directly \
  -fvk-t-shift 0 all -fvk-u-shift 0 all -O0 -o case.spv
spirv-val --target-env vulkan1.2 case.spv
```

第二轮将 Od/O0 替换为 O3。脚本沿 buffers 对应的 access-chain 找到实际 OpLoad 指针，
检查 NonUniform（5300）和 capability（5308）。本轮每个模块恰好识别到一个目标指针。
如果未来输出形态导致零个或多个目标指针，脚本会标记 needs_review，不会将其误判为
成功。这是针对该语料的检查器，不是通用 SPIR-V 数据流分析器。

## 解释与工程影响

- DXC 三个版本在全部受测写法上的结论一致；即使源码直接标记最终索引，仍缺少
  storage-buffer capability。局部资源别名等路径还会丢失最终加载指针标记。
- 最新 Slang 已正确处理本轮的直接访问、局部别名、资源对象参数和返回值等路径。
  [上游 #10656](https://github.com/shader-slang/slang/pull/10656) 于 2026-06-09 合入，
  修复资源操作数装饰传播和资源类型 capability 推导，与本轮结果吻合。
- 最新 Slang 的 11、16 号仍失败：仅在调用者标记整数实参，无法保证信息传入函数。
  06、17 号在被调用函数的实际 descriptor 索引处标记则正确。这个差异同时存在于
  两个优化等级及 noinline 对照，不能简单归因于打开或关闭优化。
- [上游 #13089](https://github.com/shader-slang/slang/pull/13089) 修复了整数算术中的
  传播，但不是所有函数边界或数据流传播都已解决的证据。本矩阵未覆盖算术、phi/select、
  循环和嵌套控制流的所有组合，也未覆盖纹理、采样器、DescriptorHandle 或 DXIL。
- 现有 RestoreStorageBufferNonUniform 是有限的 access-chain 修补，不能从完全丢失
  的 NonUniform 信息中恢复程序员意图。不能将本表原始输出结果当作修补后的结果，
  也不能仅凭新版 Slang 的大多数成功案例移除兼容处理或宣布整个项目可直接迁移。

背景、规范依据和修补函数覆盖限制见 [DXC 调查](dxc-nonuniform-investigation.md) 与
[Slang 调查](slang-nonuniform-investigation.md)。此前 Slang 报告的 26/28 是原 14 种
非均匀写法；本轮新增两个 noinline 写法，故对应数字变为 28/32，结果没有矛盾。
