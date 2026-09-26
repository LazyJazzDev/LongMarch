# DXC storage-buffer NonUniform 调查

日期：2026-09-26。关联 [graphics 兼容性修复报告](graphics-vulkan-compatibility.md)。

## 结论

目前不能通过升级到已测试的最新官方发布版本来移除
`RestoreStorageBufferNonUniform`。本机 DXC、最新稳定版与最新预览版在本次测试中
存在相同的两类问题：部分访问路径缺少最终资源指针的 `NonUniform` 装饰；即使直接
访问保留了该装饰，也缺少 `StorageBufferArrayNonUniformIndexing` capability。

这针对 **HLSL → SPIR-V 的非均匀 storage-buffer 描述符索引**，不表示普通缓冲区读写
都需要修补，也不将结论推广到 DXIL 或所有资源类型。

Slang 新旧版本的同组测试见 [Slang 对比报告](slang-nonuniform-investigation.md)。

## 实测版本

| 版本 | 来源提交 | 结果 |
| --- | --- | --- |
| 本机 `1.10(5180-e3554182)` | `e35541826046479d9787ea0368b274ed2447f913`，2026-01-23 | 复现 |
| 最新稳定版 `v1.9.2607`，2026-07-29 发布 | `0d3ee6b551b8fa768fbf825300ebab81047ef6a8` | 复现 |
| 最新预览版 `v1.10.2605.37`，2026-08-12 发布 | `c4d8f4f99aa221da58cd540bf2099a2889632ab6` | 复现 |

后两者从官方 tag、对应子模块在 macOS Apple Silicon 上用 Ninja/Release 构建，
启用 `ENABLE_SPIRV_CODEGEN`，未修改编译器源码，也未替换系统 DXC。版本判断以源码
tag/完整提交为准；浅克隆自编译版本的数字 build count 与官方二进制可能不同。

每个版本编译 15 种写法 × `-Od` / `-O3`，共 30 个模块，三套结果一致。
使用 `-spirv -T cs_6_0 -E Main -fspv-target-env=vulkan1.2`，检查的是 **DXC 原始输出**，
未经过 LongMarch、SPIRV-Cross 或我们的修补函数。

## 触发条件

索引使用同一工作组内交替的 `SV_DispatchThreadID.x % 2`，确保它可能在子组内分歧。
下面“指针标记”指实际 `OpLoad` 使用的 storage-buffer 指针，而非仅检查整数索引。

| 写法 | 最终加载指针标记 | StorageBufferArrayNonUniformIndexing |
| --- | --- | --- |
| `buffers[NonUniformResourceIndex(i)].Load(0)` | 有 | 缺失 |
| 在辅助函数内部对索引调用 NonUniformResourceIndex，再直接 Load | 有 | 缺失 |
| `ByteAddressBuffer b = buffers[NonUniformResourceIndex(i)]; b.Load(0)` | 缺失 | 缺失 |
| 先保存 `uint j = NonUniformResourceIndex(i)`，再用 `buffers[j]` | 缺失 | 缺失 |
| 缓冲区对象作为辅助函数参数或返回值 | 缺失 | 缺失 |
| 标记只放在调用者的整数实参上，辅助函数内直接用该形参索引 | 缺失 | 缺失 |
| 保存整数索引后，在实际资源索引处再次调用 NonUniformResourceIndex | 有 | 缺失 |
| `StructuredBuffer<uint>` 直接索引 | 有 | 缺失 |
| `StructuredBuffer<uint>` 或 `RWByteAddressBuffer` 的局部资源别名 | 缺失 | 缺失 |
| ByteAddressBuffer 局部别名后使用 `Load<uint>` | 缺失 | 缺失 |
| 固定长度 `[2]` 数组的直接/局部别名访问 | 分别有/缺失 | 均缺失 |
| 常量 `buffers[0]` 对照 | 无需 | 无需 |

`-Od` 与 `-O3` 的缺失结论一致；关闭优化不是解决办法。
`spirv-val` v2026.1 对这 90 个原始模块全部返回成功，因此“验证器通过”不能替代
对此运行时非均匀访问约束的检查。本次比较不声称这些模块在所有 GPU 上都会出错。

## 最小复现

保存为 `repro.hlsl`：

```hlsl
ByteAddressBuffer buffers[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);

[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) {
  ByteAddressBuffer b = buffers[NonUniformResourceIndex(id.x % 2)];
  output[id.x] = b.Load(0);
}
```

```sh
dxc -spirv -T cs_6_0 -E Main -fspv-target-env=vulkan1.2 -O3 repro.hlsl -Fo repro.spv
spirv-dis repro.spv -o repro.spvasm
spirv-val --target-env vulkan1.2 repro.spv
```

反汇编中索引的 `OpCopyObject` 有 `NonUniform`，最终 `OpLoad` 的指针没有。
将函数体改为直接访问，可保留指针装饰，但仍缺少上述 capability。

Vulkan 的 [Runtime SPIR-V 约束](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/appendices/spirvenv.adoc)
分别要求：未声明 `StorageBufferArrayNonUniformIndexing` 时，storage-buffer 资源选择
必须满足相应动态一致性约束（VUID 10136）；资源选择在相关范围内分歧时，实际资源
操作数需带 `NonUniform`（VUID 10148/10149）。这些不是“给索引标记一下”就自动满足的条件。

## 源码定位与上游记录

- [NonUniformVisitor.cpp](https://github.com/microsoft/DirectXShaderCompiler/blob/c4d8f4f99aa221da58cd540bf2099a2889632ab6/tools/clang/lib/SPIRV/NonUniformVisitor.cpp)
  沿部分 load、access-chain、算术和图像指令传播，但没有通用的局部变量 store/load
  数据流或跨函数实参/形参分析。`-fcgl` 的中间输出中，资源指针先带标记写入局部变量，
  再读出时已失去传播；随后实际成员访问没有标记。问题在驱动消费前已经存在，不能
  简单归因于驱动，亦不能仅归因于 `-O3` 优化。
- [CapabilityVisitor.cpp](https://github.com/microsoft/DirectXShaderCompiler/blob/c4d8f4f99aa221da58cd540bf2099a2889632ab6/tools/clang/lib/SPIRV/CapabilityVisitor.cpp)
  的 `getNonUniformCapability` 覆盖若干纹理、texel buffer 等类型，没有相应
  storage-buffer capability 分支，与原始输出缺少该 capability 的结果一致。
- [PR #2884](https://github.com/microsoft/DirectXShaderCompiler/pull/2884) 于 2020 年合入，
  修复 [#2436](https://github.com/microsoft/DirectXShaderCompiler/issues/2436) 的同类图像
  索引问题并引入/改进传播逻辑；这不是本次 storage-buffer 场景已修复的证据。
- 官方 [稳定版发布](https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.9.2607)
  与 [预览版发布](https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.10.2605.37)
  都包含 SPIR-V 修复，但本次实测仍复现。不能把发布说明中的“SPIR-V fixes”当作具体问题的解决记录。
- 另核对截至调查时的 main `717b24d7a487efb555e976104a263f93f181fd44`：上述两个文件
  与已测试发布版一致。这里只做了 main 的源码比较，没有整编译 main，不能据此宣称
  所有上游开发版本均已实测。

## 对现有 workaround 的影响

现有函数能补上仍可沿 access-chain 找到的 NonUniform 来源及 capability。
本机 30 个模块中，它修改了 26 个：两份均匀索引对照保持不变；另外两份 `-Od` 模块
（整数索引局部变量、标记放在函数实参处）也保持不变，但后两者仍有缺失。
额外的 `[noinline]` 整数参数辅助函数样例也无法被当前函数修复。

因此它是针对现有 shader 形态的局部兼容处理，不是完整的数据流分析器；“幂等且
修补后 spirv-val 通过”不代表可以修复任意丢标记情况。没有将本次调查包装为扩大后的
通用修复，也没有修改运行时算法。

一个已在本机验证的源码层规避方式是：保持在实际 descriptor 索引处直接调用
`NonUniformResourceIndex`，并在入口显式声明
`[[vk::ext_capability(5308)]]`。直接访问样例由此同时具备最终指针装饰和 capability，
且通过 spirv-val。但单独加 capability 不会修复局部别名的指针标记缺失；这个方式仍是
规避方案，也未在整个跨后端 shader 集上替换验证。

建议暂不以“升级 DXC 即可解决”为由移除兼容处理。长期方案应在编译器端补齐传播及
capability 推导，并使用上述原始输出回归判定修复；若改为源码规避，则需逐一核对全部
相关访问点和后端。版本判断必须包含实际 tag/提交，不能只比较 `--version` 中的 1.9/1.10。
