# Slang storage-buffer NonUniform 对比测试

日期：2026-09-26。关联 [DXC 调查](dxc-nonuniform-investigation.md)。

完整的 17 种写法、五个版本、两个优化等级逐项结果与全部源码见
[DXC / Slang 完整对比](nonuniform-compiler-comparison.md)。

## 结论与版本

旧版 Slang 同样存在最终资源指针缺少 `NonUniform` 和缺少
`StorageBufferArrayNonUniformIndexing` capability 的问题。最新正式版已修复本次
大多数样例，但仅在调用者标记整数实参、被调用函数直接用形参索引的写法仍失败。
因此不能直接断言“换成 Slang 就可以删除兼容处理”。

| 实测版本 | 非均匀样例同时具备最终指针标记与 capability | 常量索引对照 |
| --- | --- | --- |
| 本机 `2026.1-52-gc8ddf20bb` | 0/28 | 2/2 正常 |
| 官方 `2026.18.3`，2026-09-25 发布 | 26/28 | 2/2 正常 |

最新版本来自官方 [macOS ARM64 发布包](https://github.com/shader-slang/slang/releases/tag/v2026.18.3)，
独立解压运行，没有替换系统编译器。每版使用与 DXC 调查相同的 15 种 HLSL 写法，
分别以 `-O0`、`-O3` 编译；其中 14 种采用 `id.x % 2` 非均匀索引，一种采用常量索引。
检查原始 SPIR-V，未运行 LongMarch 的修补函数。

## 覆盖范围

最新版在以下写法的两个优化等级下，均正确生成最终加载指针的 NonUniform 装饰及
storage-buffer 非均匀索引 capability：直接访问、局部缓冲区别名、局部整数索引、
缓冲区对象参数和返回值、辅助函数内部标记、`Load<uint>`、StructuredBuffer 直接访问
和局部别名、RWByteAddressBuffer 局部别名、实际访问处重新标记索引，以及固定长度
数组的直接访问和局部别名。

只有“标记仅放在调用者整数实参上”在两个优化等级下仍失败：模块中完全没有
NonUniform 装饰，也没有上述 capability。旧版则在全部 28 个非均匀模块中缺少最终
指针装饰和 capability；部分中间 access-chain 上有标记，但没有传到实际 OpLoad 的指针。

全部 60 个原始模块均通过 `spirv-val` v2026.1 的 Vulkan 1.2 验证，说明验证器通过
不能证明运行时非均匀访问要求已满足。本次是编译输出测试，没有运行手机 GPU 测试，
也没有验证整个应用改用 Slang 后的渲染结果。

## 最新版仍失败的复现

```hlsl
ByteAddressBuffer buffers[] : register(t0, space0);
RWStructuredBuffer<uint> output : register(u0, space1);

uint Read(uint i) {
  return buffers[i].Load(0);
}

[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID) {
  output[id.x] = Read(NonUniformResourceIndex(id.x % 2));
}
```

保存为 `repro.hlsl`，使用 Slang 的直接 SPIR-V 后端：

```sh
slangc repro.hlsl -entry Main -stage compute -target spirv \
  -profile spirv_1_5 -emit-spirv-directly \
  -fvk-t-shift 0 all -fvk-u-shift 0 all -O3 -o repro.spv
spirv-dis repro.spv -o repro.spvasm
spirv-val --target-env vulkan1.2 repro.spv
```

将 `-O3` 换成 `-O0` 仍复现。实际 buffer 加载指针没有 NonUniform，且没有
StorageBufferArrayNonUniformIndexing。这里测试的是整数标记跨函数边界的传播，
不表示传递缓冲区对象的测试也失败。

将 Read 改为以下写法，最新版本即可正确输出两者：

```hlsl
uint Read(uint i) {
  return buffers[NonUniformResourceIndex(i)].Load(0);
}
```

额外在 Read 上添加 `[noinline]`，对失败写法和修正写法各测试 O0/O3，结果一致。
O3 失败样例保留 OpFunctionCall 和整数形参，仍缺少标记；在被调用函数内部标记
则正确。额外四个模块也均通过 spirv-val。没有针对这一问题的编译诊断。

现有 RestoreStorageBufferNonUniform 依赖仍然存在的 NonUniform 来源，无法凭空恢复
已全部丢失的标记；因此这类跨函数整数参数应在实际 descriptor 索引处显式标记。

## 上游修复记录

- [PR #10656](https://github.com/shader-slang/slang/pull/10656)，2026-06-09 合入，
  修复 [#10525](https://github.com/shader-slang/slang/issues/10525)：将 NonUniform
  传播到实际资源操作数，并推导相应资源类型的 capability，包括 storage buffer。
  这与本次新旧版本的结果相符；本次没有逐个发布版本定位首次修复的版本号。
- [PR #13089](https://github.com/shader-slang/slang/pull/13089)，2026-09-17 合入，
  继续修复整数索引算术中的传播。该修复不能被理解为所有数据流和函数边界都已覆盖，
  上面的函数实参样例在 2026.18.3 中仍实测失败。

最新版 Slang 对本项目关心的局部资源别名路径明显优于此次测试的 DXC 版本，但迁移
编译器和删除 workaround 仍需完整 shader 回归及目标 GPU 验证。本次只增加调查报告，
没有更换项目编译器或修改运行时算法。
