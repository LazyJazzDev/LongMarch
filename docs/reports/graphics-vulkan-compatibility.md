# Grassland graphics Vulkan 兼容性修复报告

日期：2026-09-26

## 范围与来源

本分支以 `origin/main` 的 `b87c27c` 为基线，提取鸿蒙开发过程中发现的通用
Vulkan 库层问题，能够独立构建，不依赖 HarmonyOS 应用、iOS 宿主或 Sparkium。
来源为 `2c43df3` 和 `be31b45` 中的 Grassland 改动；主线尚无移动端宿主的条件编译，
因此按主线代码上下文移植，而非整笔 cherry-pick。

修改集中在 `code/grassland/graphics`，以及它直接依赖的 `code/grassland/vulkan`。
原鸿蒙分支及历史保持不变。测试放在 `test/graphics`，本报告随分支提交。

## 解决的问题

| 问题及触发条件 | 修复 | 作用与限制 |
| --- | --- | --- |
| 上传使用专用 transfer queue family，而缓冲区采用 exclusive sharing，缺少跨 family 所有权转移 | 上传命令池和队列改用 graphics family | 避免上传后读取未正确移交的资源；暂不保留专用传输队列的并行收益 |
| 队列提交顺序被误当作内存可见性保证，动态 uniform、instance 数据可能仍被读取为旧值 | 每个 graphics command context 开头加入写入到后续读写的内存屏障 | 解决上传与使用、前后命令上下文之间的可见性遗漏；当前采用保守屏障，后续可细化性能 |
| 资源屏障仅覆盖 `ALL_GRAPHICS`，遗漏 compute；图像上传/读回屏障同样不完整 | 资源、图像初始化及传输边界改为覆盖 `ALL_COMMANDS`，补齐访问掩码 | compute 写入、后续使用及 CPU 读回有明确的同步关系 |
| attachment load/blend 等读取以及 late depth tests 未纳入屏障 | color attachment 加入读访问；depth 同时包含读写和 early/late fragment tests | 防止多渲染批次间的内容、混合及深度依赖遗漏，关联此前 2048 动画闪烁/残影问题 |
| 假定映射内存始终 host coherent | Map 后 invalidate、Unmap 前 flush VMA allocation | 支持非 coherent 内存上的 CPU/GPU 可见性；coherent 内存由 VMA 处理为无需额外操作 |
| DXC 在 ByteAddressBuffer 数组经局部变量/辅助函数访问时，可能只保留索引上的 NonUniform，丢失 storage-buffer 访问指针上的装饰 | 创建 Vulkan shader module 前补全相关访问链的 NonUniform，并补齐所需 capability；启用 storage-buffer array nonuniform indexing feature | 修正分歧描述符访问，关联此前 Cornell Box 局部渲染瑕疵；仅处理 storage-buffer 指针访问链，不泛化修改所有 SPIR-V 指令 |
| `vkQueueSubmit`、fence/transfer 等待和一次性传输的失败被忽略 | 通过现有 `ThrowIfFailed` 传播失败，错误包含数值 VkResult；shader module 创建同样检查返回值 | GPU reset/device lost 不再在这些位置被静默视为成功；本次并未重写所有 Vulkan 错误处理或析构路径 |
| 大型软件追踪单次 compute dispatch 在手机上可能触发 GPU watchdog | 增加 `CommandContext::CmdDispatchBase`，Vulkan 实现使用 `vkCmdDispatchBase`，compute pipeline 设置相应创建标志 | 调用方可拆分工作量并保留全局线程坐标；本分支只提供库 API，不自动拆分所有 dispatch |

`CmdDispatchBase` 的非零 base 当前仅由 Vulkan 实现。其他后端默认接受零 base 并转发
普通 dispatch，对非零 base 显式抛出不支持错误。调用方应检查后端，且不能假设分块后的
`NumWorkgroups` 与整帧 dispatch 相同。

## 刻意保留在应用分支的内容

- HarmonyOS 的全屏、触摸、签名、HDR10/PQ/BT.2020 呈现和 XEngine 能力探测。
- iOS/Metal 宿主、移动窗口输入、离线 shader cache、无 GLFW/headless 构建支持。
- Sparkium 的 128×128 分块调度调用、采样计数更新、shader graph/BSDF 函数拆分。

因此，单独合入本分支不会自动完成 Texture 或 Blender 的全部应用级修复，也不会启用
硬件光追。Blender Junkshop 长时间编译的问题仍未解决，不属于本分支的完成项。

## 独立分支验证

环境：macOS Apple Silicon，Vulkan/MoltenVK，DXC 1.10，SPIR-V Tools。
使用 Ninja、Release 配置；构建同时编译现有 Metal 后端，以检查公共虚函数新增后的兼容性。
测试不依赖场景素材，也不需要安装 HarmonyOS SDK。

```sh
cmake -S . -B out/graphics-compat-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DVCPKG_PATH=/path/to/vcpkg \
  -DLONGMARCH_DISABLE_PYTHON=ON
cmake --build out/graphics-compat-build --target test_vulkan_compatibility -j 6
VK_LAYER_VALIDATE_SYNC=1 \
  out/graphics-compat-build/test/graphics/test_vulkan_compatibility
python3 -m unittest discover -s test/graphics -p 'test_spirv_nonuniform.py' -v
```

- `TiledComputeSeesUploadsAndPreservesGlobalCoordinates`：259×130 图像，跨越多个
  128×128 分块并包含不满块的边缘；连续两次更新动态参数，在 GPU 上累加，再逐像素
  验证坐标、参数值和前一轮结果。覆盖普通图像上传、compute 读写、分块 base 和读回。
- `DivergentStorageBuffersThroughHelperProduceCorrectValues`：同一工作组中交替访问
  两个 storage buffer，经辅助函数和局部 ByteAddressBuffer 获取数据；逐项验证 GPU
  结果，并覆盖静态上传、普通 dispatch 与缓冲读回。
- Python 编译回归：直接/局部缓冲访问 × 均匀/分歧索引四种组合；验证修复后 SPIR-V
  通过 `spirv-val`、需要的装饰/capability 存在、重复修复幂等、均匀访问字节不变。

本次结果：2 项 GPU 测试及 1 项 Python 回归（含 4 种编译组合）全部通过，
Vulkan 同步验证未报告错误。非 coherent 内存、多个 queue family 和真实设备丢失
不能由单台 MoltenVK 主机完整覆盖；D3D12 本次未构建或运行。构建出现本机 vcpkg 静态库
使用较新 macOS deployment target 的链接警告，未阻止链接和测试运行。

## 此前设备证据（不是本独立分支的新增真机验收）

鸿蒙集成分支在 MLN-AL00 / Maleoon 935F / 6.1.0.135 上定位到完整分辨率 Texture
出现 GPU reset/device lost；配合 Sparkium 分块调用后已出图。此前也验证了 2048/GoL、
Cornell、HDR，以及低采样 Monster/Classroom。独立分支提取了相关通用库修复，没有复制
应用呈现代码，也没有将这些集成验证夸大为所有 Vulkan 驱动、全部场景的覆盖。
