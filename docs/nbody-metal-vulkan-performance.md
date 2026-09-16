# nbody_cs：Metal 与 Vulkan 性能对比

测试日期：2026-09-16。机器：Apple M5 MacBook Pro，32 GB 内存，macOS 26.6.2（25G83），接入电源。使用 `rt-fallback` 工作区的 Release 双后端构建，Vulkan SDK 1.4.341.1，Apple Clang；Vulkan 在此机器上通过 MoltenVK 执行，最终也使用 Metal GPU。

当前 Metal 已默认开启 fast math；最新测量见文末「追加实验」。以下基线保留开启前的数据，便于对照。

## 原始基线结论（Metal fast math 关闭）

初始 Metal 关闭 fast math 的设置下，65,536 粒子的完整离屏帧：Metal **56.46 ms**，Vulkan **34.09 ms**，Vulkan 每帧耗时少约 **40%**，等效吞吐量为 **1.66 倍**。按同步帧延迟换算约 **17.7 / 29.3 FPS**，不是窗口实际显示帧率。

主要差距来自浮点编译设置。基线版本 Metal 的 `metal_shader.cpp` 明确设置 `setFastMathEnabled(false)`；读取当前 MoltenVK 配置得到 `fastMathEnabled=2`（on demand）。将 Vulkan 设置为 `MVK_CONFIG_FAST_MATH_ENABLED=0` 后，纯计算 Metal / Vulkan 为 **54.22 / 54.06 ms**，差距仅 **0.3%**。这不能解读为 Vulkan API 天生比 Metal 快。

基线测试没有修改着色器算法或默认浮点设置，仅验证了关闭 Vulkan fast math 的效果。后续开启原生 Metal fast math 的测量见文末追加实验。

## 测量方法

- 同一可执行文件选择 `--backend metal|vulkan`；关闭 API / shader validation。
- 固定随机种子 1，默认 65,536 粒子，128 线程一组，共享内存分块，O(N²) 全粒子两两计算。保留原始步长 0.03、重力常数 `100/65536`、10 个星系。
- 每次预热 20 帧，再测量 80 帧；每组 3 次，后端执行顺序交替。表中为三次运行均值的中位数。
- 每帧等待 GPU 完成；统计的是同步帧延迟，包括更新、提交和等待，不是多帧流水并行的峰值吞吐。
- `compute` 包含模拟 dispatch 和新位置回拷；`offscreen` 还包含清屏、粒子加法混合光栅化、原始无附件 HDR 后处理 pass。HDR 开关默认关闭，但 pass 仍执行。固定 1920×1080 RGBA32F，无窗口、ImGui 或显示同步。
- 初始化、shader 编译、预热、CSV 写入和最终状态读回不计入测量。每次运行独占本次测试的 GPU 工作，没有并行启动 benchmark。
- Metal GPU 时间来自完成后的 command buffer `GPUStartTime/GPUEndTime`；Vulkan 来自帧命令首尾 timestamp query。边界不完全相同，单独的 uniform 上传不包含在 Vulkan 帧 GPU query 内；端到端 wall time 是主要比较指标。
- CSV 的 `record_ms/submit_ms/wait_ms` 是 CPU 所处调用阶段，不是互斥的 GPU 阶段。MoltenVK 当前 `synchronousQueueSubmits=1`，GPU 等待大多计入 submit，而 Metal 计入 wait；不能把 Vulkan 的 submit 时间当成纯 CPU 开销。

## 原始默认设置结果

时间单位 ms；后端列为 **wall / GPU**；末列为 Metal / Vulkan 的 wall 比值。

| 模式 | 粒子数 | Metal | Vulkan | 耗时比 |
| --- | ---: | ---: | ---: | ---: |
| compute | 16,384 | 3.97 / 3.74 | 2.21 / 1.94 | 1.80× |
| compute | 32,768 | 14.47 / 14.23 | 8.05 / 7.68 | 1.80× |
| compute | 65,536 | 56.19 / 55.93 | 30.65 / 30.33 | 1.83× |
| offscreen | 16,384 | 4.27 / 4.02 | 5.18 / 4.86 | 0.82× |
| offscreen | 32,768 | 14.98 / 14.68 | 11.01 / 10.67 | 1.36× |
| offscreen | 65,536 | 56.46 / 56.15 | 34.09 / 33.70 | 1.66× |

粒子数翻倍时，纯计算 GPU 时间接近四倍，符合 O(N²) 算法。65,536 粒子时计算是主要瓶颈。16,384 粒子时，Metal 完整离屏帧反而少耗时约 18%，说明不能把大规模计算的结论泛化到所有规模。

对比两种模式的 GPU 均值，Metal 增加光栅化等工作约 0.2–0.5 ms，Vulkan 约 2.9–3.4 ms。这是**分别运行两种模式的差值估计**，不是同一帧的分阶段 GPU timestamp；不足以进一步断言某个具体光栅 pass 占用了这些时间。

部分长测出现抖动：默认 65,536 粒子 compute 的 Metal 三次均值为 56.19 / 55.17 / 64.69 ms；offscreen 的 Vulkan 为 34.09 / 33.97 / 38.82 ms。保留全部原始数据，使用中位数减少异常运行影响。

## 关闭 Vulkan fast math 的对照

保持其他参数不变，仍为三轮，每轮 20 帧预热、80 帧计时；环境变量只影响 MoltenVK，Metal 同时重测作为对照。

| 模式，65,536 粒子 | Metal wall / GPU（ms） | Vulkan wall / GPU（ms） |
| --- | ---: | ---: |
| compute | 54.22 / 53.94 | 54.06 / 53.73 |
| offscreen | 60.56 / 60.24 | 58.83 / 58.43 |

纯计算结果稳定，两边基本持平，足以解释主要差距。严格浮点的 offscreen 组抖动较大（Metal 三轮 55.05–67.42 ms，Vulkan 57.27–65.02 ms），不用于判断几个百分点的胜负。未固定 GPU 频率，也未记录温度，无法把这些波动归因于某一种原因。

## 窗口与计时开销检查

另测 65,536 粒子、每轮预热 20 帧、计时 60 帧、两轮交替运行：

| 检查 | Metal wall（ms） | Vulkan wall（ms） |
| --- | ---: | ---: |
| 离屏，关闭 GPU 计时 | 55.82 | 33.92 |
| 窗口，包含 ImGui 与呈现 | 56.19 | 34.50 |

关闭计时与主测试差异约 1.1% / 0.5%，远小于两后端的性能差距；这不是精确的计时开销估计，因为是分别执行的运行。

窗口参数为请求 1920×1080，保留当前后端呈现策略，未强制统一 VSync 或 Retina drawable 尺寸。窗口结果同样逐帧等待 GPU，约合 17.8 / 29.0 帧/秒，仅作为实际窗口路径的补充验证；受显示环境影响，主要结论仍采用固定分辨率离屏数据。benchmark 模式禁用 ImGui ini，避免读取用户布局或保存测试布局。

复现时在脚本后附加 `--particles 65536 --modes offscreen --frames 60 --repeats 2 --no-gpu-timing --output out/nbody-profile/no-timestamps`，或 `--particles 65536 --modes window --frames 60 --repeats 2 --output out/nbody-profile/window`。

## 正确性与兼容性

4,096 粒子、种子 1、单步更新、无预热的状态读回：两边所有数值有限，位置逐位相同，速度最大绝对差 `1.1920928955078125e-7`，RMSE `3.971592373124724e-9`。每个后端的 compute / offscreen 状态完全相同。这是单步检查，不代表长时间混沌演化也逐位一致。

为了让原始 nbody 帧在 Metal 上完整执行，补齐无颜色/深度附件的 raster pass 支持，并让 viewport/scissor 在同一 command context 的多个 pass 之间持续生效。新增回归检查使用 5×4 图像，验证第二个无附件 pass 只修改继承 scissor 范围内的 6 个像素。

Release 双后端和 Metal-only 的 `demo_nbody_cs` 均构建成功。`MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` 下，`sparkium_fallback_test` **12 项通过、1 项硬件光追测试跳过**，新增无附件 pass 测试通过。

## 复现与原始数据

```sh
cmake --build cmake-build-metal --target demo_nbody_cs sparkium_fallback_test -j8

python3 scripts/profile_nbody.py \
  --exe cmake-build-metal/demo/nbody_cs/demo_nbody_cs

MVK_CONFIG_FAST_MATH_ENABLED=0 python3 scripts/profile_nbody.py \
  --exe cmake-build-metal/demo/nbody_cs/demo_nbody_cs \
  --particles 65536 --output out/nbody-profile/strict-math

# 单次测量
cmake-build-metal/demo/nbody_cs/demo_nbody_cs \
  --backend metal --mode offscreen --particles 65536 \
  --warmup 20 --frames 80 --seed 1 --width 1920 --height 1080 \
  --csv out/nbody-profile/example.csv
```

脚本：[scripts/profile_nbody.py](../scripts/profile_nbody.py)。默认输出目录为 `out/nbody-profile/results`，包含每帧 CSV、stdout/stderr 日志、每次运行命令与结果 `runs.json`、三轮汇总 `summary.json`。严格浮点对照在 `out/nbody-profile/strict-math`。这些运行产物位于被 Git 忽略的 `out/`，本报告保存主要结果。

## 追加实验：Metal 默认开启 fast math

将 `metal_shader.cpp` 的 `setFastMathEnabled(false)` 改为 `true`，重新构建 nbody、Sparkium CLI/GUI 和回归测试。着色器源码和模拟参数没有变化。

同一机器、Release、65,536 粒子、1920×1080，每组预热 20 帧、计时 80 帧，三轮交替运行；Vulkan 保持默认 on-demand fast math。数字为三轮均值的中位数：

| 模式 | Metal 开启前 wall（ms） | Metal 开启后 wall / GPU（ms） | 本轮 Vulkan wall / GPU（ms） |
| --- | ---: | ---: | ---: |
| compute | 56.19 | 31.15 / 30.82 | 31.00 / 30.68 |
| offscreen | 56.46 | 31.24 / 30.91 | 34.16 / 33.76 |

相对上面的原始 Metal 基线，纯计算和完整离屏帧耗时均减少约 **45%**，等效吞吐提高约 **1.80 倍**。开启前数据来自上一轮测量；本轮 Vulkan 也重新测量作为对照。

开启后，两边纯计算耗时基本相同（差约 0.5%）；完整离屏帧 Metal 比 Vulkan 少耗时约 **8.5%**，约合 **32.0 / 29.3 FPS**。这些仍是同步帧延迟换算，不是显示帧率。结果保存在 `out/nbody-profile/metal-fast-math`。

对同样的 65,536 粒子、种子 1，分别读回第 1 步和第 100 步的位置及速度：开启后的 Metal 与 Vulkan **二进制逐位一致**，所有值有限。与保留的 Metal strict-math 旧可执行文件对比，第 1 步位置最大绝对差为 `1.91e-6`、速度为 `2.38e-7`；第 100 步位置 RMSE 为 `0.0910`（参考位置 RMS 的 **1.13%**），速度 RMSE 为 `0.1833`（参考速度 RMS 的 **14.13%**）。因此开启 fast math 会改变累积模拟轨迹，不能保证与旧 strict-math 版本相同；与 Vulkan 一致也不代表物理误差为零。状态文件、检查脚本及结果在该输出目录的 `state/` 和 `check_state.py`。

```sh
python3 scripts/profile_nbody.py \
  --exe cmake-build-metal/demo/nbody_cs/demo_nbody_cs \
  --particles 65536 --output out/nbody-profile/metal-fast-math
```

开启后再次启用 Metal API / shader validation：12 项测试通过，1 项硬件光追测试跳过。Sparkium 与 Vulkan 的图像对照全部通过：6 个光栅场景（RMSE 阈值 0.001），以及 6 个 compute fallback 场景加 1 个 Blender 材质图场景（96×96、256 spp、12 次反弹，阈值 0.03）。光栅最大归一化 RGB RMSE 为 0.000024；compute fallback 最大值为 0.010625。图像及对照结果分别保存在 `out/metal/fast-math-raster` 和 `out/metal/fast-math-compute`。
