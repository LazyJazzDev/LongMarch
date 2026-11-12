# 2048 / Game of Life RHI 迁移验证

验证日期：2026-11-12。旧版基线为 `e233d265`（main，PR #113），迁移分支为
`migrate-demos-rhi`。初始迁移在独立 worktree 中完成，本机补充验证与手势迁移在主工作目录的该功能分支上完成。

## 实现

两个 demo 直接链接 `grassland_rhi`、`grassland_util` 和各自需要的 glm / FreeType /
tinyfiledialogs，不再通过 `LongMarch` 间接依赖旧 `grassland_graphics` 渲染器。

- 使用 RHI Core、Device、Main Queue、不可变 Program、BindingGroup 和 ObjPtr。
- 网格、实例、GoL 单元数据和 uniform 通过命令上下文上传，与本帧绘制按队列顺序提交。
- 缓冲区扩容替换物理分配，已提交命令保留旧资源；移除网格析构中的全 GPU 等待。
- 两个 submission slot 限制在途帧数；resize、退出和截图显式等待提交完成。
- 保留原 Slang shader、深度排序、逐轴 3x 超采样及第二帧混合过渡。
- 截图通过 RHI download token 读取 resolve 后的 RGBA8 图像；呈现显式声明 sRGB 编码。
- 可预期初始化、录制、提交和截图失败返回失败状态，进程返回非零，不把空资源当作成功。
- 应用直接持有 RHI `IWindow`；原生捏合事件由 RHI 公共窗口层提供，GoL 只注册业务回调。验证完成后已移除临时窗口适配层及 benchmark 注入。

## 行为一致性

以下 14 组新旧 RGBA 截图逐像素一致：最大通道差值 0，差异像素数 0。

| 验证范围 | 场景 | 配对数 |
| --- | --- | ---: |
| Windows D3D12 / Vulkan、Meshy-PC-3090 Vulkan/X11 | 固定种子 2048、暂停的 128×128 / 256×256 GoL（35% 初始活细胞） | 9 |
| Meshy-PC-3090，真实 X11/GLFW 输入事件，开启呈现 | 2048 Left/Down/Right/Up、菜单、960×720 resize + unmap/map | 3 |
| 同上 | GoL 稳定块播放/暂停、单元编辑、两次缩放、右键平移；滑块/方向键、720×960 resize + unmap/map | 2 |

回放在独立 Xvfb 显示中进行，通过实际键鼠事件和窗口管理请求驱动 GLFW；没有修改游戏逻辑
来绕过输入。X11 unmap/map 验证隐藏/恢复，不等同于所有桌面窗口管理器的最小化行为。
本次未完整自动化文件选择对话框或所有平台的 DPI/显示器切换。

- Windows 原生 Release 和 Debug 两个 demo 构建通过。
- Debug：两个 demo 均在 D3D12、Vulkan 运行 240 帧；包括 2048 自动游玩和 GoL 播放。
  未发现验证错误。D3D12 报告 820/821 性能警告：现有 RHI 图像分配器未指定优化 clear value。
- 31 项 GoL 测试通过，覆盖演化、边界、文件格式、网格尺寸/视图和模拟时钟。
- 2048 AI 模型核对 512 个随机棋盘及各方向；1 ms 搜索预算的单局冒烟达到 2048。
  这不是 AI 策略强度的统计评估。
- 原 PR 的 macOS/Metal 验证空缺由下方本机验证补充。

### 回放截图

图片是应用导出的 resolve 图像，不是桌面窗口截图。每轴 3x 超采样；旧版与 RHI 成对图像完全一致。

2048 移动后，720×960：

![2048 RHI 移动后](https://media.githubusercontent.com/media/LazyJazzDev/LongMarchAssetsLFS/5a9ed185d725024b5bc7f1a5dec2117d8a078100/reports/demo-rhi-migration/2048-moves-rhi.png)

GoL 编辑、缩放和平移后，1280×720：

![GoL RHI 缩放和平移](https://media.githubusercontent.com/media/LazyJazzDev/LongMarchAssetsLFS/5a9ed185d725024b5bc7f1a5dec2117d8a078100/reports/demo-rhi-migration/gol-controls-rhi.png)

GoL 调整网格及竖屏窗口后，720×960：

![GoL RHI 竖屏](https://media.githubusercontent.com/media/LazyJazzDev/LongMarchAssetsLFS/5a9ed185d725024b5bc7f1a5dec2117d8a078100/reports/demo-rhi-migration/gol-resize-rhi.png)

截图由 [LongMarchAssetsLFS PR #43](https://github.com/LazyJazzDev/LongMarchAssetsLFS/pull/43)
保存，主仓库固定引用 squash 提交 `5a9ed185d725024b5bc7f1a5dec2117d8a078100`。

## 渲染性能

**这里测量离屏应用帧循环吞吐，不是 GPU timestamp，也不是显示器实际呈现 FPS。**
绘制、上传、超采样与 resolve 全部保留，只跳过呈现，排除后台窗口节流、VSync 和 Xvfb
传输造成的偏差。事件处理和游戏 UI 更新仍计入 CPU 时间。

每次先预热 120 帧，每帧额外等待 5 ms，使启动动画有至少 0.6 秒结束；预热不计入结果。
随后测量 6000 帧，计时前后均等待 GPU 完成。每项交替运行两版各 3 次，表中取吞吐中位数。
原始逐次结果同时记录 CPU 帧时 p50 / p95，详见
[原始数据](benchmarks/demo-rhi-migration.json)。

- 构建：Release / Ninja；同机两版使用相同工具链，禁用 CUDA 编译和 Python 扩展。
- 2048：720×960，固定种子 1，不启用 AI。
- GoL：1280×720，暂停，128×128 或 256×256，随机种子 0，35% 活细胞。
- 以上图像均每轴 3x 超采样：2048 内部 2160×2880，GoL 内部 3840×2160。
- Windows：Core Ultra 7 270K Plus / RTX 5080，NVIDIA 617.42，MSVC 19.51.36260.0。
- Meshy-PC-3090：i9-11900K / RTX 3090，NVIDIA 615.71.10，GCC 15.2.0，Linux 7.0.0-38-generic。
  两版均强制 X11，在独立 1920×1080×24 Xvfb 显示中创建窗口，测量期间不呈现。

| 环境 | 场景 | graphics 帧/秒 | RHI 帧/秒 | 吞吐变化 |
| --- | --- | ---: | ---: | ---: |
| Windows / RTX 5080 / D3D12 | 2048 | 2935 | 7980 | +171.9% |
| Windows / RTX 5080 / D3D12 | gol-128 | 2460 | 4798 | +95.0% |
| Windows / RTX 5080 / D3D12 | gol-256 | 1928 | 2759 | +43.1% |
| Windows / RTX 5080 / Vulkan | 2048 | 6909 | 12364 | +79.0% |
| Windows / RTX 5080 / Vulkan | gol-128 | 3794 | 7097 | +87.1% |
| Windows / RTX 5080 / Vulkan | gol-256 | 2047 | 3113 | +52.1% |
| Meshy-PC-3090 / Vulkan X11 | 2048 | 5653 | 9887 | +74.9% |
| Meshy-PC-3090 / Vulkan X11 | gol-128 | 2835 | 4903 | +72.9% |
| Meshy-PC-3090 / Vulkan X11 | gol-256 | 1572 | 2648 | +68.4% |

这些结果反映本次场景下的整体帧循环成本。迁移批量记录上传、避免网格替换时的全 GPU 等待，
但本测试没有拆分各项 CPU/GPU 开销，不能将全部提升归因于单一机制。未锁定 GPU 时钟，
共享机器的调度和功耗状态也可能带来波动；不跨机器直接比较绝对帧率。

## 复现

以下为历史性能验证流程。临时 `frame_benchmark`、离屏环境变量开关及测量脚本已从当前 demo 移除；
复现时请先在隔离工作树检出 `303b97a`，该版本保留下列脚本及注入代码。当前 demo 始终正常呈现。

先在单独的基线工作树检出 `e233d265`，初始化依赖和素材；从迁移工作树运行：

```sh
python scripts/instrument_ui_demo_baseline.py <baseline-worktree>
```

该脚本只向旧版加入相同的计时、离屏开关和验证种子，不迁移其渲染实现。
两个工作树各自通过 CLion Release 配置生成 `cmake-build-release`，然后显式构建两个 demo：

```sh
cmake --build cmake-build-release --target demo_2048 demo_gol
python scripts/benchmark_ui_demos.py --baseline-root <baseline-worktree>/cmake-build-release/demo --candidate-root cmake-build-release/demo --backend vulkan --output out/demo-comparison
```

Windows D3D12 使用 `--backend d3d12`。Linux 无桌面环境时，在整个脚本外包一层
`xvfb-run -a -s '-screen 0 1920x1080x24'`；脚本对 Vulkan 固定 X11，避免 RHI 自动选择 Wayland。

```sh
xvfb-run -a -s '-screen 0 1920x1080x24' python3 scripts/replay_ui_demos_x11.py --baseline-root <baseline-worktree>/cmake-build-release/demo --candidate-root cmake-build-release/demo --output out/demo-replay
python scripts/compare_ui_demo_images.py out/demo-comparison out/demo-replay
```

回放需要 `xdotool` 和 `libX11`，图像比较需要 Pillow。比较脚本默认容差为 0，输出
`images.json`；尺寸不一致、缺少配对或差异超过容差均失败。性能脚本输出每次日志、
`results.json`、`summary.json` 和首轮两版 PNG。

## RHI 原生捏合接口

`rhi::IWindow` 与 `rhix::Window` 提供 `SupportsMagnifyGestures()` 和 `MagnifyEvent()`。
事件使用 `rhi::MagnifyGesture`：`scale` 是相对上一次事件的增量缩放（1 表示不变），
`x/y` 为窗口内容区左上角起算的逻辑坐标，与鼠标移动事件一致；`phase` 区分开始、更新、结束和取消。
取消事件的 scale 为 1，不自动撤销已处理的更新。Scroll / Ctrl+scroll 保持独立，不重复合成手势。

macOS 由 RHI 公共窗口层接收 Cocoa 事件，适用于 Metal 和 Vulkan；其他平台可注册回调但暂不产生原生捏合事件，能力查询返回 false。
监听器随窗口创建、销毁；分派通过窗口注册表获取临时强引用，允许回调期间释放窗口，且不会串到其他窗口。
GoL 的焦点缩放及网格视图逻辑仍留在应用内。demo 不再编译 Objective-C++ 事件桥接或直接链接 Cocoa。

## macOS / Metal 本机验证补充

Apple M5，macOS 27.0.1，CLion Release/Ninja。基线 `e233d265`，迁移版 `532078e0`；
同一构建配置依次构建、保存二进制后交替测量，无并行构建干扰。两版使用相同本地素材。
预热 120 帧、测量 6000 帧，每场景每版 3 次，取中位数。此次保持屏幕原生 Retina 缩放：
2048 输出 **1440×1920**（内部 4320×5760），GoL 输出 **2560×1440**（内部 7680×4320）。
因此不要将绝对 FPS 与上文 Windows/Linux 较低分辨率的数据直接比较。

| Metal 场景 | graphics FPS | RHI FPS | 变化 |
| --- | ---: | ---: | ---: |
| 2048 | 469.003 | 715.453 | +52.55% |
| GoL 128² | 513.617 | 640.532 | +24.71% |
| GoL 256² | 496.196 | 622.491 | +25.45% |

18 次原始结果、环境及截图数值比较见 [macOS 原始数据](benchmarks/demo-rhi-migration-macos.json)。
三组新旧 resolve 输出逐像素一致（最大通道差值 0）。图像仅用于本机数值比较，未作为 UI 截图发布。
计时仍是关闭呈现的应用帧循环吞吐；没有测量 GPU timestamp，没有锁频，也未将提升归因于单一原因。

手势迁移后的 Release / Debug 构建通过。两种配置下各运行 5 项事件/窗口回归测试全部通过，
包括 Metal、Vulkan 原生窗口的阶段/坐标转换、多窗口隔离、回调取消注册、回调释放最后外部窗口引用和窗口重建。
事件载荷由测试合成，坐标转换使用真实 NSWindow/NSView；未宣称完成物理触控板的人工捏合验收。
Release 两个 demo 在 Metal、Vulkan 各呈现 240 帧并正常退出；Debug 两个 demo 在 Metal 各呈现 240 帧。
最终 Metal 的 2048、GoL 128² 输出与迁移手势前的捕获仍逐像素一致。
