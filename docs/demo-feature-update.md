## 本次更新

本 PR 为 GOL 增加局面文件保存与加载、两组可直接运行的预设图案，并调整控件布局与动画；同时修复 2048 的新局 AI 状态和大分数显示。

## GOL：保存局面，随时继续

- 点击**打开／保存**图标，使用 tinyfiledialogs 原生文件对话框读取或写入 `.cells` 文件；支持 **Ctrl/Cmd + O / S**。
- 保存完整网格尺寸及空白边界；加载后同步 W/H 滑条、适配视图并暂停演化，便于检查局面。
- 取消文件选择保留当前局面；无效文件给出错误提示。支持 Unicode 路径。

![GOL 侧栏与 295P5H1V1 图案](https://media.githubusercontent.com/media/LazyJazzDev/LongMarchAssetsLFS/4c025131f0f88306df9fe5073dcc0b36d10c0116/reports/pr56-demo-features/gol-sidebar.png)

*100 × 100 网格中的 295P5H1V1 预设。左侧依次为文件操作、并排尺寸滑条、速度和播放；重置与随机化位于另一侧两个角。*

### 控件与反馈

- 横向布局将主工具栏放在顶部，重置和随机化位于底部两角。
- 文件图标采用粗线条连续轮廓；箭头与成功勾选平滑切换，成功／失败提示色渐入渐出。
- 尺寸滑条使用粗笔画圆角字模；骰子明暗按透视投影后的屏幕坐标计算，并响应按钮状态。

![GOL 顶部工具栏与 Gosper glider gun](https://media.githubusercontent.com/media/LazyJazzDev/LongMarchAssetsLFS/4c025131f0f88306df9fe5073dcc0b36d10c0116/reports/pr56-demo-features/gol-horizontal.png)

*100 × 25 网格中的 Gosper glider gun 初始局面。双滑条上下排列，总高度与两侧按钮一致。静态截图展示最终布局，不代表动画时序。*

### 两组预设图案

| 图案 | 演化行为 | 使用方式 |
| --- | --- | --- |
| 295P5H1V1 | 每 5 代沿对角线移动一格 | 打开 `demo/gol/patterns/295P5H1V1.cells` |
| Gosper glider gun | 每 30 代发射一个滑翔机 | 打开 `demo/gol/patterns/gosper-glider-gun.cells` |

也可从命令行加载并开始演化：

```sh
cmake-build-ninja/demo/gol/demo_gol 100 100 --pattern demo/gol/patterns/295P5H1V1.cells --play
cmake-build-ninja/demo/gol/demo_gol 100 25 --pattern demo/gol/patterns/gosper-glider-gun.cells --play
```

## 2048：新局回到手动，大分数自动适配

- **NEW GAME / TRY AGAIN** 关闭 AI，清除待执行动作与启动手势计数，分数栏恢复 **SCORE**；显式启动参数 `--ai` 仍可用于首次启动。
- 分数根据背景框可用宽度自动缩小，保留左右留白；分数变化或窗口缩放时重新计算，小分数恢复默认字号。

<img src="https://media.githubusercontent.com/media/LazyJazzDev/LongMarchAssetsLFS/4c025131f0f88306df9fe5073dcc0b36d10c0116/reports/pr56-demo-features/2048-score-fit.png" alt="2048 游戏结束界面：十位分数自动适配背景框" width="480">

*真实游戏结束界面渲染，分数 `2147483647` 由运行时探针注入，用于验证十位数字排版，并非实际游玩成绩。*

## 验证与截图说明

- Ninja Release 构建：`demo_gol`、`demo_2048` 通过；GOL 相关 **19 项测试全部通过**。
- Metal 运行时验证：文件读写、取消／失败恢复、加载后的网格绑定、AI 新局重置，以及长分数、空文本和窗口缩放适配。文件读写探针使用模拟文件选择结果，未自动操作原生对话框。
- 所有配图均通过 macOS 系统窗口截图命令 `screencapture -x -l` 生成，保留真实标题栏、窗口外框和系统阴影。环境为 macOS / Apple M4 / Metal，功能代码版本 `a23cf51`。包含窗口与阴影的 PNG 尺寸：GOL 侧栏 **2784 × 1728**、顶部工具栏 **2696 × 1640**，2048 **1664 × 2208**。应用渲染分别使用每轴 2×（GOL）和 3×（2048）超采样。
- 图片通过 Git LFS 发布，固定到素材提交 `4c025131f0f88306df9fe5073dcc0b36d10c0116`；[素材 PR #26](https://github.com/LazyJazzDev/LongMarchAssetsLFS/pull/26) 已 squash 合并。[截图来源与复现说明](https://github.com/LazyJazzDev/LongMarchAssetsLFS/blob/4c025131f0f88306df9fe5073dcc0b36d10c0116/reports/pr56-demo-features/README.md)。
