# Camera 模型拆分与薄镜演示

将包含镜头参数的单一 Camera 拆分为针孔和薄镜模型，并用独立 demo 展示景深与散景。低面数平滑网格的阴影终止线使用可按网格调节的 Cycles Geometry Offset 修正。

## 实现与使用

- `Camera` 保留为抽象基类，公共接口继续接受 `Camera *`；`CameraPinhole` 与 `CameraThinLens` 分别实现成像模型。具体实现归入 `sparkium/camera/`、`pipelines/raytracing/camera/` 和 `shaders/camera/`。
- 光圈、焦距平面距离、叶片数、旋转与横向比例只属于 `CameraThinLens`；软件追踪、Ray Query 和原生 RT 按具体模型选择着色器，切换模型时刷新相应程序。
- JSON 支持 `camera.type: "pinhole" | "thin_lens"`。未指定类型时，以正的 `aperture_radius` 选择薄镜，否则选择针孔；保持旧场景的选择语义。C++ 中直接构造 `Camera` 的调用需改为具体类型。
- 新增 `demo_thin_lens`：近、中、远三个彩色球、棋盘地面与背景发光点；支持焦点预设、光圈形状、实际针孔相机对比及无窗口输出。球体保持 `Sphere(32,16)`，无表面装饰小点。
- 移植 Cycles Geometry Offset 的抛物线高度近似、局部线性包络、掠射角权重和反射／透射方向处理；阴影射线保留原有遍历与 `TMin` 处理，不额外引入来源三角形过滤。保留 Apache-2.0 许可和固定上游来源。

```cpp
sparkium::CameraThinLens camera(&core, view, fovy, aspect);
camera.aperture_radius = 0.22f;
camera.focus_distance = 5.4f;

mesh.SetShadowTerminatorGeometryOffset(0.1f); // 默认值；范围 [0,1]
mesh.SetShadowTerminatorGeometryOffset(0.0f); // 关闭
film.Reset(); // 修改参数后清空已有累积
```

Geometry Offset 参数控制掠射角作用范围，不是世界空间位移量。参数由网格实例共享，更新现有 GPU 几何缓冲，无需重建 BLAS；非有限值和越界值会被拒绝。Demo 提供 `Geometry offset` 滑块及 `--geometry-offset` 参数。

```sh
cmake --build cmake-build-release --target demo_thin_lens -j8
cmake-build-release/demo/thin_lens/demo_thin_lens --backend metal
```

## macOS 实际渲染对比

以下均为**无窗口渲染结果，不是应用界面截图**。系统窗口截图接口未能捕获 demo 窗口；GUI 运行验证另列于下方。

共同条件：macOS 26.6.2、Apple M5、Metal 原生 Ray Query、1100×700、64 帧 × 8 spp = **512 spp**、4 次反弹、标准 film view transform，相同程序化场景及视图变换。

| 图 | 相机 | 镜头与几何参数 |
| --- | --- | --- |
| 薄镜 | `CameraThinLens` | 焦平面距离 5.4，光圈半径 0.22，六叶片，旋转 0，横向比例 1；Geometry Offset 0.1 |
| 针孔 | `CameraPinhole` | Geometry Offset 0.1 |
| 针孔、关闭修正 | `CameraPinhole` | Geometry Offset 0 |

薄镜：中间绿色球位于焦点附近，近处橙球、远处蓝球与背景出现不同程度的虚化，背景灯呈六边形散景。

![Thin lens, 512 spp](https://media.githubusercontent.com/media/LazyJazzDev/LongMarchAssetsLFS/9f199854f2ec89f5ed64188a6497109150504e24/reports/camera-models-thin-lens/thin-lens.png)

针孔：切换到实际针孔模型；同一场景没有镜头景深虚化。

![Pinhole, geometry offset 0.1, 512 spp](https://media.githubusercontent.com/media/LazyJazzDev/LongMarchAssetsLFS/9f199854f2ec89f5ed64188a6497109150504e24/reports/camera-models-thin-lens/pinhole.png)

关闭 Geometry Offset：与上一图只改变几何修正参数，用于比较低精度球面的明暗交界。场景主光源较大，差异集中在球面的终止线附近；不把本场景视为所有几何的正确性证明。

![Pinhole, geometry offset disabled, 512 spp](https://media.githubusercontent.com/media/LazyJazzDev/LongMarchAssetsLFS/9f199854f2ec89f5ed64188a6497109150504e24/reports/camera-models-thin-lens/pinhole-offset-off.png)

素材与完整复现命令：[固定素材提交](https://github.com/LazyJazzDev/LongMarchAssetsLFS/tree/9f199854f2ec89f5ed64188a6497109150504e24/reports/camera-models-thin-lens)。素材 PR：[LongMarchAssetsLFS #36](https://github.com/LazyJazzDev/LongMarchAssetsLFS/pull/36)，已 squash 合并；主仓库引用该合并提交。

## 验证

- Camera 拆分阶段：Release GUI／CLI／示例及 Debug GUI 构建通过；Cornell 与 Junkshop 输出与拆分前逐字节一致；零光圈 ThinLens 与 Pinhole 的 demo 输出逐字节一致。
- 最终 Geometry Offset 实现：Release `demo_thin_lens`、`sparkium_fallback_test` 和 Debug `demo_thin_lens` 构建通过，Metal GUI 三帧运行测试通过。
- 最终相关回归测试 **23 通过、2 跳过**：包含相机生成／切换、JSON 兼容、共享 shader 编译、BVH 遍历、透明阴影、Cycles 高度／角度公式与参数校验／缓冲更新。
- 本机不支持原生 RT 管线，因此原生 RT 管线切换及图像一致性两项跳过。共享原生 RT 着色器完成 SPIR-V 编译检查；未在本机完成 Vulkan／D3D12 原生 RT 实际渲染验证。
- 本 PR 的渲染图使用代码提交 `1d8cece`，512 spp；移除阴影 `skip` 后重新渲染薄镜、针孔及关闭 Geometry Offset 三组结果，均与已发布图片逐字节一致。
- 阴影 `skip` 参数及为其新增的 Any-Hit 着色器已移除，不透明阴影恢复任意命中提前结束；原有透明阴影处理保持不变。移除后 Release 构建及上述 23 项回归测试通过，2 项仍因设备能力跳过。pre-commit 检查通过。

## 范围与限制

Geometry Offset 只调整直接光阴影可见性，不增加网格精度、不改变轮廓或间接路径起点；大参数可能影响接触阴影。当前未引入 Cycles 的独立 Shading Offset 或 Bump Map Correction。Raster 管线不扩展新功能。
