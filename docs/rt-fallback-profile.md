# Sparkium fallback 性能分析（2026-09-16）

主要耗时集中在 **路径追踪 compute kernel**。512×512、8 spp 时，该阶段占整帧约 90%–98%；128×128、1 spp 时，频繁的同步小 buffer 上传成为主要开销。

## 测量条件

- Apple M5，Vulkan / MoltenVK 1.4.1，Release 构建，强制 `rt_fallback`，关闭验证层。
- 六个基础 demo，统一 12 次最大 bounce，固定场景与相机，逐帧累积采样。
- 主测试每个场景运行 2 个进程，每个进程 24 帧，排除前 4 帧，合计 40 个稳定帧；下表为中位数。
- 低分辨率测试每个场景 1 个进程、24 帧，排除前 4 帧，共 20 个稳定帧。
- 整帧为 `Core::Render + Film::Develop`，不含场景读取、初始化 graphics core、窗口 UI、Present、图像下载、PNG 编码或 CSV 写盘。
- GPU 用 Vulkan timestamp 分段计时；CPU 项为宿主调用的墙钟时间，包含 GPU 同步等待。CPU 的嵌套项和 GPU 时间不能相加。
- 使用正常动态频率，结果反映本机这组测试；不同进程和帧之间存在波动。

## 稳定帧：512×512、8 spp

单位：毫秒。路径追踪包括相机采样、BVH 遍历、材质 / BSDF、直接光照、阴影和多次反弹，尚未细分这些 kernel 内部工作。

| 场景 | 整帧 | 整帧 P95 | GPU 路径追踪 | GPU BVH 构建 | CPU 场景更新 | 追踪 / 整帧 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cornell_box | 89.13 | 95.58 | 80.89 | 0.156 | 6.85 | 90.7% |
| area_light | 177.85 | 202.27 | 168.98 | 0.158 | 7.21 | 95.0% |
| point_light | 143.09 | 149.49 | 136.22 | 0.019 | 5.51 | 95.2% |
| principled | 100.91 | 102.60 | 92.41 | 0.168 | 7.13 | 91.6% |
| specular | 81.53 | 83.71 | 73.36 | 0.169 | 6.76 | 90.0% |
| texture | 478.61 | 501.55 | 469.35 | 0.058 | 7.37 | 98.1% |

其余 GPU 阶段均较小：光源功率 / CDF 预处理合计约 0.04–0.21 ms，film resolve 与 tone mapping 合计约 0.04–0.06 ms。稳定帧的 `blas_rebuilt` 均为 0；此时 BVH 阶段只重建 TLAS。

`render_submit` 的宿主时间会接近 GPU 路径追踪时间：`VulkanCore::SubmitCommandContext` 内部包含 fence 等待和 transfer queue idle 等同步。这个宿主范围同时包含 CPU 编码与 GPU 等待，需要结合 GPU 时间戳解读。

## 同步上传开销

以下静态 buffer 上传时间分布在场景更新、渲染参数和 tone mapping 参数更新中，已包含在整帧内。

| 场景 | 每帧上传次数 | 每帧字节数 | 上传宿主耗时（ms） |
| --- | ---: | ---: | ---: |
| cornell_box | 35 | 6236 | 6.89 |
| area_light | 35 | 6284 | 7.22 |
| point_light | 29 | 3140 | 5.83 |
| principled | 37 | 6644 | 7.30 |
| specular | 35 | 6212 | 6.92 |
| texture | 37 | 7036 | 7.33 |

代码原因：`VulkanStaticBuffer::UploadData` 每次都会先 `WaitGPU()`，随后通过 `SingleTimeCommand` 单独提交复制，后者又执行 `vkQueueWaitIdle()`。即使材质和物体未变化，也会重复上传：例如 `MaterialLambertian::Buffer()` 总是调用 `SyncMaterialData()`，`LightGeometryMaterial::SamplerData()` 总是上传变换。这个开销主要来自小任务提交和同步，而非传输带宽。

## 低采样预览：128×128、1 spp

| 场景 | 整帧（ms） | GPU 路径追踪（ms） | 静态上传宿主耗时（ms） |
| --- | ---: | ---: | ---: |
| cornell_box | 9.25 | 0.84 | 7.22 |
| area_light | 12.23 | 3.38 | 7.47 |
| point_light | 10.62 | 3.34 | 6.17 |
| principled | 9.91 | 0.97 | 7.68 |
| specular | 9.04 | 0.87 | 6.98 |
| texture | 15.31 | 6.52 | 7.23 |

分辨率和 spp 降低后，追踪工作显著减少，上传仍保持约 6–8 ms。这个场景下应先减少重复上传和队列同步。

## 首帧

下表为两个新进程首帧的平均值，系统驱动 shader cache 可能已经预热。首帧未计入稳定帧统计，也不包含前面的 JSON 加载 / graphics core 初始化。

| 场景 | 首帧总计（ms） | fallback 渲染 shader / pipeline 编译（ms） | GPU BLAS + TLAS 构建（ms） |
| --- | ---: | ---: | ---: |
| cornell_box | 345.13 | 143.41 | 1.39 |
| area_light | 469.68 | 160.70 | 5.37 |
| point_light | 357.46 | 141.54 | 3.86 |
| principled | 607.23 | 401.72 | 1.43 |
| specular | 319.11 | 141.08 | 1.06 |
| texture | 987.38 | 383.59 | 8.25 |

此外，5 个 BVH 构建 kernel 的编译和 program 创建约需 24 ms。首帧编译与后续帧的瓶颈需要分开处理。

## 计时开销对照

另外运行了同进程 ABBA 对照：开启 / 关闭 / 关闭 / 开启 GPU 标记，共 64 帧，排除前 4 帧，两组各 30 帧。

| 场景 | 无 GPU 标记整帧（ms） | 有 GPU 标记整帧（ms） | 中位数差异 |
| --- | ---: | ---: | ---: |
| cornell | 82.20 | 82.01 | -0.23% |
| texture | 490.23 | 492.18 | +0.40% |

这组同进程对照的差异不超过 0.40%，仍包含帧间噪声。先前独立进程的开启 / 关闭测试曾出现约 8%–12% 差异，不能将其全部归为计时开销；不同时段 / 进程的绝对时间存在波动。本次瓶颈排序在这些运行中保持一致。

```sh
cmake-build-rt-fallback/demo/sparkium_cli/demo_sparkium_cli \
  out/rt-profile/512-8spp/texture.json --pipeline rt_fallback \
  --frames 64 --profile out/rt-profile/alternate-texture.csv \
  --profile-alternate-gpu -o out/rt-profile/alternate-texture.png
```

## 优化优先级

1. **正常分辨率：继续细分追踪 kernel。** 当前数据只能确认整个路径追踪阶段最贵，不能把时间直接归因于遍历或 BSDF。下一步可以测量节点访问 / 三角形测试数量，评估近优先遍历、BVH 质量和线程分歧；这些方向还没有用本次数据验证收益。
2. **低采样预览：减少同步小上传。** 给材质、变换、灯光分布等加正确的失效判定，只更新变化项，合并上传或使用适合逐帧数据的 buffer。每帧几 KB 数据目前花费约 6–8 ms。
3. **首帧：关注编译和 pipeline 缓存。** 本次 GPU BVH 构建最多约 8 ms，而渲染 program 创建达 140–402 ms。

## 复现

```sh
cmake --build cmake-build-rt-fallback --target demo_sparkium_cli -j 6
python3 scripts/profile_rt_fallback.py \
  --cli cmake-build-rt-fallback/demo/sparkium_cli/demo_sparkium_cli \
  --output out/rt-profile/512-8spp
python3 scripts/profile_rt_fallback.py \
  --cli cmake-build-rt-fallback/demo/sparkium_cli/demo_sparkium_cli \
  --size 128 --spp 1 --repeat 1 --output out/rt-profile/128-1spp
```

单独测某个 JSON，可使用 `--frames 24 --profile timings.csv`。`--profile-cpu-only` 关闭 GPU 标记，保留宿主计时；`--profile-alternate-gpu` 按开启 / 关闭 / 关闭 / 开启的顺序逐帧交替，用于同进程开销对照。不要在性能采集时同时运行其他测试或打开验证层。

原始 CSV、场景副本、PNG、日志和汇总 JSON 位于本机 `out/rt-profile/`。普通运行不启用计时，图像输出流程不变；profile 模式每帧执行 Develop，以覆盖显示图像处理。

## 验证

CLI、GUI 和 fallback 测试目标构建通过；原有 CTest 检查通过。Cornell 的 8 帧交替计时在验证层下无报错，开启计时与普通运行生成的 PNG 字节完全一致。
