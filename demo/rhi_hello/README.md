# RHI Hello

用于学习 RHI 的独立示例。统一入口先选择 API 和设备，再选择 demo，创建一次 Core/Device，然后调用对应的 `RunXxx(IDevice *)`。不带参数启动时使用终端菜单（非终端环境退化为编号/名称输入）；仅有一个后端或设备时自动选择。

```sh
demo_rhi_hello
demo_rhi_hello --list
demo_rhi_hello --backend metal --module triangle --device auto
demo_rhi_hello --backend vulkan --module cube --device 1
```

设备编号从 1 开始，与设备菜单一致；`auto` 使用 RHI 的自动选卡策略。关闭窗口或按 Escape 结束当前 demo。正常返回后重新选择下一个 demo，复用已选后端和设备；非零返回值直接作为进程退出码。取消 demo 菜单退出程序，`--module` 只指定首次运行的 demo。多窗口 demo 关闭一个窗口后继续运行，所有窗口关闭后返回菜单。

| Demo | 学习内容 |
| --- | --- |
| window | 创建窗口、获取帧图像、清屏、提交和呈现 |
| triangle | 顶点/索引缓冲、shader、图形 Program、绘制 |
| cube | Uniform、旋转变换、深度附件及窗口缩放 |
| texture | 上传图片、纹理和采样器绑定 |
| mip | GPU MIP 链、三线性和各向异性过滤、九个旋转纹理立方体 |
| blend | Alpha 混合 |
| nbody | Compute、粒子绘制、异步速度读回和交互 |
| raytracing | BLAS/TLAS、压缩、更新及光追管线 |
| ray_query | Compute shader 中的 inline ray query |
| hdr | HDR 颜色空间、参考白及呈现模式切换 |
| imgui | 事件、ImGui 更新和自动呈现 |
| imgui_multi_window | 两个独立窗口和 ImGui context |

光追示例检查所选设备的能力，不支持时通过 `tui::MessageBox` 显示原因，确认后返回菜单。非终端环境输出消息并等待 Enter，EOF 不阻塞。

每个 `modules/<name>/module.cpp` 自行创建资源、注册事件、执行循环并释放资源，只依赖库接口和本目录 shader。没有 ModuleBase、生命周期回调框架或父目录工具依赖。`module.h` 仅为 main 提供函数声明，示例自身不包含它。Shader 在构建时分别嵌入各模块，不依赖运行目录，也不共享全 demo shader 注册表。

典型循环依次进行事件轮询、窗口状态/交换链处理、获取图像、录制渲染和 Present、提交命令。局部 ObjPtr 管理资源，已接受的 GPU 工作保活所需资源至完成；入口中的 Device/Core 始终比 demo 内资源存活更久。

所有示例只运行窗口模式，不支持 `--headless`、`--all`、`--frames`、`--save-last-frame`、字体或字号参数。最基础的绘图示例只保留 Escape 退出；HDR、NBody 和 ImGui 保留各自演示内容需要的运行中交互。性能基准、图片读回验证应由专门的测试程序承担。

## MIPMAP 示例

```sh
demo_rhi_hello --backend d3d12 --module mip --device auto
demo_rhi_hello --backend vulkan --module mip --device auto
```

从左到右是无 MIP（双线性且 `max_lod = 0`）、三线性、各向异性过滤（最高 16x，受设备上限约束）。从下到上依次是大、中、小立方体。三列使用相同几何、角度和 2048×2048 程序化纹理，纹理包含细棋盘、横竖条纹和粗网格；GPU 从第 0 级生成完整链。Space 暂停/继续旋转，Escape 返回菜单。不支持 GPU 生成或各向异性的设备会报告原因。

`test_rhi --gtest_filter=*RhiMip*` 使用相同场景代码离屏绘制，验证物体数量、尺寸顺序、旋转变化和过滤差异，不需要桌面会话。设置 `LONGMARCH_RHI_CAPTURE_DIR` 为已存在目录时，测试可导出 PPM 渲染结果；它不包含窗口和 ImGui 标签。

## MIPMAP 接口

D3D12/Vulkan 的 `PhysicalDevice::mip_feature` 提供图像尺寸、各向异性和 LOD bias 范围。独立的 `PhysicalDevice::format_features[IMAGE_FORMAT_COUNT]` 普通数组使用 `PhysicalDeviceFormatFeature` 描述逐格式的采样、线性过滤、存储、线性 blit、最大 MIP 级数和 MIP 计算生成能力。可通过 `ICore::GetPhysicalDevices()` 或 `IDevice::GetPhysicalDevice()` 查询。能力描述硬件支持，创建仍可能因资源组合或内存不足失败。

```cpp
// 省略 mip_levels 时仍只创建一级；0 表示按尺寸创建完整链。
ObjPtr<IImage> texture = device->CreateImage({1024, 512, IMAGE_FORMAT_R8G8B8A8_UNORM, 0});
// 以下调用均应检查返回值。
commands->CmdUploadImage(texture.Get(), pixels, row_pitch);           // 第 0 级
commands->CmdUploadImage(texture.Get(), level2, level2_pitch, 2);     // 单级
commands->CmdUploadImage(texture.Get(), {{level0, pitch0, 0}, {level1, pitch1, 1}});
commands->CmdGenerateMipmaps(texture.Get());                          // 从第 0 级生成后续全部级别
commands->CmdGenerateMipmaps(texture.Get(), 2, 3);                    // 读取第 2 级，生成第 3、4 级
ObjPtr<IDownloadToken> result = commands->CmdDownloadImage(texture.Get(), {0, 0, 0, 0, 3});
group->BindResource(0, ImageView{texture.Get(), 2, 3});                 // 采样第 2～4 级
```

`ImageView::mip_count = 0` 对采样绑定表示剩余全部级别，对存储绑定表示单级；存储绑定只能选择一级。旧的 `IImage*` 绑定继续有效，采样时可见整条链，存储时选择第 0 级。着色器 LOD 0 相对于视图的 `base_mip`。绑定组保存已解析范围，修改绑定组不会改变已录制命令。

采样器可独立设置 `min_filter`、`mag_filter`、`mip_filter`，`address_u/v/w`，`min_lod`、`max_lod`、`mip_lod_bias` 和 `max_anisotropy`。旧的构造方式和 `.filter` 赋值继续支持；未指定的独立过滤参数继承 `.filter`。各向异性大于 1 时三个过滤参数须为 LINEAR。

GPU 生成采用逐级线性缩小，支持非二次幂尺寸；它不是离线纹理压缩器，也不执行 gamma 转换。格式须支持 `mip_compute_generation`，源级必须已有有效数据。生成命令使用自己的计算程序和绑定，调用后应重新绑定应用程序的 Program/Group；只能在渲染作用域之外调用。Metal 暂不报告/提供 GPU 生成能力。

上下文按 mip 追踪资源状态和访问依赖，允许同一 dispatch 读取一个 mip 并写入另一个 mip，拒绝重叠范围的读写绑定。提交时恢复各级默认状态；跨队列仍须使用 `SubmitInfo::waits`，不能同时无依赖地访问同一资源。上传和读回的尺寸、行间距及区域均相对于所选 mip。现有渲染附件和 standalone clear 仍针对第 0 级。本轮不增加 TextureArray。
