# RHI Hello

用于学习 RHI 的独立示例。统一入口选择 API、demo 和设备，创建一次 Core/Device，然后调用对应的 `RunXxx(IDevice *)`。不带参数启动时使用终端菜单（非终端环境退化为编号/名称输入）；仅有一个后端或设备时自动选择。

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
| blend | Alpha 混合 |
| nbody | Compute、粒子绘制、异步速度读回和交互 |
| raytracing | BLAS/TLAS、压缩、更新及光追管线 |
| ray_query | Compute shader 中的 inline ray query |
| hdr | HDR 颜色空间、参考白及呈现模式切换 |
| imgui | 事件、ImGui 更新和自动呈现 |
| imgui_multi_window | 两个独立窗口和 ImGui context |

光追示例检查所选设备的能力，不支持时输出原因并退出。

每个 `modules/<name>/module.cpp` 自行创建资源、注册事件、执行循环并释放资源，只依赖库接口和本目录 shader。没有 ModuleBase、生命周期回调框架或父目录工具依赖。`module.h` 仅为 main 提供函数声明，示例自身不包含它。Shader 在构建时分别嵌入各模块，不依赖运行目录，也不共享全 demo shader 注册表。

典型循环依次进行事件轮询、窗口状态/交换链处理、获取图像、录制渲染和 Present、提交命令。局部 ObjPtr 管理资源，已接受的 GPU 工作保活所需资源至完成；入口中的 Device/Core 始终比 demo 内资源存活更久。

所有示例只运行窗口模式，不支持 `--headless`、`--all`、`--frames`、`--save-last-frame`、字体或字号参数。最基础的绘图示例只保留 Escape 退出；HDR、NBody 和 ImGui 保留各自演示内容需要的运行中交互。性能基准、图片读回验证应由专门的测试程序承担。
