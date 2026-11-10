# Vulkan 小上传任务开销定位

测量日期：2026-11-10。源码基线：633cb9c7。本文是诊断报告，实验性后端及 benchmark 改动均已撤回。

## 结论

不能把 Assist 比 Main 慢归因于 vkQueueSubmit 的 CPU 调用成本。Linux Nsight trace 中两者提交次数相同，API 中位耗时相近，Assist 的累计提交 API 时间甚至略低。

实现确实有逐 Submission 的重复完成查询。临时将零超时 vkWaitSemaphores 换成 vkGetSemaphoreCounterValue 后，Windows Assist 吞吐提高约 8%，Linux 基本不变。因此它值得优化，但不能解释整个差距。取消 SubmitNative 临时 vector 的堆分配，以及绕过 Token 通知直接等待 GPU Timeline，都没有解决 Assist 退化。

本例最有效的已验证措施是合并小任务的提交：上传与计算录入同一个 Main Context，Windows 比两次 Main 提交快 32.9%，Linux 快 58.3%。这同时减少命令缓冲、提交、信号和回收工作，不能仅将收益算到某个 Vulkan API 上。Assist 的剩余代价与跨队列依赖/完成链有关，但目前缺少有效 GPU 执行时间线，尚不能把驱动调度、GPU semaphore 和执行间隙分开量化。

## 方法与环境

沿用 [吞吐报告](rhi-queue-throughput.md) 的环境：Windows RTX 5080 / NVIDIA 617.42；Meshy-PC-3090 Linux RTX 3090 / NVIDIA 615.71.10。Release、关闭验证层、无窗口和交换链，每帧上传 256 KiB，65536 个元素各执行 8 次整数计算，8 个资源槽位、32768 帧、预热后计时。初始化与最终读回不计时，最终全部槽位逐元素检查。

命令：demo_rhi_queue_bench vulkan MODE upload 65536 8 8 32768 0。

Main 与 Assist 每帧均两次提交；Assist 的计算提交等待上传 Event。main-batched 每帧一次提交。original/counter 每组 5 次交替测量；另外两种实验和分段计时每组 3 次。顺序未随机化，百分之一量级的变化不作为确证。所有 114 次无 Nsight 运行均退出成功、mismatches=0；另有两次 Nsight trace，均零错误。无 Nsight 原始数据见 [CSV](rhi-vulkan-small-upload-profile.csv)。

本轮 Linux 原始 Main 中位值低于上次大矩阵，故只在本轮数据内对比，不混用两轮作为 A/B。

## 无 profiler 对照结果

单位为帧/秒，取中位数。

| 平台 / 实验 | Main 两次提交 | Assist 两次提交 | Main 一次提交 |
|---|---:|---:|---:|
| Windows 原始实现 | 30030.8 | 17361.9 | 39915.4 |
| Windows 改为计数查询 | 31171.1 | 18746.2 | 43128.8 |
| Windows 提交数组用栈存储 | 30315.1 | 17409.4 | 39914.8 |
| Windows 槽位直接等 GPU Event | 28523.4 | 17355.8 | 37390.6 |
| Linux 原始实现 | 18261.5 | 14753.7 | 28902.3 |
| Linux 改为计数查询 | 18553.6 | 14797.1 | 28744.8 |
| Linux 提交数组用栈存储 | 18477.9 | 14946.1 | 28737.3 |
| Linux 槽位直接等 GPU Event | 20322.5 | 14721.4 | 28642.7 |

各实验独立修改原始实现：

- counter：VulkanCommandContext::PollCompletion 中，用 GetCompletedValue(completed) 和 completed < signal.value，替换 queue->Wait(signal.value, 0)。这仍逐条查询，未实现每 Queue 一次快照。
- noheap：仅 SubmitNative 的五个局部 vector 换成容量 16、越界立即终止的临时栈数组封装。此次只有 0/1 个 wait、无 present，容量足够。Queue::Submit 的 wait 去重 vector 保持原样。此诊断封装未保留到产品代码。
- nativewait：仅 benchmark 的槽位复用及末尾 drain 从 s.done->WaitHost() 改为 s.done->GetEvent().timeline->WaitHost(value)。Collector 仍运行，最终读回仍等 Token，正常成功路径下资源复用仍受 GPU 完成约束；它不替代 Token 的结果语义。

直接等待 Timeline 不是一个完全隔离的 Collector 实验：它增加了主线程原生等待，并改变调度、池复用时机。不过两平台 Assist 均无明显改善，不支持“只是 Token 通知太慢”的解释。

## Nsight CPU API trace

Meshy-PC-3090，Nsight Systems 2026.3.2：

    nsys profile --trace=vulkan,osrt --sample=none --cpuctxsw=none --vulkan-gpu-workload=batch -o /tmp/rhi-small-MODE APP vulkan MODE upload 65536 8 8 4096 0

从导出的 SQLite VULKAN_API 和 StringIds 表读取。排除预热的前 64 次 Submit 和读回的末尾 8 次 Submit，取余下 8192 次 Submit 的首尾 CPU 时间窗口。窗口略小于完整计时区间，不包含末尾 drain。

| 计时窗口内指标 | Main | Assist |
|---|---:|---:|
| 窗口长度 | 259.93 ms | 302.02 ms |
| vkQueueSubmit 次数 | 8192 | 8192 |
| vkQueueSubmit 总时间 | 53.57 ms | 50.61 ms |
| vkQueueSubmit 中位数 | 2.455 µs | 2.513 µs |
| vkQueueSubmit P95 | 36.594 µs | 17.006 µs |
| vkWaitSemaphores 次数（Collector 线程） | 41868 | 67519 |
| vkWaitSemaphores 中位数 | 0.350 µs | 0.345 µs |

整进程各 8264 次提交，包含预热和读回。两者各创建 2 个 semaphore、16 个 command pool、16 个 command buffer，说明不是每帧重新创建这些原生对象。

vkWaitSemaphores 同时包含零超时查询与真正阻塞的 WaitForProgress；累计时间不能直接当作 CPU 消耗。进程末尾还有约 50 ms 的等待，不能纳入每帧开销。Trace 下吞吐显著下降，不能用它代替上面的无 profiler 测量。

此 trace 的 VULKAN_WORKLOAD 表只有一条无用的极短区间，没有有效的逐任务 GPU 执行数据；未采集 CPU 调度上下文。因此本文不声称已测出 GPU engine overlap、纯 GPU semaphore 延迟或纯 CPU active time。原始 nsys-rep 保存在本地 out/multiqueue/rhi-small-{main,assist}.nsys-rep，远端 /tmp 同名文件；这些临时二进制不提交 Git。

## CPU 分段计时

在 benchmark 主线程按阶段读取 steady_clock，仅计入正式帧。CSV 保存各段每帧平均耗时；墙钟时间包含抢占与阻塞，不能当成纯指令成本。

Windows 分段测量使用 counter 后端（CSV 标为 counter-phase），Linux 使用原始后端（phase）。Windows 三次运行中：

- Main 槽位 WaitHost：4.56–5.10 µs/帧；Assist：20.87–23.09 µs/帧。
- Main 上传录制：5.44–5.66 µs/帧；Assist：5.31–5.46 µs/帧。
- Main 上传 Submit：10.65–10.88 µs/帧；Assist：8.95–11.00 µs/帧。
- Main 计算 Submit：6.82–6.95 µs/帧；Assist：11.79–12.49 µs/帧。

这定位了主线程额外耗时主要出现的位置，但 WaitHost 包含 GPU 进度及完成通知，Submit 包含 Seal、锁、对象分配和原生提交，不能单凭分段耗时确定底层因果。Linux 各阶段在运行间波动明显，尤其上传录制与槽位等待的分摊，不据此宣称 memcpy 是瓶颈。

## 源码中可改进的地方

1. backend_device.cpp 的 Collect 已查询每条 Timeline，随后 backend_queue.cpp 的 CollectCompleted 又对每条记录调用 PollCompletion。Vulkan 实现逐记录 vkWaitSemaphores(..., 0)，产生重复原生调用。可让每 Queue 的一次完成值快照驱动该队列已完成前缀的回收，并保留设备失效和每提交结果处理。同步维护下一次等待值，不能引入遗漏进度的竞态。
2. SubmitNative 为 semaphore、stage、value 建多个短命 vector；Queue::Submit 还分配 Token、list 节点并获取锁。确有优化空间，但本例 noheap 对照没有可观收益，不应优先重构整个提交接口。
3. Vulkan 的 buffer barrier 使用 ALL_COMMANDS 和宽泛 MEMORY_READ/WRITE，Seal 再恢复宽泛访问状态，跨队列 wait 也使用 ALL_COMMANDS。可能限制并行，但本次没有测量收窄 barrier 的收益，不能直接删除或当作已确认根因。
4. 上传 buffer 已池化且持续映射；Write 是 memcpy 加 vmaFlushAllocation 调用，不是每帧 Map/Unmap。不能因为代码调用 flush 包装就断言每帧执行了昂贵原生 flush。

优先级：调用侧先合并细碎上传/计算提交；后端再实现批量完成快照回收并重新测量；最后在有效 GPU trace 或 timestamp 测量下评估精确同步范围。当前证据不支持为此整体替换 Vulkan API，也不支持把所有小上传迁到 Assist。
