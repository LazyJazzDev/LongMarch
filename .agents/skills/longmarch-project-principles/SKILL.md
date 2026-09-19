---
name: longmarch-project-principles
description: Apply the user-confirmed build, commit-message, squash-merge, PR-history, and asset conventions for LazyJazzDev/LongMarch and LongMarchAssetsLFS. Use when building or editing these projects, creating commits, preparing or merging PRs, or changing branch history and asset references.
---

# LongMarch 项目原则

这些是用户已确认的项目约定。适用于 LongMarch 及其 LongMarchAssetsLFS 素材库；不要套用到无关项目。当前用户明确提出的新要求优先于本文件。

## 构建

- LongMarch 默认使用 CMake 的 `Ninja` 生成器（`-G Ninja`），除非用户明确指定其他方案。单配置 Release 构建使用 `-DCMAKE_BUILD_TYPE=Release`。
- 切换生成器时使用独立构建目录，避免复用其他生成器的 CMake 缓存。
- Ninja 下示例和测试通过 `EXCLUDE_FROM_ALL` 排除在默认目标外；用户要求全量编译时，也要显式构建这些目标。

## 更新与版本管理

- 每次完成代码更新后，都要及时提交到 Git；文档、测试和构建配置等配套修改一并纳入版本管理，不将已完成的修改长期留在工作区。
- 按可独立理解的逻辑单元提交，完成与改动相符的验证后再提交；持续任务在完成阶段性工作时及时建立提交记录，并在结束回复前提交本次已完成的修改。
- 提交前检查 diff 和暂存范围，仅纳入本次工作相关的文件，不混入无关修改、构建产物、临时诊断文件或本机运行状态。
- 提交必须通过仓库的 pre-commit 检查；首次使用工作副本时执行 `pre-commit install --install-hooks`，提交前执行 `pre-commit run` 检查暂存文件。钩子自动修改文件后重新检查差异、暂存并运行，直到全部通过；不得用 `--no-verify` 或跳过钩子来绕过检查。
- 格式遵循仓库 `.clang-format` 及子目录覆盖配置，使用 `.pre-commit-config.yaml` 固定的 clang-format 版本。更新格式配置时执行 `pre-commit run --all-files`，确认全仓受管文件通过检查；保留函数和类型定义之间的空行，着色器 include 顺序按其目录配置处理。
- 此原则明确授权对已获准的工作创建本地提交，无需每次再次确认；推送、合并和改写已发布历史仍遵循各自的授权范围。
- 完成后报告提交哈希及验证情况；若提交失败，说明原因并保留修改，不将未提交的工作报告为已提交。

## 提交信息

普通 commit 和最终 squash commit 的标题都使用：

```text
[type][module] Subject
```

- 两组方括号之间没有空格，第二组后有一个空格。标题沿用项目的简洁英文描述。
- `type` 选用与改动相符的类型，例如 `feat`、`fix`、`docs`、`chore`、`refactor`、`perf`、`test`。
- `module` 使用实际模块，例如 `sparkium`、`graphics`、`assets`、`vscode`；不强行把跨模块改动归入错误模块。
- 不要替换为 `feat(scope): ...` 等其他格式。PR 标题可以是自然语言，但不能直接将没有前缀的 PR 标题作为最终提交标题。
- “提交 comment 格式”指 commit message，不是源代码注释或 GitHub 讨论评论。

示例：

```text
[fix][sparkium] Display the resolved automatic pipeline
[feat][graphics] Add native Metal acceleration structures
[feat][sparkium] Add native Metal ray queries (#40)
```

## PR 合并

- 所有修改必须在独立工作分支上提交，经过充分验证后通过 PR 合并到主分支；禁止直接向主分支推送提交，包括普通推送和强制推送。此规则同样适用于代码、文档、配置和素材仓库的更新。
- 开始修改或提交前确认当前分支；若处于主分支，先创建或切换到独立工作分支。不得以本地合并后推送主分支等方式绕过 PR。
- PR 合并前完成与改动范围相符的构建、测试和必要的人工检查，记录验证结果，并确认适用的 CI 与仓库合并要求已满足。
- 合并 PR 使用 **squash merge**：每个 PR 在目标分支形成一个提交，不保留分支内分批提交的记录。不要选普通 merge commit 或 rebase merge。
- 显式指定符合上面格式的 squash 提交标题。正文描述最终行为及相关验证，不采用自动生成的逐条分支提交清单。
- 根据当前远端检查目标分支、待合并的提交、合并状态及适用检查。使用 GitHub CLI 时，优先通过 `--match-head-commit` 校验已检查的 head，避免合入检查后新增的未知提交。
- 合并完成后核实 PR 状态和目标分支上的 squash commit，再报告成功。
- 父分支已经 squash 合入 main 时，后续分支的 PR 应只包含尚未合入的工作。检查 merge base 和最终 diff，必要时重新整理基线；不要把旧的父分支提交重复当作新增功能。

GitHub CLI 的相关参数为 `--squash`、`--match-head-commit`、`--subject` 和 `--body-file`。正文通过文件传入，保留真实换行；不要把前述示例中的 PR 编号当作当前任务的编号。

## 素材库与引用

- 二进制文件统一保存到 `assets/`，无论是图片还是其他素材；图文报告中的插图、渲染结果图、截图和对比图也遵循此规则。
- 需要提交到 Git 的二进制素材在生成或导出时直接使用 `assets/` 下的路径，不将素材散落在 `docs/` 或其他目录。报告正文与生成脚本可以保存在各自目录，通过路径引用 `assets/` 中的素材；移动素材时同步更新引用和生成脚本的输出路径。
- 例外：`out/` 中的临时报告及其插图、渲染结果等文件不会提交到 Git，不受上述存放位置限制，可以直接保存在 `out/`。若后续需要将其中的二进制素材纳入版本管理，应先移入 `assets/` 并更新引用。
- 素材使用 LongMarchAssetsLFS 的 Git LFS 流程，主仓库通过子模组提交引用对应素材版本。
- 大小写文件名冲突要区分命名、分别保存，并同步修改场景中的引用。不能用覆盖或丢弃其中一个文件来消除冲突。
- 修改素材时检查子模组指针与场景资源路径是否一致；不因普通素材更新改写旧 LongMarchAssets 仓库。

## 授权与维护

本 skill 规定做事方式，不额外授予推送、合并或改写已发布历史的权限。当前任务中已有的明确授权继续有效，不重复请求确认；改写已发布历史时确认授权确实涵盖该改写范围，并用带旧提交校验的 `--force-with-lease`。

仓库中的 `.agents/skills/longmarch-project-principles/` 是可共享、可版本管理的规则来源。本机安装在用户技能目录的副本用于自动发现，并在旧分支或素材仓库缺少项目副本时提供这些约定。两者有差异时，以当前项目副本和用户最新要求为准；修改规则时同步本机安装副本。不要把临时分支拓扑、构建目录、PR 编号、凭据或一次性授权写成永久原则。
