---
name: longmarch-project-principles
description: Apply the user-confirmed working-directory, commit-message, squash-merge, PR-history, and asset conventions for LazyJazzDev/LongMarch and LongMarchAssetsLFS. Use when editing these projects, switching branches, creating commits, preparing or merging PRs, or changing branch history and asset references.
---

# LongMarch 项目原则

这些是用户已确认的项目约定。适用于 LongMarch 及其 LongMarchAssetsLFS 素材库；不要套用到无关项目。当前用户明确提出的新要求优先于本文件。

## 工作目录

- 能在用户的主工作目录完成的操作，都在主工作目录内完成，包括切换分支、编辑、构建和验证。需要区分构建产物时，使用主工作目录内的不同构建子目录。
- 不自行在主工作目录之外新建 worktree、克隆或复制分支文件，也不因切换分支或隔离任务而将开发迁到外部目录；用户明确要求外部工作目录时除外。
- 切换前检查未提交修改和已有 worktree 的分支占用，保留用户文件；已有外部 worktree 不作为继续在外部开发的默认理由。

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

- 合并 PR 使用 **squash merge**：每个 PR 在目标分支形成一个提交，不保留分支内分批提交的记录。不要选普通 merge commit 或 rebase merge。
- 显式指定符合上面格式的 squash 提交标题。正文描述最终行为及相关验证，不采用自动生成的逐条分支提交清单。
- 根据当前远端检查目标分支、待合并的提交、合并状态及适用检查。使用 GitHub CLI 时，优先通过 `--match-head-commit` 校验已检查的 head，避免合入检查后新增的未知提交。
- 合并完成后核实 PR 状态和目标分支上的 squash commit，再报告成功。
- 父分支已经 squash 合入 main 时，后续分支的 PR 应只包含尚未合入的工作。检查 merge base 和最终 diff，必要时重新整理基线；不要把旧的父分支提交重复当作新增功能。

GitHub CLI 的相关参数为 `--squash`、`--match-head-commit`、`--subject` 和 `--body-file`。正文通过文件传入，保留真实换行；不要把前述示例中的 PR 编号当作当前任务的编号。

## 素材库与引用

- 素材使用 LongMarchAssetsLFS 的 Git LFS 流程，主仓库通过子模组提交引用对应素材版本。
- 大小写文件名冲突要区分命名、分别保存，并同步修改场景中的引用。不能用覆盖或丢弃其中一个文件来消除冲突。
- 修改素材时检查子模组指针与场景资源路径是否一致；不因普通素材更新改写旧 LongMarchAssets 仓库。

## 授权与维护

本 skill 规定做事方式，不额外授予推送、合并或改写已发布历史的权限。当前任务中已有的明确授权继续有效，不重复请求确认；改写已发布历史时确认授权确实涵盖该改写范围，并用带旧提交校验的 `--force-with-lease`。

仓库中的 `.agents/skills/longmarch-project-principles/` 是可共享、可版本管理的规则来源。本机安装在用户技能目录的副本用于自动发现，并在旧分支或素材仓库缺少项目副本时提供这些约定。两者有差异时，以当前项目副本和用户最新要求为准；修改规则时同步本机安装副本。不要把临时分支拓扑、构建目录、PR 编号、凭据或一次性授权写成永久原则。
