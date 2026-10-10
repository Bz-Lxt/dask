# 单机调度器：任务失败后继续完成无关分支（`on_error="continue"`）

## Context

单机一次 `dask.compute(a, b, ...)` 多个共享输入的惰性结果时，三个本地调度器
（`dask/local.py:get_async`，服务于 `get_sync` / `dask.threaded.get` / `dask.multiprocessing.get`）
在 [local.py](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/3a853f85-eaa7-4fe1-8975-d3e085d838b8/repo/dask/local.py#L534-L547) 的主循环里一旦收到任一任务失败，立刻 `raise_exception` 退出：
无关分支不再执行，已完成的结果随缓存一起丢弃，且失败来源只暴露第一个。

目标（可选行为，默认完全不变）：

1. 一个分支出错时，继续把**与之无关**的分支跑完，成功叶子照常返回真实值。
2. 返回值保持 `dask.compute` 输入的嵌套结构；失败叶子放公开标记对象，
   下游未能运行的叶子通过因果链指向**真正失败的任务**（不是"N 个任务失败"）。
3. 线程/进程下共享工作仍是**合并图单次执行**，每个共享任务只做一次。
4. 失败分支独占的数据及时释放；只服务死分支、尚未提交的任务直接跳过。
5. `KeyboardInterrupt` / `SystemExit` 在两种模式下都立即中止；线程池/进程池
   不被破坏，随后的 `compute()` 正常工作。

## API 决策（已替用户决定）

- 新关键字 `on_error: Literal["raise", "continue"] | None = None`。
  - `"raise"`（默认）：现状，首个异常立即抛出。
  - `"continue"`：容忍模式，不抛异常，返回带失败标记的同构结果。
  - `None` 时读配置 `scheduler.on-error`（默认 `"raise"`），非法值 `ValueError`。
- 入口：`get_async`、`get_sync`、`threaded.get`、`multiprocessing.get`（经现有
  `**kwargs` 透传），以及 [dask/base.py:compute](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/3a853f85-eaa7-4fe1-8975-d3e085d838b8/repo/dask/base.py#L685-L771)
  显式增加形参并在 docstring 文档化；`DaskMethodsMixin.compute`（[base.py:368](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/3a853f85-eaa7-4fe1-8975-d3e085d838b8/repo/dask/base.py#L368-L393)）
  自动透传，Delayed/Array/DataFrame 的 `.compute(on_error=...)` 即生效。
- 分布式调度器不支持：`compute` 中当 `on_error` 非 None 且解析出的 scheduler
  不属于 `local.get_sync / local.get_async / threaded.get / multiprocessing.get`
  （含 `partial(get_async, ...)` 的 Executor 形式）时，抛 `NotImplementedError` 明确提示。

## 核心改动：`dask/local.py`

### 1. 失败标记类型（新增公开类，顶层 `dask` 再导出）

```python
class TaskFailure(Exception):
    key            # 该叶子对应的 key
    exception      # 真正抛错的根任务上为异常对象；被剪枝叶子为 None
    traceback      # 根任务的 traceback（进程调度无 tblib 时为字符串）
    caused_by      # 直接上游的 TaskFailure；根失败为 None
    @property
    def root_cause(self) -> TaskFailure   # 沿 caused_by 走到真正失败点
    def reraise(self) -> None             # 用原异常/traceback 重新抛出
    # __repr__ 只渲染 key + exception，不打印 traceback
```

进程调度在父进程构造标记（标记本身不跨进程 pickle）；不得调用 multiprocessing 的
包装版 `reraise`，直接用 `loads` 后的 `(exc, tb)`。异常本身不可 pickle 时沿用
[multiprocessing.py:115-124](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/3a853f85-eaa7-4fe1-8975-d3e085d838b8/repo/dask/multiprocessing.py#L115-L124)
的 fallback（落到标记里的就是序列化失败异常）。

### 2. state 扩展

[start_state_from_dask](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/3a853f85-eaa7-4fe1-8975-d3e085d838b8/repo/dask/local.py#L144-L232)
的 state 增加：`failed: dict[key, (exc, tb)]`、`pruned: dict[key, 直接原因key]`、
`release_after: set`、`skipped: set`。同步更新 [test_local.py:test_start_state](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/3a853f85-eaa7-4fe1-8975-d3e085d838b8/repo/dask/tests/test_local.py#L14-L34) 的期望字典。

### 3. 失败剪枝 + 孤儿释放（新模块函数，核心）

任务 K 失败（仅 `on_error="continue"` 且异常非 `KeyboardInterrupt/SystemExit` 时）：

1. `failed[K] = (exc, tb)`，`running.discard(K)`。
2. 沿 `dependents` BFS：所有传递下游 P 记入 `pruned[P]`（首个根因优先，已在
   `pruned/failed` 中的跳过——覆盖两支失败汇聚的情况），删除 `waiting[P]`。
   不变式：被剪枝的 P 当时只可能在 `waiting`（其依赖 K 未完成则不可能 ready/running）。
3. 统一的"移除消费者"helper：从 `waiting_data[dep]` 删除消费者；集合空且
   `dep not in results` 时，dep 成为孤儿，按状态处理：
   - 在 `cache`：立即 `release_data`（失败分支独占的中间数据及时释放）；
   - 在 `running`：无法中止，记入 `release_after`，完成后释放；
   - 在 `ready`/`waiting`（从未提交）：`skipped.add`，移出 ready 栈、删 `waiting`，
     递归对它的依赖做同样的消费者移除（连带跳过一串无用任务）。
   死分支与活分支共享的依赖因活消费者仍在集合中而保留，之后由正常 `finish_task` 释放。
4. 对 K 自身依赖和 `waiting_data[K]` 也走同一 helper（K 没进 cache，不 release 自身）。

[finish_task](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/3a853f85-eaa7-4fe1-8975-d3e085d838b8/repo/dask/local.py#L290-L317)
两处适配：遍历 dependents 时 `waiting` 已被剪枝删除要跳过
（`s = state["waiting"].get(dep); if s is None or key not in s: continue`）；
结尾若 `key in release_after` 则 `release_data(key)`。

### 4. 主循环

[get_async](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/3a853f85-eaa7-4fe1-8975-d3e085d838b8/repo/dask/local.py#L382-L561)：

- 入口解析 `on_error`（参数优先，否则 `config.get("scheduler.on-error", "raise")`，校验）。
- 失败分支：`exc, tb = loads(res_info)` 后，先判
  `isinstance(exc, (KeyboardInterrupt, SystemExit))` → 任何模式都 `raise_exception`；
  `rerun_exceptions_locally` 在 continue 模式下跳过（该调试路径会在调度器内重跑并抛出，文档说明）；
  `"raise"` 走原路径；`"continue"` 调剪枝函数后 `continue`。
- 循环终止条件不变（剪枝同步清空 waiting/ready/running 计数）。
- 回调契约：`pretask` 对失败任务已触发、`posttask` 不触发；`succeeded = not state["failed"]`。
- 输出组装：新增与 [nested_get](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/3a853f85-eaa7-4fe1-8975-d3e085d838b8/repo/dask/local.py#L320-L336)
  同形的解析器：`cache` 命中→值；`failed`→根标记；`pruned`→沿原因链构造的下游标记
  （按 key memoize，重复 key 返回同一对象；成功的结果 key 受 `not in results` 保护必定在 cache）。
  标量 key 直接返回，嵌套 list 返回嵌套 tuple，与现状一致。`nested_get` 保留。
- `dask.compute` 的 `repack()`（[base.py:587-590](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/3a853f85-eaa7-4fe1-8975-d3e085d838b8/repo/dask/base.py#L587-L590)）
  经 simple_get 把 `results[i]` 原样嵌入容器，标记对象不是 dask collection，原样穿透，嵌套结构天然保留。

### 5. 中断/中止清理（两种模式统一）

- `fire_tasks` 提交的 future 记入 `pending_futs`；done 回调改为先查 `aborted` 标志，
  中止后不再 `queue.put`（避免遗弃的 Queue 被回调长期保活）。
- 主循环整体 `try/except BaseException`：置 `aborted=True`，对 `pending_futs` 逐个
  `.cancel()`（仅取消排队中任务；运行中的线程无法杀），然后 re-raise；`finally`
  保留现有 callback `finish(dsk, state, not succeeded)`。
- 共享线程池（[threaded.py](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/3a853f85-eaa7-4fe1-8975-d3e085d838b8/repo/dask/threaded.py#L94-L124)
的 `default_pool`/pools）绝不关闭，后续 `compute()` 使用全新 queue/cache，不受影响。
- [multiprocessing.get](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/3a853f85-eaa7-4fe1-8975-d3e085d838b8/repo/dask/multiprocessing.py#L234-L252)
的 finally：成功保持 `pool.shutdown()`；异常且 `cleanup=True`（自建池）时
`pool.shutdown(wait=False, cancel_futures=True)`（3.9+ 支持，项目最低 3.10），
不再像现在这样阻塞等运行中任务；用户自带池（`cleanup=False`）一律不关。
- 现有中断测试 [test_threaded.py:~194-202](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/3a853f85-eaa7-4fe1-8975-d3e085d838b8/repo/dask/tests/test_threaded.py#L194-L202) 须继续通过。

## 其他文件

- `dask/threaded.py`：仅 docstring 增补 `on_error`（kwargs 已透传）。
- `dask/multiprocessing.py`：同上 + 池关闭改动。
- `dask/base.py`：`compute` 增加 `on_error=None` 形参、调度器白名单校验、
  转发与 docstring（含示例）。
- `dask/dask.yaml`：新增 `scheduler:\n  on-error: "raise"`；
  `dask/dask-schema.yaml`：对应 `type: string` + `enum: [raise, continue]` + 描述。
- `dask/__init__.py`：导出 `TaskFailure`。

## 测试计划

- [test_local.py](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/3a853f85-eaa7-4fe1-8975-d3e085d838b8/repo/dask/tests/test_local.py)：
  (a) 一支失败、兄弟支返回；(b) 菱形共享依赖 + 计数器断言只执行一次、根标记 key 正确；
  (c) `get_sync` 下 continue；(d) 三级链 + 两支汇聚失败，pruned 标记 `root_cause.key`
  指向真正失败任务；(e) 回调检查 `state["released"]` 含死分支独占中间数据、
  成功结果 key 不在其中；(f) `skipped` 中存在只服务死分支且未提交的任务
  （单 worker + ready 排序构造）；(g) 默认 `"raise"` 行为不变；更新 test_start_state。
- [test_threaded.py](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/3a853f85-eaa7-4fe1-8975-d3e085d838b8/repo/dask/tests/test_threaded.py)：
  continue 模式端到端；扩展中断用例：中止后再次 `get()` 成功；continue 模式下中断仍抛出。
- `dask/tests/test_multiprocessing.py`：continue 端到端 + 标记内容；异常退出不挂池。
- [test_base.py](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/3a853f85-eaa7-4fe1-8975-d3e085d838b8/repo/dask/tests/test_base.py)：
  `dask.compute({"a": ok, "b": [bad, ok]}, on_error="continue")` 嵌套结构与叶子位置；
  `Delayed.compute(on_error=...)`；`dask.config.set({"scheduler.on-error": "continue"})`
  全局生效；非法值 ValueError；分布式/非本地 scheduler 下 NotImplementedError。

## 验证

```bash
pixi run test dask/tests/test_local.py
pixi run test dask/tests/test_threaded.py
pixi run test dask/tests/test_multiprocessing.py
pixi run test dask/tests/test_base.py
pixi run lint lint
```

手工冒烟（threads 与 processes 各一次）：两个 delayed 共享同一计数任务，
其中一支 `raise`，`dask.compute({"ok": a, "bad": [b, a]}, scheduler=..., on_error="continue")`，
断言：共享任务计数为 1、`ok` 为真实值、`bad[0]` 是 `TaskFailure` 且
`root_cause.key` 为真正失败 key、`bad[1]` 为真实值。
