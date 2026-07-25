---
title: MS-Swift GRPO Pipeline Walkthrough
date: 2026-06-16 10:00:00 +0800
categories: [Artificial Intelligence, Reinforcement Learning]
tags: [Tech, AI, RL, LLM, MS-Swift, GRPO]
math: True
description: 基于源码 walkthrough 拆解 ms-swift 的 GRPO 训练管线：项目目录地图、调用链、scheduler/turn/trajectory/rollout 概念、训练前向与数据流、逐文件骨架，以及如何修改 forward pass、loss 与梯度并完成注册。
---

> 本文基于 ms-swift v4.4.0-dev 源码，逐层 trace GRPO 训练管线的逻辑流。由 Claude Code (Opus 4.8, ultracode 多智能体编排) 整理，作者审校。
{: .prompt-info }

## 预备知识：GRPO 三分钟

如果已熟悉 GRPO，可直接跳到下一节的项目目录地图。

**RLHF 的基本局。** 一个已经 SFT 过的模型（policy）对 prompt 生成回答，一个打分机制（reward）评判好坏，RL 算法据此更新权重，让高分回答里的 token 概率升高、低分的降低。

**从 PPO 到 GRPO 一句话。** PPO 需要额外训练一个 critic（value network）来估计 baseline；GRPO（Group Relative Policy Optimization，DeepSeekMath, Shao et al., 2024）把 critic 省掉：同一个 prompt 采样一组（group，例如 8 个）回答，用**组内平均 reward** 当 baseline，每个回答的 advantage = (自己的 reward − 组平均) / 组标准差。比组平均好的被推高，比组平均差的被压低。

**训练一个 step 到底发生什么**（这也是全文要逐层拆的东西）：

1. 从 dataset 抽一批 prompt；
2. **rollout**：用当前模型给每个 prompt 生成 `num_generations` 个回答（用 vLLM 加速，是 wall-clock 的大头）；
3. **reward**：给每个回答打分；
4. **advantage**：组内归一化；
5. **loss**：PPO-clip 公式（重要性比率 × advantage，截断，加 KL 惩罚）；
6. backward + `optimizer.step()`，权重变了，回到 1（新权重还要同步给 vLLM）。

对 GRPO/PPO/DPO 等算法本身的系统推导，见本博客 [LLM RL Algorithms](/posts/LLM-RL-Algorithms/) 一篇；本文专注另一件事：**这套流程在 ms-swift 代码里长什么样、每一层的职责边界在哪、想定制该从哪里下手**。

**本文的行进路线**：先给**项目目录地图**（空间图：代码在仓库的什么位置、哪些文件值得读、哪些可以无视），再给**调用链总览**（时间线：一个训练 step 从 shell 到 loss 依次经过谁）；然后把概念说死，从 turn / trajectory / rollout / scheduler 的区别讲起；接着依次讲训练侧的 forward pass、采样数据存在哪里、训练的完整流程；再用逐文件骨架把整体结构钉死（类与函数签名、行号、注册表都是真实源码，函数体替换成注释）；接着是动手部分：怎么改 forward pass、怎么改学习算法与手动改梯度、怎么注册；末尾用 Plugin 机制与 vLLM 数据流两节补齐参考细节，并以魔改速查表和启动命令收束。

**四种读法**：

- 只想先跑起来：直接翻到文末“启动命令完整示例”，先跑最小示例；
- 想看懂框架逻辑：顺读“项目目录地图”“一图概览”与“上面那些名字到底是什么”三节，即可建立全图；
- 想搞懂训练机理：读“forward pass”“采样出来的数据存在哪里”“训练的流程”三节；
- 想魔改：直奔“改 forward pass”与“改学习算法与手动改梯度”两节，再用“魔改指南”速查表按需定位。

---

## 项目目录地图：这些代码在仓库哪里

先建立**空间图**：要读的代码在 `ms-swift` 仓库的什么位置。顶层 `swift/` 下共 28 个包，与 GRPO 相关的只有 8 个（★ 标注），其余可以放心无视：

```
swift/                                  # ms-swift 主包（★ = GRPO 主线必读，共 8 个）
├── agent_template/    # Agent 工具调用格式模板（ReAct、Hermes 等）（GRPO 无关，可忽略）
├── arguments/         # ★ 命令行参数定义，rlhf_args.py 中的 RLHFArguments 是 GRPO 参数入口
├── callbacks/         # Trainer 回调（early_stop、LISA 等）（GRPO 无关，可忽略）
├── cli/               # ★ 命令行入口，swift rlhf / swift rollout 两条命令从这里进
├── config/            # DeepSpeed ZeRO / FSDP 的 JSON 配置模板（纯配置文件，可忽略）
├── dataloader/        # 分布式 DataLoader 分片与调度（通用基建，可忽略）
├── dataset/           # 数据集注册、加载与预处理（通用基建，GRPO 直接复用，无需细读）
├── hub/               # ModelScope / HuggingFace 模型下载上传封装（GRPO 无关，可忽略）
├── infer_engine/      # ★ 推理引擎封装，vllm_engine.py / grpo_vllm_engine.py 负责 rollout 采样
├── loss/              # SFT 各类 loss 实现，GRPO 的 loss 在 trainer 内部（GRPO 无关，可忽略）
├── loss_scale/        # Agent 训练的 token 级 loss 加权（GRPO 无关，可忽略）
├── megatron/          # Megatron 并行训练栈（GRPO 无关，可忽略）
├── metrics/           # 训练与评测指标（acc、nlg 等）（GRPO 无关，可忽略）
├── model/             # 模型注册表、model_meta 与加载 patch（通用基建，无需细读）
├── optimizers/        # GaLore、Muon 等特殊优化器（GRPO 无关，可忽略）
├── pipelines/         # ★ 训练/推理流水线，train/rlhf.py 的 rlhf_main 是 GRPO 主流程
├── ray/               # Ray 分布式的 Megatron 适配（GRPO 无关，可忽略）
├── ray_utils/         # Ray 多机资源管理 RayHelper（单机 GRPO 用不到，可忽略）
├── rewards/           # ★ 内置奖励函数：orm.py 结果奖励、prm.py 过程奖励、rm_plugin.py 奖励模型插件
├── rlhf_trainers/     # ★ 核心所在：grpo_trainer.py 与 rollout_mixin.py 都在这个包里
├── rollout/           # ★ 多轮 rollout 调度器（MultiTurnScheduler）与 Gym 环境接口
├── sequence_parallel/ # Ulysses / Ring-Attention 序列并行（GRPO 无关，可忽略）
├── template/          # 各模型 chat template 的编码实现（通用基建，无需细读）
├── trainers/          # ★ trainer_factory.py 按 rlhf_type='grpo' 分发出 GRPOTrainer
├── tuner_plugin/      # tuner 插件桥接层（LoRA plumbing，可忽略）
├── tuners/            # LoRA 及各类参数高效微调实现（GRPO 无关，可忽略）
├── ui/                # Gradio Web UI（含 llm_grpo 页面，只是壳，可忽略）
└── utils/             # 日志、环境变量、IO 等通用工具（无需细读）
```

把 ★ 的包展开，GRPO 主线真正需要认识的文件如下（行数为 v4.4.0-dev 实测）：

```
swift/
├── cli/
│   ├── rlhf.py                #    7 行｜训练入口：只做一件事，调 swift.pipelines.rlhf_main()
│   └── rollout.py             #    5 行｜rollout server 入口（server mode 时单独起）
├── pipelines/train/
│   ├── __init__.py            #    5 行｜导出 SwiftRLHF 与 rlhf_main
│   ├── rlhf.py                #  246 行｜SwiftRLHF(SwiftSft)：准备策略/ref/reward 模型与 template
│   └── sft.py                 #  340 行｜父类主循环，L175 经 TrainerFactory 拿到 GRPOTrainer 并启动训练
├── arguments/
│   └── rlhf_args.py           #  619 行｜L170 RLHFArguments(TeacherModelArguments, GRPOArguments, PPOArguments, …)
├── trainers/
│   └── trainer_factory.py     #   73 行｜TRAINER_MAPPING['grpo'] → swift.rlhf_trainers.GRPOTrainer
├── rlhf_trainers/
│   ├── args_mixin.py          #  464 行｜GRPOArgumentsMixin / VllmArguments：GRPO 全部超参定义在此
│   ├── arguments.py           #  123 行｜L97 GRPOConfig(GRPOArgumentsMixin, TrainArgumentsMixin, HfGRPOConfig)
│   ├── grpo_trainer.py        # 2732 行｜★ GRPOTrainer：奖励计算、优势归一化、GRPO loss 与训练步
│   ├── rollout_mixin.py       # 1654 行｜★ RolloutTrainerMixin：采样调度、colocate/server 两种模式、权重同步
│   ├── rlhf_mixin.py          #  182 行｜RLHFTrainerMixin：各 RLHF trainer 的公共基类
│   ├── vllm_client.py         #  536 行｜VLLMClient：server mode 下与外部 rollout server 通信
│   └── utils.py               # 2016 行｜权重打包传输等杂项工具（按需查阅，不必通读）
├── rollout/
│   ├── multi_turn.py          #  832 行｜RolloutScheduler / MultiTurnScheduler：多轮采样调度基类与内置实现
│   ├── gym_env.py             #  127 行｜Env 抽象基类 + envs 注册表（gym 式环境交互）
│   └── agent_loop.py          #  252 行｜run_multi_turn：多轮循环的实际执行函数
├── rewards/
│   ├── orm.py                 #  464 行｜结果奖励：MathAccuracy、Format 等 + orms 注册表
│   ├── prm.py                 #  156 行｜过程奖励 PRM + prms 注册表
│   └── rm_plugin.py           #  233 行｜奖励模型插件：DefaultRMPlugin / GenRMPlugin
└── infer_engine/
    ├── infer_engine.py        #  314 行｜InferEngine 公共基类
    ├── vllm_engine.py         #  908 行｜VllmEngine：vLLM 推理封装，colocate mode 的采样后端
    ├── grpo_vllm_engine.py    #  147 行｜GRPOVllmEngine(VllmEngine)：GRPO 专用采样引擎
    └── …                      #  sglang / lmdeploy / transformers 等其他后端（GRPO 无关，略）
examples/train/grpo/plugin/
└── plugin.py                  # 1228 行｜自定义 reward / scheduler / Env 的官方范本，外部插件照此注册
```

空间图建好了。下一节调用链的每一层，在这张图里都有唯一的文件归属；后文“整体结构”会把其中最重要的几个文件直接贴出来。

---

## 一图概览整个调用链

有了空间图，再看**时间线**：一个训练 step 从 shell 到 loss 依次经过谁。

```
[shell]  swift rlhf --rlhf_type grpo --model X --reward_funcs Y --multi_turn_scheduler Z
   │
   ▼
[CLI entry]  swift/cli/rlhf.py
   │   from swift.pipelines import rlhf_main
   │   rlhf_main()
   ▼
[Pipeline]  swift/pipelines/train/rlhf.py
   │   class SwiftRLHF(SwiftSft):           ← 继承 SFT pipeline
   │      args_class = RLHFArguments
   │      _prepare_single_model()           ← 加载 actor / ref / reward model
   │      run() → trainer.train()
   ▼
[Factory]  swift/trainers/trainer_factory.py
   │   TrainerFactory.get_trainer_cls(args)
   │      'grpo' → swift.rlhf_trainers.GRPOTrainer
   ▼
[Trainer]  swift/rlhf_trainers/grpo_trainer.py  (2732 行)
   │   class GRPOTrainer(RolloutTrainerMixin, SwiftMixin, HFGRPOTrainer):
   │      __init__:
   │          prepare_rollout()            ← MultiTurnScheduler 初始化在这里
   │          _prepare_rewards()
   │      train()  (继承 HF Trainer.train)
   │          for step in steps:
   │              inputs = sampler.next()
   │              inputs = _prepare_inputs(inputs)   (实际由 training_step 内部调用)
   │                  └─ _generate_and_score_completions()  ← ★ 自定义 rollout 入口
   │                       ├─ _generate_completions()         (调 MultiTurnScheduler)
   │                       ├─ _score_completions()           (reward 函数)
   │                       ├─ _prepare_batch_inputs()        (tokenize, mask)
   │                       └─ _compute_advantages()          (GRPO advantage)
   │              loss = training_step(model, inputs)
   │                  └─ compute_loss()
   │                       └─ _compute_loss()
   │                            └─ _compute_loss_and_metrics()  ← ★ 自定义 loss override 点
   │              optimizer.step()
   ▼
[Rollout]  swift/rollout/multi_turn.py + swift/rlhf_trainers/rollout_mixin.py
   │   MultiTurnScheduler.run() → for each turn:
   │      infer_engine.infer_async(request) → vLLM/SGLang/Transformers
   │      step() → next request
   │      check_finished() → terminate?
   ▼
[Engine]  swift/infer_engine/vllm_engine.py
       支持 guided decoding (structured_outputs_regex → vLLM structured outputs)
```

---

## 上面那些名字到底是什么（概念 + 代码职责）

如果你只看上面那张图会觉得每个方框只是一个英文标签。这一节把每个名字**当代码里的一个角色 / 一个 class / 一个文件**来讲：它**拥有什么状态**、**对外提供什么方法**、**为什么必须独立存在而不合并到上下游**。叙述从 rollout 侧的概念区别（turn / trajectory / rollout / scheduler）讲起，再到引擎与装配组件，最后落到 loss 的三大原料。

### Turn、Trajectory、Rollout：把三个词说死（先看例子再下定义）

这三个词贯穿全文，但都不自明。先看一个具体例子，再回头给定义。

**例子：一条“带计算器的数学题”轨迹。** 设 prompt 是“计算 (3+4)×5”，我们给模型配了一个计算器工具（对应后文 plugin 一节里真实存在的 `ToolCallScheduler`，它用 ReAct 文本格式表达工具调用）。rollout 过程中 `messages` 里始终只有两条消息，其中 assistant 那条的内容跨 turn 不断生长：

```python
messages = [
  {'role': 'user', 'content': '计算 (3+4)×5，可以使用计算器工具'},   # 初始 prompt，来自 dataset
  {'role': 'assistant', 'content':
      '先算括号。Action: calculator\nAction Input: 3+4\n'  # 第 1 个 turn：模型生成 → 这段 token 参与 loss
    + '7\n'                                                # scheduler 执行计算器后拼进来 → loss_mask=0
    + 'Action: calculator\nAction Input: 7*5\n'            # 第 2 个 turn：模型续写 → 参与 loss
    + '35\n'                                               # scheduler 拼进来 → loss_mask=0
    + '所以答案是 35。'},                                   # 第 3 个 turn：解析不到新的工具调用 → 这局结束
]
```

注意两点。其一，工具结果和模型输出**混在同一条 assistant 消息里**，文本层面看不出谁写的哪段；真正的边界在 token 层面，由 scheduler 逐轮累积的 `response_loss_mask` 划定：模型生成的 token 记 1，拼进来的工具结果记 0。其二，“怎么把环境反馈塞回对话”本身是 scheduler 的自由：这里选择续写 assistant 消息，`GYMScheduler` 则把环境观测作为新的 user 消息 append。

对着这个例子，三个词就可以说死了：

- **Turn（轮）**：**对 inference engine 的一次生成调用**，产出一段 assistant 文本（第一轮新建 assistant 消息，之后的轮在这个例子里直接续写同一条消息）。上例有 3 个 turn。它不是聊天界面意义上的“一问一答”：整个过程没有真人参与，夹在中间的工具结果是 scheduler 代表环境拼进去的。单轮任务（纯数学题、不用工具）每条轨迹恰好 1 个 turn：模型一口气写完 `<think>…</think><answer>…</answer>` 就结束，此时不需要 scheduler。
- **Trajectory（轨迹）**：**从初始 prompt 到终止判定为止的完整一局**，即上面 messages 的最终形态，连同逐轮累积的 token_ids / loss_mask / logprobs（打包成 `RolloutOutput`）。RL 术语里这就是一个 episode。一个 prompt 配 `num_generations=8` 时会**独立跑出 8 条不同的轨迹**：采样有随机性，8 条的对错、长短、轮数各不相同，这正是 GRPO 组内比较的原料。
- **Rollout**：**“让当前策略把轨迹跑出来”这个动作**（名词化后也指跑出来的那批数据）。词源即经典 RL 的 roll out a policy：把策略往前“滚”，滚出 (state, action, reward) 序列。SFT 不需要这一步，因为它的训练数据是人预先写好的；RL 的训练数据**由当前策略自己生成**，权重一变、数据分布就变，所以每隔几步就得重新 rollout。

和经典 RL 的对应关系，一张表看完：

| LLM RL 术语 | 经典 RL 对应 | ms-swift 里的载体 |
| --- | --- | --- |
| prompt | 初始状态 $s\_0$ | dataset 一行的 `messages` |
| 一个 turn | 一个宏动作 + 一次环境转移 | 一次 `infer_async` 调用 + `scheduler.step()` |
| trajectory | 一个 episode | 跑完的 `messages` + `RolloutOutput` |
| rollout | 用当前策略采 episode | `_generate_and_score_completions` 的第①步 |
| 环境（environment） | `env.step` 与终止判定 | scheduler 的 `step()` / `check_finished()` |
| reward | episode 末端的回报 | 轨迹结束后 reward 函数打分 |
| 策略 π | 动作分布 | 训练中的 LLM 本体（逐 token 的条件分布） |

注意一个粒度错位：**交互按 turn 进行，优化按 token 进行**。loss 里的重要性比、KL、advantage 广播全落在 per-token 粒度上（advantage 是轨迹级标量，广播给该轨迹的每个未被 mask 的 token）；turn 只是数据生产阶段的组织单位，进了 loss 之后只剩“哪些 token 参与、哪些被 mask”之分。

- **代码里在哪**：
  - `GRPOTrainer._generate_and_score_completions()` 是高阶 entry，负责“采一批 + 算 reward + 算 advantage”
  - `_generate_completions()` 负责真正调 inference
  - `RolloutTrainerMixin._fast_infer()` / `_server_rollout()` / `_colocate_rollout()` 负责把 rollout 请求分发给 inference engine
  - `MultiTurnScheduler.run()` 负责“一条 trajectory 内部多个 turn 怎么串”
- **为什么 rollout 独立于 trainer 主循环**：rollout 用的是**当前 model 权重的快照**，可以批量做、可以放到不同 GPU、可以异步流水线。把它从 backward 路径里分离出来，让两边的并行度独立调优（rollout 用 vLLM tensor parallel，training 用 DDP/FSDP）。
- **类比**：拍电影里的“先拍一段素材，再剪辑”。Rollout 是拍素材，loss/backward 是剪辑。

### MultiTurnScheduler（scheduler 到底是什么：环境 + 回合裁判）

- **一句话**：scheduler 就是 RL 里的**环境（environment）加上 episode 控制器**。回看上面计算器的例子：“真的去执行 3+4、把结果 `7` 拼回那条 assistant 消息、决定还要不要再来一轮”，这些事**模型不干（它只生成文本）、trainer 也不干（它只管训练循环）**，全是 scheduler 干的。凡是“生成之外的回合逻辑”，都归它。
- **它在一条 trajectory 里被调用的四个位置**（对照例子）：
  1. 开局：`on_trajectory_start()`，可修改初始 request（比如 gym 场景注入环境初始观测）；
  2. 每个 turn 之后：`check_finished()` 判断这局是否结束（上例的 `ToolCallScheduler` 在回复里解析不到新的工具调用就停；默认实现看 `finish_reason='length'` 或轮数达到 `max_turns`）；
  3. 若不停：`step()` 造出下一轮的 request：上例中它解析 `Action:` 行、执行计算器、把 `7` 拼进 assistant 消息，并把这些非模型生成的 token 的 loss_mask 置 0；
  4. 终局：把逐轮累积的 token_ids / loss_mask / logprobs 打包成 `RolloutOutput` 交还 trainer。
- **拥有什么状态**：`self.infer_engine`（指向下文要讲的 engine）、`self.max_turns`、`self.tokenizer`。**注意**：每条 trajectory 的状态（messages 历史、累计 token_ids、当前 turn 计数）放在 `run()` 的局部变量里，不放 instance attribute，因为同一个 scheduler 实例会被并发跑多条 trajectory（async）。
- **对外提供的方法**：
  - `run(request, cfg)` → 跑完整条 trajectory，返回 `RolloutOutput`。**默认实现**是一个 while 循环（后文有逐行骨架）；想完全自定义就 override 它。
  - `step(request, response, turn)` → 给出“下一轮的 request 长什么样”。基类不提供实现，子类必须写。
  - `check_finished(request, response, turn)` → 是否应该停。
- **为什么独立于 trainer**：HF Trainer 假设“一个 input → 一次 generate → 一个 output”（单轮）。多轮场景（tool use、agent 任务、多角色对话等）需要在 generate 之间**重新构造 prompt**、追加 history、执行环境逻辑。这段逻辑跟 trainer 无关、跟具体任务有关，所以单独拎成一个可插拔的 class；不配 scheduler 时管线退化为单轮（一条轨迹恰好一个 turn），此时它完全不出场。
- **类比**：桌游的**裁判**。玩家（model）只负责出牌（生成文本）；裁判（scheduler）负责翻开下一张公共牌（工具结果、环境反馈）、判定这一局是否结束、最后把牌局记录（`RolloutOutput`）交给赛事组织方（trainer）。engine 只是玩家的手，负责把“出牌”这个动作物理地执行出来。

### Engine（推理后端）

- **是什么**：负责“把一组 prompt → 一组 completion”这件事的执行器。可换实现：`VllmEngine` / `SglangEngine` / `LmdeployEngine` / `TransformersEngine`（HF generate 的薄包装）。
- **拥有什么状态**：加载的 model 权重副本（vLLM 内部有自己的 KV cache + 调度器）、tokenizer、采样参数模板、HTTP 端口（server 模式）。
- **对外提供的方法**：`infer(requests, request_config)` 同步版、`infer_async(...)` 异步版、`_batch_infer_stream(...)` 流式 batch。
- **为什么独立于 trainer**：trainer 不关心是 vLLM 还是 SGLang 在跑 inference；engine 也不关心调用方是 GRPO 还是 PPO。一个清晰的 `infer_async(request) → response` 接口让两边都能换。**vLLM colocate 模式**下 engine 和 trainer 在同一进程同一 GPU；**vLLM server 模式**下 engine 是独立进程，trainer 通过 HTTP 调它（`VllmClient`）。
- **类比**：相机机身 vs 镜头。导演（trainer）下“拍这帧”指令；不同的镜头（engine）能拍但成像方式不同。

### vLLM（Engine 的主流具体实现）

- **是什么**：一个专门**把 LLM 生成文本跑到极快吞吐**的开源推理引擎。它就是上面 Engine 那个接口最常用的具体实现（ms-swift 默认 rollout 引擎）。它不参与 backward / optimizer.step，只干一件事：**给一批 prompt，以最高吞吐吐出 completion**。
- **它解决什么问题：朴素 generation 太慢**。LLM 是 autoregressive（一次一个 token），HF 原生 `model.generate()` 有两个致命浪费：
  1. **KV cache 碎片**：每个请求要预留最大长度的连续显存放 KV cache，实际生成短就大量浪费，利用率常只有 20-40%。
  2. **静态批处理干等**：一批里要等最慢的请求生成完才能换下一批，短请求算完后 GPU 干等。
- **vLLM 的两个关键技术（为什么快 10-20 倍）**：
  1. **PagedAttention**（招牌）：把 KV cache **像操作系统分页那样**切成不连续的小 block，每请求一张 block table（类比页表），用多少分配多少，碎片浪费降到 <4%，同显存能塞多得多并发请求。名字里的 v 就是 virtual memory（SOSP 2023, Kwon et al., UC Berkeley）。
  2. **Continuous batching**（连续批处理）：iteration-level 调度，谁生成完立刻踢出、新请求立刻补位，GPU 永远满载，没有干等。
- **在 RL 训练里的作用：rollout 引擎**。RL 每个训练 step 都要用当前 policy 重新采样生成，这个 generation 是 RL 训练**头号 wall-clock 瓶颈**（常占 60-80% 时间）。用 HF generate 慢到没法实用；ms-swift 接 vLLM 后 rollout 快 10-20 倍，RL 训练才实用。代码里它藏在 scheduler 的 `run()` 里的 `self.infer_engine.infer_async(request, cfg)` 背后。
- **RL 特有的麻烦：权重同步**（值得专门知道）。vLLM 持有**自己一份 model 权重**（要做自己的显存布局 / 量化 / 并行），但训练每个 step 都改权重，所以每 step 之后必须把**新权重推进 vLLM** 重新同步。ms-swift 的 `rollout_mixin.py:_move_model_to_vllm()` / `_load_state_dict_to_vllm()` 干这件事。这也是 RL 框架接 vLLM 比纯部署接 vLLM 难得多的原因。
- **colocate vs server**：
  - **colocate**：vLLM 和训练同进程同 GPU，权重进程内直接拷，省 GPU 但 rollout 和 training 不能重叠。
  - **server**：vLLM 是独立进程（`swift rollout` 起一个 vLLM server），训练经 HTTP 调它（`VllmClient`），权重走网络推。支持异步重叠 + dynamic 多轮。
- **guided decoding（结构化输出）**。vLLM 内置 structured outputs（后端用 XGrammar / Outlines）：给一个 JSON schema，它就用 **logit masking** 在每步采样屏蔽掉会让输出不合法的 token id，**强制保证**输出是合法 JSON。需要模型严格输出某个结构（如固定字段的 JSON）时，这比靠 retry 兜底干净得多，也省 token。
- **事实边界**：vLLM 开源（Apache-2.0，GitHub `vllm-project/vllm`），社区维护。除上述外还有 tensor parallelism、prefix caching、speculative decoding、多种量化、OpenAI-compatible server 等。
- **类比**：vLLM 是 Engine 这个“相机机身接口”的一款高性能具体型号；SGLang 是另一款，Transformers 是慢的兜底款。

### Tokenizer（文字 ↔ 数字的翻译器，token 世界的地基）

- **是什么**：一个把**字符串和整数 ID 序列双向翻译**的对象。神经网络只会算数字，不认字；tokenizer 就是文字和数字之间唯一的桥。每个 model 配套自己的一个 tokenizer，**不能跟别的 model 混用**（像配套的密码本）。
- **核心机制：subword（子词）切分**。它既不是按字符切，也不是按单词切，而是按“高频子词”切（BPE / WordPiece / Unigram / SentencePiece 等算法）。比如 `"unbelievable"` 可能被切成 `["un", "believ", "able"]` 三个 token，`"42"` 可能是一个 token，中文一个字常常是一到几个 token。背后有一张固定大小的**词表（vocabulary）**，每个 token 对应一个整数 ID（Qwen3 词表约 15 万）。
  - 为什么用 subword 而不是整词：词表能保持可控大小，又能拼出没见过的词（拆成已知子词），还天然多语言友好。
- **对外提供的两个方法**：
  - `encode(text) -> List[int]`：文字变 token id 序列（喂给模型 forward 之前必做）
  - `decode(List[int]) -> text`：token id 序列变回文字（模型吐 token 后给人看）
- **拥有什么状态**：基本是只读的纯函数式对象（词表 + 切分规则固定）。被 template / engine / trainer / scheduler **共享同一个实例**。它属于 model 的一部分。
- **为什么贯穿所有环节（这才是关键）**：本文后面反复出现的 “token” 单位全是 tokenizer 切出来的：
  - 模型 forward 吃的是 token id，不是文字 → 所有 prompt / completion 都先过 `encode`
  - `per_token_logps` 的 “per_token” 就是 per 这个切分单位，每个 token id 一个 log prob
  - `completion_mask` 标记哪些 token id 参与 loss，单位也是 token
  - scheduler 返回的 `response_token_ids` 就是 `encode` 出来的整数序列
  - vLLM 的 guided decoding 做 logit mask 也是在 token 粒度上屏蔽不合法 token id
- **一个微妙但重要的陷阱**：`encode(decode(ids))` **不保证等于** `ids`（编解码不对称，因为多种 token 切法可能 decode 成同一串文字）。所以 ms-swift 的 scheduler **优先直接累积并返回 `response_token_ids`**，而不是让 trainer 把文本重新 `encode` 一遍，因为后者可能切出不同的 token 边界，导致训练时 log prob 对不齐。这也是自定义 scheduler 应该在每轮逐步累积 `token_ids` 而不是事后重新编码的原因。
- **字符位置与 token 位置的换算**：有时需要知道“原文里某段字符对应第几到第几个 token”（比如定位某个答案子串对应的 token、或做可视化对齐），这要用 tokenizer 的 **offset mapping**（每个 token 对应原文哪几个字符）来反查；靠数字符是数不准的，因为 token 边界与字符边界并不对齐。
- **类比**：电报的摩斯密码本。收发双方必须用**同一本**密码本，否则译出来是乱码；model 和它的 tokenizer 就是这种绑定关系。

### CLI（命令行入口）

- **是什么**：一个极薄的 shell-to-Python 适配层。`swift/cli/rlhf.py` 全文 7 行，作用是 `python -m swift.cli.rlhf args...` 时调用 `swift.pipelines.rlhf_main()`（命令行参数在 rlhf_main 内部由 RLHFArguments 从 sys.argv 解析）。
- **拥有什么状态**：基本没有（无状态）；只负责“安装命令、设置好 fork-safe 单设备模式、调 main”。
- **为什么独立**：把 `setup.py` 的 entry point 和真正的训练逻辑解耦。后者可以独立用 Python 直接调（不走 shell）。
- **类比**：Linux 的 `/usr/bin/git` 命令本身只是个 dispatcher，真正的逻辑在 `libgit2`。

### Pipeline（流程编排器）

- **是什么**：一个高阶的“训练任务跑通从头到尾要走哪几步”对象。`SwiftRLHF` 继承自 `SwiftSft`，一个 instance 对应一次训练任务。
- **拥有什么状态**：解析过的 `args` (RLHFArguments)、加载好的 `model` / `ref_model` / `reward_model` / `tokenizer` / `template` / `dataset`。
- **对外提供的方法**：`run()`，里面顺序做这些事：(1) 下载/加载模型权重；(2) 加载 dataset；(3) 拼装 template；(4) 调 `TrainerFactory` 选 trainer 类；(5) 把上面准备好的东西塞进 trainer 构造函数；(6) `trainer.train()`。
- **为什么独立**：训练任务的“准备阶段”（model/data/template 加载）和“训练循环本身”（forward/backward）是两件事。把准备阶段放在 Pipeline 里，trainer 就可以专注训练循环；新增一种 RL 算法只需要写新 trainer，不用重复实现 model 加载。
- **类比**：电影制片厂的“项目经理”角色：选演员、定场地、定预算，然后把开机交给导演（trainer）。

### Factory（工厂）

- **是什么**：一个查表函数，输入是 args（含 `--rlhf_type grpo`），输出是**要 instantiate 的 trainer 类**（不是 instance，是 class 对象）。
- **拥有什么状态**：一个硬编码的字典 `TRAINER_MAPPING = {'grpo': 'swift.rlhf_trainers.GRPOTrainer', 'dpo': '...', 'ppo': '...', ...}`。
- **对外提供的方法**：`TrainerFactory.get_trainer_cls(args)` → 反射 import 对应 class。`get_training_args(args)` 同理返回 trainer 专属的 config dataclass。
- **为什么独立**：一个 CLI (`swift rlhf`) 要支持很多 RL 算法（DPO/PPO/GRPO/KTO/CPO/ORPO/RM/GKD），每个算法用不同 trainer 类。Factory 把“`--rlhf_type` 字符串 → trainer 类”的映射集中在一处，避免 Pipeline 里写一堆 `if args.rlhf_type == 'grpo': ...`。
- **类比**：编程语言里的 design pattern “Factory Method”，给名字、返回对象。

### Trainer（训练循环 owner）

- **是什么**：训练循环本身的载体。`GRPOTrainer` 继承自 `RolloutTrainerMixin`（rollout 基建）、`SwiftMixin` 与 TRL 的 `GRPOTrainer` 基类（后者继承 HF `Trainer`）。一个 instance 对应一次训练任务。
- **拥有什么状态**：`self.model`、`self.ref_model`、`self.reward_funcs`、`self.optimizer`、`self.lr_scheduler`、`self.train_dataloader`、`self.template`、`self.multi_turn_scheduler`、`self.infer_engine`、`self.accelerator`（DDP / FSDP / DeepSpeed 包装器）、当前 step 计数、buffered rollout 数据 …… 几乎所有训练相关的状态都在这。
- **对外提供的方法**：
  - `train()` → 跑整个 epoch loop，是顶层入口
  - `_prepare_inputs(batch)` → 把 dataset 里的原始 batch 变成 forward 能用的 tensor（GRPO 里这一步会触发 rollout + reward + advantage 计算）
  - `compute_loss(model, inputs)` → 算 loss，HF Trainer 通过这个 hook 调子类
  - `_compute_loss_and_metrics(model, inputs)` → GRPO 自家加的，真正写 GRPO loss 公式的地方
  - `training_step()` / `optimizer_step()` → 继承自 HF，标准 backward+step
- **为什么独立于 Pipeline**：Pipeline 是“装东西的盒子”，Trainer 是“开机后跑循环的状态机”。一个 Pipeline run 里只 instantiate 一个 Trainer，但同样一个 Trainer 类可以被 Pipeline、Jupyter notebook、Ray actor、单测脚本反复 instantiate。
- **类比**：导演 + 整个剧组 + 剧本 + 摄影机的合体。Pipeline 把人和设备凑齐交给它后，喊“Action”开始拍。

### Reward function（奖励函数 / Reward Model）

- **是什么**：一个 callable 或 nn.Module，输入 completion 字符串（和可选的 dataset 列），输出 float reward。自定义 reward 通常继承 `ORM`。
- **为什么必须独立**：reward 的定义和算法（GRPO/PPO/…）正交。同一个 reward 函数（如 “math 答案是否正确”）可被任意 RL 算法复用；同一个 RL 算法可以挂任意 reward。
- **数据流上的位置**：rollout 完成之后、advantage 计算之前。`_score_completions()` 调用 reward function，得到 `rewards_per_func` tensor。
- **类比**：考试的评卷标准。学生（policy）怎么写题是模型自己事；评卷规则（reward）独立于“用什么教学方法（RL 算法）”。

### Advantage（优势 = 归一化后的 reward）

- **是什么**：`advantage = reward - baseline`，作为 policy gradient 公式里 $\nabla \log p \cdot \text{advantage}$ 的标量权重。GRPO 的特色是**用 group mean 当 baseline**：同 prompt 跑 `num_generations` 个 rollout，组内归一化。这样省掉了 PPO 需要的额外 critic network。
- **为什么必须独立**：reward 的绝对值意义模糊（10 分还是 100 分？）、variance 大；归一化后才是稳定的梯度信号。
- **代码里在哪**：`GRPOTrainer._compute_advantages()`，输出 `[batch_size]` 的 tensor 塞进 `inputs['advantages']`。

### per_token_logps / completion_mask / KL（loss 公式的三大原料）

- **per_token_logps**：`[batch_size, seq_len]`，每个 token 在当前 model 下的 log probability。是 forward 跑出来的，loss 公式的核心输入。
- **completion_mask**：`[batch_size, seq_len]`，01 mask，标记哪些 token 是“模型生成的、应该参与 loss”（completion 部分）vs “prompt / system message 部分，不参与 loss”。多轮 / tool-use 场景里“把环境返回的内容排除在 loss 之外”，最终也是落到这一层的 mask 上生效（scheduler 返回的 `response_loss_mask` 会合并进来）。
- **ref_per_token_logps**：reference model（冻结的 policy，通常是 SFT 模型）算的 log prob，用来算 KL penalty（防止 RL 偏离原 model 太远）。
- 这三者一起进 `_compute_loss_and_metrics`，组装出 PPO-clip loss + KL penalty。

### 那张图重读一遍（带上面的概念）

```
shell → CLI（命令翻译）
        → Pipeline（"把这次训练任务的所有装备凑齐"）
            → Factory（"按 args 挑出该用哪个 Trainer 类"）
                → Trainer instance（训练循环的状态机 + 拥有 model/optim/engine/scheduler）
                    → train() 大循环：
                          - sampler 出 batch
                          - _prepare_inputs 触发 _generate_and_score_completions
                                  - _generate_completions 让 engine 经 scheduler 跑 rollout
                                          (scheduler 是裁判，engine 是玩家的手)
                                  - _score_completions 让 reward function 评分
                                  - _compute_advantages 把 reward 变成 advantage
                          - compute_loss → _compute_loss_and_metrics
                                  组装 per_token_logps × advantage 出 loss
                          - backward + optimizer.step
```

每个方框都是一个**有明确状态和接口的 class/模块**，可以独立替换：
- 换 reward？写新 ORM，扔进 `orms` registry
- 换 rollout 模式？换 scheduler 子类
- 换 inference backend？换 engine
- 换 RL 算法？换 Trainer 类（Factory 重映射或自写 train script）

这套设计的好处是：自定义扩展通常只需走“换 scheduler + 换 reward + 子类化 Trainer”路径，没必要碰 Pipeline / Factory / CLI 这些上层装配代码。

### 一个 batch 的一生（数据形状追踪）

概念都认识了，再从数据的视角把管线走一遍。设 `per_device_train_batch_size=2`、`num_generations=4`，则一个 generation batch 共 B = 8 条轨迹，序列长度记作 L：

| 阶段 | 数据长什么样 | 谁产生的 |
| --- | --- | --- |
| dataset 的一行 | `{'messages': [{'role':'user','content': ...}], 其余自定义列}` | 数据集 |
| rollout 之后 | 每条轨迹一个 `RolloutOutput`：完整 `messages`、`response_token_ids`（每轮一个 list）、`response_loss_mask`、`rollout_infos` | scheduler + engine |
| reward 之后 | `rewards_per_func: [B, num_reward_funcs]`，加权合成 `rewards: [B]` | reward 函数 |
| tokenize 之后 | `input_ids` / `attention_mask` / `completion_mask`：`[B, L]` | `_prepare_batch_inputs` |
| advantage 之后 | `advantages: [B]`（组内归一化，同 prompt 的 4 条共享 baseline） | `_compute_advantages` |
| forward 之后 | `per_token_logps: [B, L]`（只有 `completion_mask=1` 的位置参与 loss） | 模型 |
| loss | 标量：PPO-clip(ratio, advantage) + β·KL，按 mask 做 token 平均再对 batch 取均值 | `_compute_loss_and_metrics` |

全程只有两条平行的主线：**文本世界**（`messages`，给人、给 reward 函数、给下一轮 prompt 用）和 **token 世界**（`token_ids` / `mask` / `logps`，给模型 forward 和 loss 用）。scheduler 在多轮里同时维护两边，tokenizer 是两边之间唯一的桥。凡是魔改出 bug，十有八九是这两条线在哪里对不齐了。

### 三个最容易卡住的问题

**问题一：`train()` 里只看到从 dataloader 采样，rollout 到底发生在哪一行？**

这是读 HF Trainer 系代码最大的一个陷阱。展开 `Trainer.train()` 的主循环，长这样：

```python
# transformers Trainer.train() 主循环（示意）
for epoch in range(num_epochs):
    for batch in dataloader:                    # ← 这里“采样”出的只是 dataset 的几行 prompt，不是经验数据
        loss = self.training_step(model, batch) # ← training_step 的第一件事就是 self._prepare_inputs(batch)
        ...
        optimizer.step()
```

关键在两点：

1. **dataloader 采出来的只是 prompt**（dataset 里的 `messages` 列）。此时既没有 completion，也没有 reward，没法算任何 loss。
2. HF `Trainer.training_step()` 开头会调 `inputs = self._prepare_inputs(inputs)`。原版 Trainer 里这个 hook 只做“把 tensor 搬上 GPU”之类的杂务，所以读代码的人默认它无关紧要；**GRPOTrainer 恰恰把整个 rollout 藏在了这个不起眼的 hook 里**：override 后的 `_prepare_inputs`（后文骨架 L187）会在时机到达时调 `_generate_and_score_completions`，把“生成 → 打分 → 编码 → 算 advantage”一整套跑完，返回的才是能算 loss 的张量。

所以答案是：**rollout 不在 `train()` 的字面里，而在 `training_step` 开头的 `_prepare_inputs` 钩子里**。“采样 prompt”（dataloader）和“采样轨迹”（rollout）是两次不同的采样，前者只是后者的输入。

**问题二：有没有 buffer？`_buffered_inputs` 算不算 replay buffer？**

不算。DQN 式的 replay buffer 是一个**长期蓄水池**：容量几十万条，新旧经验混存，训练时随机抽取，一条经验可能在几万步后还被用到。而 `_buffered_inputs` 只是一个**一次性的暂存架**，生命周期精确到步（对照后文骨架 L202-212）：

1. 第 `_step % generate_every == 0` 步：rollout 一整个 generation batch，切成 `steps_per_generation` 份 mini-batch 存进 `_buffered_inputs`；
2. 接下来的每个小步：按 `_step % num_rollout_samples` 取一份做 forward/backward（`num_rollout_samples` 即 `steps_per_generation`，开序列并行时再乘并行度）。同一批数据总共被用 `num_iterations × steps_per_generation` 个小步；
3. 窗口用完：整个 `_buffered_inputs` 被下一次 rollout **整体覆盖**，旧数据永不回头。

换句话说，这里没有“经验回放”，只有“**一批新鲜数据在被丢弃前多用几次**”。这是 PPO 家族的标准做法（PPO 论文里的 K 个 epoch），目的是摊薄昂贵的 rollout 成本；它与 off-policy 算法囤积并反复榨取历史数据是两回事。

**问题三：GRPO 是 on-policy 还是 off-policy？**

归类上是 **on-policy 算法（PPO 家族），带受控的短程数据复用**，即通常说的“近似 on-policy”。按数据的新鲜度可以分四档看：

1. **严格 on-policy 的部分**：每个 generation 窗口内的第一个梯度步。数据恰好由当前权重生成，`old_per_token_logps` 与当前 logps 相同，重要性比恒为 1，clip 不起作用。
2. **轻微 off-policy 的部分**：窗口内其余梯度步。权重已更新了几次，数据还是窗口开头那个策略生成的。这正是 loss 里 $\rho = \pi\_\theta / \pi\_{\text{old}}$ 和 PPO-clip 存在的理由：用比值校正这点漂移，用 clip 限制单步偏离。`_prepare_batch_inputs` 在 rollout 当场就把 `old_per_token_logps` 存下来（后文骨架 L878-891），就是为了记住“这批数据是哪个策略生成的”。
3. **主动引入的一步 off-policy**：`async_generate` 模式下，训练第 N 个窗口的同时后台在 rollout 第 N+1 个窗口，训练用的数据恒定落后一个窗口。这是拿一点 off-policy 换 rollout 与 training 的时间重叠。
4. **数值上的 off-policy**：rollout 概率由 vLLM 算，训练概率由 transformers 的 forward 算，同一份权重在两套实现下有微小数值差；`rollout_importance_sampling` 系列开关校正的就是这一层（后文骨架 L2424-2489）。

对照真正的 off-policy 算法（DQN、SAC 可以用任意旧策略的数据训练）：GRPO 不能，它只容忍“几步以内”的新鲜数据，靠 ratio 加 clip 撑住这几步；数据过期就整批扔掉、重新 rollout。

---

## 训练侧的 forward pass：一次前向到底算什么

概念说完了，进入正题的第一站：训练时的前向传播。GRPO 里其实有**两种前向**，很容易混：

| | rollout 生成前向 | 训练打分前向 |
| --- | --- | --- |
| 谁在跑 | vLLM（或退化的 TransformersEngine） | transformers 的 `model.forward` |
| 方式 | 自回归：一次一个 token，反复前向 | teacher forcing：整段序列一次前向 |
| 产出 | 新 token（completion） | 已有序列每个 token 的 log prob |
| 梯度 | 无（推理引擎根本没有 autograd 图） | 视用途而定，见下表 |
| 权重 | vLLM 自己那份副本（需同步） | 训练进程里的那份 |

本节只讲第二种。它被调用来算**三种 logps**，时机与梯度状态各不相同：

| logps | 什么时候算 | 用哪个模型 | 带梯度吗 | 代码 |
| --- | --- | --- | --- | --- |
| `old_per_token_logps` | rollout 刚结束、编码成张量时（一个窗口只算一次） | 当前策略（此刻＝生成时的策略） | 否（`torch.no_grad()`，且临时关掉 gradient checkpointing） | `_prepare_batch_inputs` L878-880 |
| `ref_per_token_logps` | 同上，紧随其后（`beta==0` 时跳过） | `ref_model`；LoRA 时用 `null_ref_context()` 禁用 adapter 冒充 ref | 否 | L881-891 |
| 当前策略 `per_token_logps` | **每个梯度小步都重算** | 当前策略（权重可能已更新过几次） | **是**，这是 backward 唯一的入口 | `_compute_loss_and_metrics` L1095-1096 |

三种 logps 走的是**同一个入口** `_get_per_token_logps_and_entropies`（L1721），内部按情况三选一：

```python
def _get_per_token_logps_and_entropies(self, model, inputs, compute_entropy=False):  # L1721
    # L1734：dynamic_num_samples（server 多轮返回样本数不定）时才走 _chunked 保证各 rank 块数一致；
    # 正常训练一律进 _single（L1740）
    ...

def _get_per_token_logps_and_entropies_single(self, model, inputs, compute_entropy=False):  # L1740
    # 三选一（L1762-1783）：
    # 路径①：can_use_super —— 纯文本模型 + 非 padding_free + 无序列并行 + 模型签名支持 logits_to_keep
    #        → 直接用 trl 基类的实现（内部同样是 forward → 截尾 → 除温度 → selective_log_softmax）
    # 路径②：sequence_parallel_size > 1 → _get_logps_via_sp（L1582，切序列多卡前向再 GatherLoss 拼回）
    # 路径③：其余（padding_free / 多模态 / 模型不支持 logits_to_keep）→ _get_logps_via_local_forward
    ...

def _get_logps_via_local_forward(self, model, inputs, logits_to_keep, input_ids, compute_entropy=False):  # L1676
    model_inputs = self._prepare_model_inputs(inputs)   # 剔除 GRPO 专属 key（advantages、completion_mask 等），
                                                        # 只留模型 forward 签名认识的参数（L2716-2726）
    logits = model(**model_inputs).logits               # L1688：真正的前向就这一行
    logits = logits[:, -(logits_to_keep + 1):-1, :]     # 截尾对齐：位置 i 的 logit 预测 token i+1，
                                                        # 且只保留 completion 段（prompt 段不算）
    logits = logits / self.temperature                  # L1691：除以采样温度，让训练概率和 rollout 同标度
    per_token_logps = selective_log_softmax(logits, completion_ids)  # 只取被采样 token 的 logp（trl 工具函数，
                                                        # 避免物化整个 [B, L, |V|] 的 log_softmax，省大量显存）
    return per_token_logps, entropies
```

四个值得记住的细节：

1. **`logits_to_keep` 是显存的救命稻草**：它在 `_prepare_batch_inputs`（L833-876）由 `labels != -100` 推出，等于 completion 的长度。前向时只保留这一段的 logits，prompt 段（往往比 completion 长）的 logits 根本不物化。`completion_mask` 也在同一处生成，scheduler 给的 `response_loss_mask` 就是在这里并进去的。
2. **温度不是超参装饰**：训练前向的 logits 除以与采样相同的 `temperature`，否则 $\rho = \pi\_\theta / \pi\_{\text{old}}$ 的分子分母不在同一标度上，ratio 系统性偏移。
3. **entropy 只在当前策略前向里算**（开关 `compute_entropy = log_entropy or top_entropy_quantile < 1.0`，L2224）；old/ref 两次前向不算 entropy。
4. **改前向要认清入口**：三种 logps 共用 `_get_per_token_logps_and_entropies`，你 override 它就同时改了 old / ref / current 三者；而 override `_get_logps_via_local_forward` 只影响路径③（纯文本小模型常走路径①的 trl 实现，根本不经过它）。后文“改 forward pass”一节详述。


---

## 采样出来的数据存在哪里

一句话总答案：**全程在内存里接力，训练管线不落盘**；唯一的样本级持久化是可选的 `completions.jsonl`。下面按数据形态的五次“变身”逐站看（行号均实测）。

**第一站：`RolloutOutput`（纯内存的 pydantic 对象）。** 由 engine 或 scheduler 创建（单轮：`GRPOVllmEngine.infer` 包装 vLLM 结果，grpo_vllm_engine.py L54；多轮：scheduler 的 `run()` 亲手构造，multi_turn.py L349）。server 模式下它的旅程是：worker 子进程（真正持有 vLLM）→ 进程间 Pipe（pickle）→ FastAPI 把它序列化成 **HTTP JSON 响应**→ 训练侧 `VLLMClient` 用 `RolloutOutput.model_validate` 还原（vllm_client.py L149-156）→ `broadcast_object_list` 广播给所有 rank、各自切片（rollout_mixin.py L1054-1061）。也因此 `rollout_infos` 必须可 JSON 序列化（图像等会被 `model_post_init` 自动转 base64，protocol.py L529-543）。

**第二站：样本 dict。** `_postprocess_rollout_outputs`（rollout_mixin.py L1128-1198）把 `RolloutOutput` 的字段逐个合并进（deepcopy 的）训练样本字典：`messages`、`response_token_ids`（有 loss_mask 才一并写入）、`rollout_infos`、`rollout_logprobs`、`finish_reason` / `is_truncated`。**`RolloutOutput` 的生命到此为止**：此后管线里流动的是纯 dict。

**第三站：张量字典。** `_prepare_batch_inputs`（grpo_trainer.py L795-995）把样本 dict 编码成训练张量，组织成 `[steps_per_generation][batch]` 的两层结构。每个 mini-batch dict 里有：`input_ids` / `attention_mask`（有 `response_token_ids` 时经 `replace_assistant_response_with_ids` 直接注入，跳过重新 tokenize，L813-821）、`completion_mask`、`logits_to_keep`、`truncated_mask`、`old_per_token_logps`、`ref_per_token_logps`、`rollout_per_token_logps`（默认 None）、`num_items_in_batch`，稍后再补 `advantages`（L263-266）。非张量的额外 key 只保留白名单（L166-168：`prompt_id` / `request_id` / `response_token_ids` / `finish_reason` / `is_truncated` / `add_eos`）。

**第四站：`_buffered_inputs`（唯一的复用缓冲，纯内存）。** 上一站的整个列表在窗口边界整体赋给 `self._buffered_inputs`（L206-208），窗口内每个小步按 `_step % num_rollout_samples` 取一份（L209），下个窗口整体覆盖、旧数据交给 GC。它不进 checkpoint：resume 之后第一步因 `_buffered_inputs is None` 必然重新 rollout，所以**断点续训不需要恢复任何采样数据**。

**第五站：`self._logs`（日志用环形缓冲）。** 一个 dict of `deque(maxlen=generation_batch_size)`（L2218-2226），存最近一个窗口的 prompt / completion / 各 reward 列 / advantages（数据集有 `solution` 列或多轮 `num_turns` 时自动加列，L289-301）。旧样本被自动挤出，从不显式清空。

**持久化盘点**（想找“训练数据存在哪个文件里”的人看这里）：

- **checkpoint 里没有任何 rollout 样本**（已核实：GRPO 不覆写 `_save_checkpoint`，走标准的权重 + 优化器 + `trainer_state.json`；后者只含 reward 均值等聚合标量）。
- `{output_dir}/logging.jsonl`：聚合标量（loss、reward 均值、KL 等），无样本文本。
- `{output_dir}/completions.jsonl`：**唯一的样本级落盘**，需 `--log_completions true`（默认关），主 rank 每个 logging step 追加一张表：step、prompt、completion、各 reward、advantage（grpo_trainer.py L2019、L2217）。开了 wandb / swanlab 的话同一张表也会上传成 completions 表格。
- `async_generate` 的 `DataCache`（rollout_mixin.py L53-57）：进程内 `Queue`，恰好领先一个 generation 窗口，消费即释放，同样不落盘。

**实用推论**：想拿训练期的采样数据做离线分析，正道是开 `--log_completions true`；要更多字段（比如逐轮的 rollout_infos），在 reward 函数或 scheduler 里自己写文件最干净：那两处能拿到整条 trajectory，且不碰训练管线。


---

## 训练的流程：一个 generation 窗口的完整时序

前向和数据都就位了，把一个完整训练窗口（从一次 rollout 到数据用尽）按时间顺序串起来。外层大循环是原封不动的 HF `Trainer.train()`（行号为 transformers v4.57.6，仓库推荐版本）：

1. **取 prompt batch**：`_inner_training_loop` 从 dataloader 取一个 batch（trainer.py L2673），内容只是 dataset 里的 prompt 行。
2. **进 training_step**：调用链 `GRPOTrainer.training_step`（grpo_trainer.py L1940，只加了 async_generate 的等待）→ trl `GRPOTrainer.training_step`（计时，并且 **`self._step += 1` 就发生在这里**）→ transformers `Trainer.training_step`。
3. **`_prepare_inputs` 钩子触发 rollout**（trainer.py L4014 → grpo_trainer.py L187）：若 `_step` 到达窗口边界（每 `generate_every = steps_per_generation × 序列并行度 × num_iterations` 个小步），执行 `_generate_and_score_completions`：
   - `_fast_infer`：必要时先 `_move_model_to_vllm()` 同步新权重，再让 vLLM/scheduler 生成整个 generation batch；
   - 打分 →（可选）动态重采样 → `_prepare_batch_inputs` 编码 + **no_grad 预计算 old/ref logps** → 组内优势；
   - 结果整体存进 `_buffered_inputs`。
   随后（无论是否刚 rollout）按 `_step % num_rollout_samples` 取出本小步的 mini-batch。
4. **算 loss**：`compute_loss`（trainer.py L4019 → grpo_trainer.py L1007 → `_compute_loss_and_metrics` L1090）：带梯度地重算当前策略 logps（前文“训练侧的 forward pass”一节讲的那次前向），组装 PPO-clip + KL。
5. **梯度累积缩放**：swift 把 `model_accepts_loss_kwargs` 置 False（grpo_trainer.py L139-142），强制走 HF 的 `loss = loss / gradient_accumulation_steps`（trainer.py L4059-4064）。
6. **backward**：`self.accelerator.backward(loss)`（trainer.py L4071）。注意这行在 transformers 里，swift 全仓库没有任何 backward 调用。
7. **累积与同步**：非同步小步在 `accelerator.no_sync` 下重复 2-6（梯度只累积在本 rank）；到同步小步，DDP all-reduce 随 backward 完成。
8. **裁剪与更新**：`clip_grad_norm_`（trainer.py L2714-2718）→ `optimizer.step()`（L2739）→ `lr_scheduler.step()` → `model.zero_grad()` → `global_step += 1`。
9. **窗口推进**：回到第 1 步。窗口内其余小步跳过 rollout 直接取 buffer；窗口用尽后，下一次 `_fast_infer` 先推权重再采样，进入新窗口。

把粒度对齐一下，三个“步”不要混：**`_step`**（trl 维护）数的是梯度小步（含累积），驱动 rollout 缓冲；**`global_step`**（HF 维护）数的是参数更新次数，驱动权重同步（`_last_loaded_step` 与它比较）与 logging/save；**generation 窗口**是 `_step` 的每 `generate_every` 个小步，驱动数据的生灭。


---

## 整体结构：调用栈逐层拆解（第 1-3 层：CLI → Pipeline → Factory）

前向、数据与流程都清楚了，这一部分把整体结构钉死：沿调用链自顶向下，把每一层的文件直接贴出来（薄的三层合在本节，重的第 4-6 层各占一节）。贴法统一：**类与函数签名、行号、注册表是真实源码**（v4.4.0-dev 实测），函数体替换成注释，注释写明这一块在干什么；与 GRPO 主线无关的部分一律折叠成一行（略）。每个文件在上面的目录地图里都能找到位置。

### 第 1 层：CLI 入口 `swift/cli/rlhf.py`（7 行，全文）

```python
# Copyright (c) ModelScope Contributors. All rights reserved.

if __name__ == '__main__':
    from swift.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    from swift.pipelines import rlhf_main
    rlhf_main()
```

`swift` 命令通过 `setup.py` 注册 entry point；`swift rlhf <args>` 等价于 `python -m swift.cli.rlhf <args>`。这一层没有逻辑，直接进 Pipeline。

### 第 2 层：Pipeline `swift/pipelines/train/rlhf.py`（246 行）

```python
# —— L1-20：真实 import 段——引入 RLHFArguments、safe_snapshot_download、prepare_adapter 等
# 模型加载工具，以及父类 SwiftSft（run() 与 TrainerFactory 的调用都在父类里，见本代码块末尾 rlhf_main 处的注释）


class SwiftRLHF(SwiftSft):  # L23
    args_class = RLHFArguments  # L24：--rlhf_type grpo 等 CLI 参数由它解析
    args: args_class

    @staticmethod  # L27
    def _get_model_task_type(model_dir):  # L28
        # 判断一个模型目录是 causal_lm 还是 seq_cls（reward model 通常是 seq_cls 且 num_labels=1）；
        # 先读 args.json，没有则从 HF config 的 architectures/num_labels 推断，
        # GRPO 加载 reward model 时靠它决定挂分类头还是语言模型头
        ...

    def _prepare_single_model(self, key, origin_key, model_type, model_revision):  # L55
        # 按 key（ref/reward/value/teacher）加载一个「配角」模型，GRPO 只用到 ref 和 reward：
        # 1. 从 args.{key}_model 取 model id，为 None 直接返回（GRPO 不配 reward_model 时即如此）
        # 2. safe_snapshot_download 拉取权重，_get_model_task_type 判定加载方式
        # 3. args.get_model_processor 实例化 model + processor（tokenizer）
        # 4. ref/reward/teacher 一律 requires_grad_(False).eval()：只做前向打分，永不参与训练；
        #    value 分支仅 PPO 用（会走 prepare_model 变成可训练），GRPO 不涉及
        # 5. 关掉 use_cache，返回 (model, processor)
        # （PPO 多 reward 断言、teacher_deepspeed、sequence_parallel 等与 GRPO 主线无关的分支，略）
        ...

    def _prepare_model_tokenizer(self):  # L111
        # GRPO 的全部「配角」模型在这里就位（被训练的 actor 由末尾的 super() 调用加载）：
        # 1. 遍历 ['ref', 'value', 'teacher']：GRPO 只加载 ref_model（算 KL 的参考模型，冻结）；
        #    value 仅 ppo、teacher 仅 gkd，GRPO 下直接 continue 跳过
        # 2. reward model：GRPO 支持传入多个 reward_model（列表），逐个调 _prepare_single_model
        #    加载进 self.reward_model 列表；且仅 GRPO 会为每个 reward model 额外构建
        #    self.reward_template——reward model 可能与 actor 不同源，打分时需用它自己的
        #    chat template 重新编码 rollout 文本
        # 3. 非 GRPO 把 reward_model 塌缩回单个；GRPO 保持列表（多奖励加权）
        # 4. super()._prepare_model_tokenizer() 加载 actor 模型与 tokenizer
        ...

    @classmethod  # L169
    def prepare_model(cls, args, model, *, template=None, train_dataset=None, task_type=None):  # L170
        # 先走父类的 tuner/全参训练准备；（ref_adapters 的 LoRA plumbing 与 GRPO 主线无关，略）
        ...

    def _prepare_template(self) -> None:  # L188
        # 按 rlhf_type 设定 template 模式：GRPO 映射为 'train'，即 rollout 样本按普通训练格式
        # 编码；dpo 等则走 'rlhf' 的成对编码。（ppo 的 stop_token_id 设置，略）
        ...

    def _get_dataset(self):  # L197
        # 复用父类取数据集；GRPO 的数据只需 prompt 列，无需改造（KTO 的特殊数据准备分支，略）
        ...

    def _prepare_chord_sft_dataset(self):  # L204
        # （CHORD 混合 SFT 数据的准备，与 GRPO 纯 RL 主线无关，略）
        ...

    def _get_trainer_kwargs(self):  # L220
        # 本文件与 GRPOTrainer 的交接点：汇总所有要塞进 trainer 构造函数的额外 kwargs：
        # 1. ref_model / reward_model（GRPO 下为列表）按是否存在放入
        # 2. reward_template：与 reward_model 一一对应，GRPO 专属
        # 3. GRPO/GKD 共用：vllm_client（server 模式下 rollout 引擎的客户端句柄）
        # 4. GRPO 专属：reward_funcs（accuracy/format 等自定义奖励函数，可与 reward_model 并存）
        # （chord_sft_dataset、GKD teacher 相关 kwargs 与 GRPO 主线无关，略）
        ...


def rlhf_main(args: Optional[Union[List[str], RLHFArguments]] = None):  # L245
    # CLI 入口：swift rlhf --rlhf_type grpo 最终落到这一行
    # 调用链：SwiftRLHF(args).main()（base.py L49）→ 继承自 SwiftSft 的 run()（sft.py L159）：
    #   ① _prepare_dataset 取数据 → ② prepare_model 给 actor 套 tuner
    #   → ③ TrainerFactory.get_trainer_cls(args)（sft.py L175）按 rlhf_type 选出 GRPOTrainer
    #   → ④ trainer_cls(model=actor, args=training_args, template=..., train_dataset=...,
    #        eval_dataset=..., **self._get_trainer_kwargs())
    #   上面 L220 收集的 ref_model / reward_model / reward_template / vllm_client / reward_funcs
    #   就在第 ④ 步注入 GRPOTrainer，随后 trainer.train() 开始 rollout-打分-更新循环
    return SwiftRLHF(args).main()
```

### 第 3 层：Factory `swift/trainers/trainer_factory.py`（73 行）

```python
# —— swift/trainers/trainer_factory.py（全文 73 行）——
# —— L1-9：importlib/inspect/asdict 等 import 与 logger 初始化（略）——

class TrainerFactory:
    # —— L13-28：训练方法名 → Trainer 类路径的注册表。GRPO 调用链的第一跳：
    #    rlhf_type='grpo' 时由此查到 swift.rlhf_trainers.GRPOTrainer ——
    TRAINER_MAPPING = {
        'causal_lm': 'swift.trainers.Seq2SeqTrainer',
        'seq_cls': 'swift.trainers.Trainer',
        'embedding': 'swift.trainers.EmbeddingTrainer',
        'reranker': 'swift.trainers.RerankerTrainer',
        'generative_reranker': 'swift.trainers.RerankerTrainer',
        # rlhf
        'dpo': 'swift.rlhf_trainers.DPOTrainer',
        'orpo': 'swift.rlhf_trainers.ORPOTrainer',
        'kto': 'swift.rlhf_trainers.KTOTrainer',
        'cpo': 'swift.rlhf_trainers.CPOTrainer',
        'rm': 'swift.rlhf_trainers.RewardTrainer',
        'ppo': 'swift.rlhf_trainers.PPOTrainer',
        'grpo': 'swift.rlhf_trainers.GRPOTrainer',  # ← GRPO 主线
        'gkd': 'swift.rlhf_trainers.GKDTrainer',
    }

    # —— L30-45：训练方法名 → TrainingArguments 类路径，条目与上表一一对应；
    #    'grpo' 对应 'swift.rlhf_trainers.GRPOConfig'（其余条目略）——
    TRAINING_ARGS_MAPPING = {...}

    @staticmethod
    def get_cls(args, mapping: Dict[str, str]):
        # 若 args 带 rlhf_type 则按 rlhf_type 查表（GRPO 走这一支），否则退回 task_type；
        # 把 'swift.rlhf_trainers.GRPOTrainer' 这样的字符串拆成模块路径与类名，
        # importlib 动态 import 后返回类对象。惰性导入使得未用到的 trainer 不会被加载。
        ...

    @classmethod
    def get_trainer_cls(cls, args):
        # 查 TRAINER_MAPPING：GRPO 场景下返回 GRPOTrainer 类（尚未实例化）。
        ...

    @classmethod
    def get_training_args(cls, args):
        # 查 TRAINING_ARGS_MAPPING 得到 GRPOConfig 类；把 swift 侧的 args dataclass 用 asdict
        # 展平成 dict，再按 GRPOConfig.__init__ 的签名过滤掉不认识的键（swift 参数集是
        # TRL 参数集的超集），经 args._prepare_training_args 钩子微调后实例化并返回，
        # 该 training_args 随后被传入 GRPOTrainer 构造函数。
        ...
```

`--rlhf_type grpo` 在 `TRAINER_MAPPING` 查到 `GRPOTrainer`，实例化后 `trainer.train()` 进入训练循环。下一站就是全文的主角。

---

## 第 4 层：GRPOTrainer 主线骨架（`grpo_trainer.py`，2732 行）

这个 2732 行的文件是整条管线的心脏。下面是它的主线骨架。`__init__` 里那一串 `_prepare_*`（CHORD、Liger、metrics 之类）对理解 GRPO 无关紧要，各自折叠成一行；值得记住的只有两个成员变量（`_step` 与 `_buffered_inputs`）和训练路径上的那几个方法。

```python
# swift/rlhf_trainers/grpo_trainer.py（共 2732 行）主线骨架，行号以 # Lnnn 标注，均已对照源码核实
# —— L1-82：imports、vLLM/trl 兼容 patch、logger 初始化（与 GRPO 主线无关，略）——

class GRPOTrainer(RolloutTrainerMixin, SwiftMixin, HFGRPOTrainer):  # L84

    def __init__(self, model=None, ref_model=None, reward_model=None, reward_funcs=None, *_args, **kwargs):  # L87
        self._prepare_algorithm_params()  # L107：把 GRPOConfig 里的算法开关读成属性，定义在本文件 L2240
        super().__init__(model, ref_model, *_args, **kwargs)  # L108
        # L109：_prepare_chord_dataset()（CHORD 混合 SFT，与 GRPO 主线无关，略）
        self.prepare_rollout()  # L110：搭建 vLLM/rollout 基础设施，定义在 rollout_mixin.py 的 RolloutTrainerMixin（L108）
        self._prepare_rewards(reward_funcs, reward_model, reward_templates)  # L111：注册 reward 函数与 reward model，本文件 L2297
        # L124-125：_prepare_liger_loss()/_prepare_metrics()（Liger kernel 与 metrics 容器，略）
        # L130-158：随机种子、非 vLLM 时退化到 TransformersEngine、SyncRefModelCallback、序列并行等杂项（略）
        self._step = 0  # L161：前向＋反向的迭代计数（含梯度累积内的小步），驱动 _prepare_inputs 的缓冲逻辑；自增在 trl 的 training_step 里
        self._buffered_inputs = None  # L164：缓存一整个 generation batch 的 rollout 结果，供多个梯度步复用

    # L170-184：_get_data_collator/_get_train_sampler（数据管道杂项，略）

    @profiling_decorator  # L186
    def _prepare_inputs(self, generation_batch):  # L187
        # 训练主入口：Trainer.training_step 每个梯度小步都会调到这里；核心是“生成一次、复用多步”
        mode = 'train' if self.model.training else 'eval'  # L202
        if mode == 'train':
            num_rollout_samples = self.args.steps_per_generation * self.template.sequence_parallel_size  # L204
            generate_every = num_rollout_samples * self.num_iterations  # L205
            if self._step % generate_every == 0 or self._buffered_inputs is None:  # L206：只在每 generate_every 步真正 rollout 一次
                self._buffered_inputs = self._generate_and_score_completions(generation_batch)  # L207-208
            inputs = self._buffered_inputs[self._step % num_rollout_samples]  # L209：按当前小步取出对应 mini-batch 切片
        else:
            inputs = self._generate_and_score_completions(generation_batch)  # L211：eval 不缓冲、逐 batch 生成
        return inputs  # L212

    def _generate_completions(self, inputs: DataType) -> DataType:  # L214
        # 生成 completion：use_fast_infer 时走 self._fast_infer（vLLM，rollout_mixin.py L934），
        # 否则 unwrap 模型后经 _infer_single_or_multi_turn 用 TransformersEngine 推理（L216-231）

    @profiling_decorator  # L233
    def _generate_and_score_completions(self, inputs: DataType) -> DataType:  # L234
        # 一次 rollout 的完整流水线，五个子步骤：
        inputs = self._generate_completions(inputs)  # L239：① 生成 completion
        total_rewards_per_func = self._score_completions(inputs)  # L240：② 打分，得到全局 gather 后的 per-func reward
        if self.dynamic_sample and mode == 'train':  # L243
            inputs, total_rewards_per_func = self._dynamic_sampling(inputs, total_rewards_per_func)  # L245：③ DAPO 动态重采样
        batch_encoded_inputs = self._prepare_batch_inputs(inputs)  # L247：④ 编码成训练张量并预计算 old/ref logps
        total_advantages = self._compute_advantages(inputs, total_rewards_per_func, batch_encoded_inputs)  # L249：⑤ 组内相对优势
        # L251-266：把全局 advantages 按进程切回本地，并写进每个 mini-batch 的 'advantages' 字段
        # L268-301：gather prompt/completion/solution 等进 self._logs（logging，略）
        return batch_encoded_inputs  # L303

    @profiling_decorator  # L305
    def _score_completions(self, inputs: DataType) -> torch.Tensor:  # L306
        # 本地调 _compute_rewards_per_func 后跨进程 gather 成 (N, num_reward_funcs)；
        # gym 环境的 total_reward 作为额外一列拼入，与 reward_funcs 共享 reward_weights（L318-341）

    def _compute_rewards_per_func(self, inputs: DataType) -> torch.Tensor:  # L343
        # 逐个执行 reward：reward model（nn.Module）走 plugin，同步函数直接调，
        # 异步函数收集后用 asyncio.gather 并行跑（L360-397）；某行全为 NaN 时告警（L399-408）

    def _compute_advantages(self, inputs, rewards_per_func, batch_encoded_inputs):  # L412
        rewards = (rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)  # L481：多个 reward 加权合成标量
        # L483-497：kl_in_reward=True 时把 β·KL 直接从 reward 里扣掉（KL 进 reward 而非 loss）
        # —— L504-583：Case 1 默认分组模式，每个 prompt 恰好 num_generations 条 ——
        grouped_rewards = rewards.view(-1, num_generations)  # L505：reshape 成 [组数, K]
        advantages = rewards - group_rewards_mean  # L526：GRPO 核心：优势 = 个体 reward − 组均值（组内基线）
        # L515-523：advantage_estimator='rloo' 时改用留一基线：A_i = (r_i − mean)·K/(K−1)
        # L528-575：按 scale_rewards（group/batch/gdpo/none）除以对应 std 归一化（+1e-4 防除零）；reinforce_plus_plus 用优势的 std
        # —— L588-689：Case 2 动态样本数模式（多轮对话）：按 request_id 去重、按 prompt_id 分组后做同样的均值/std 计算 ——
        return advantages  # L583/L689

    @profiling_decorator  # L691
    def _dynamic_sampling(self, inputs, rewards_per_func):  # L692
        # DAPO：组内 reward std=0 的组没有梯度信号，循环重采样替换这些组，直到凑够 generation_batch_size（至多 max_resample_times 次）

    # L743-781：compute_std（为 _dynamic_sampling 计算组内 reward std，略）
    # L783-792：null_ref_context（PEFT 下禁用 adapter 充当 ref model，LoRA plumbing，略）

    @profiling_decorator  # L794
    def _prepare_batch_inputs(self, inputs: DataType) -> List[DataType]:  # L795
        # 把 rollout 文本变成训练张量，按 split_by_mini_batches 切成 [steps_per_generation][bs] 结构：
        # L806-830：template.encode ＋ data_collator，得到 input_ids/attention_mask 等
        # L833-876：由 labels≠-100 推出 logits_to_keep 与 completion_mask（只有 completion token 参与 loss）
        # L878-891：no_grad 下先算 old_per_token_logps（生成时策略的快照）；beta≠0 时再算 ref_per_token_logps（ref model 或 null_ref_context）
        # L893-949：若开 rollout IS 校正，把 vLLM 返回的 rollout_logprobs 对齐到 completion 位置，存为 rollout_per_token_logps
        # L953-995：num_items_in_batch（全局 completion token 总数，DAPO 类归一化用）与长度/截断指标（logging，略）
        # 每个 mini-batch 产出的关键张量：input_ids、completion_mask、logits_to_keep、truncated_mask、
        # old_per_token_logps、ref_per_token_logps、rollout_per_token_logps、num_items_in_batch（advantages 稍后在 L263-266 补上）
        return ga_batch_encoded_inputs  # L995

    # L997-1004：_apply_chat_template_to_messages_list（logging 辅助，略）

    @profiling_decorator  # L1006
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # L1007
        # Trainer 的 loss 入口：use_liger_loss 时走 compute_liger_loss（略），否则进 _compute_loss（L1019）

    def _compute_loss(self, model, inputs):  # L1021
        # 正常走 _compute_loss_single（L1030）；动态多轮样本超出 per_device batch 时走 _compute_loss_chunked 分块（L1033）

    def _compute_loss_single(self, model, inputs):  # L1035
        loss, metrics_data = self._compute_loss_and_metrics(model, inputs)  # L1037

    # L1041-1088：_compute_fipo_influence（FIPO 算法专属，略）

    def _compute_loss_and_metrics(self, model, inputs):  # L1090
        # GRPO 的全部 loss 数学都在这个函数里，逐行讲解见下文“GRPO loss 公式”一节
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(model, inputs, compute_entropy=self.compute_entropy)  # L1095-1096
        # L1101-1126：entropy mask（top_entropy_quantile）与 overlong_filter 截断过滤（可选项，略）
        if self.beta != 0.0 and not self.kl_in_reward:  # L1130：KL 惩罚，k3 估计量，双重 clamp 防数值爆炸
            safe_ratio = torch.clamp(ref_per_token_logps - per_token_logps, min=-20, max=20)  # L1132
            per_token_kl = torch.clamp(torch.exp(safe_ratio) - safe_ratio - 1, min=-10, max=10)  # L1133
        # 兼容分支：若 old_per_token_logps 缺省则 detach 当前 logps 复用；本版本 _prepare_batch_inputs
        # （L878-880）总是预先算好 old_per_token_logps，该 None 分支实际不会触发：
        old_per_token_logps = (per_token_logps.detach() if inputs['old_per_token_logps'] is None else inputs['old_per_token_logps'])  # L1141-1142
        # L1144-1170：rollout（vLLM）与训练策略间的 off-policy 诊断指标＋IS 校正权重（可选，略）

        # 重要性采样比的三个粒度：token 即 GRPO，sequence 即 GSPO，sequence_token 即 GSPO-token：
        log_ratio = per_token_logps - old_per_token_logps  # L1172
        if self.importance_sampling_level == 'token':  # L1173
            log_importance_weights = log_ratio  # L1174
        elif self.importance_sampling_level in ['sequence', 'sequence_token']:  # L1175
            seq_level_log_weights = ((log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).unsqueeze(-1)  # L1176-1177
            if self.importance_sampling_level == 'sequence':  # L1178
                log_importance_weights = seq_level_log_weights  # L1179
            else:  # L1181：GSPO-token：sg[si(θ)] * πθ(yi,t)/sg[πθ(yi,t)]
                log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weights.detach()  # L1182-1183
        coef_1 = torch.exp(log_importance_weights)  # L1189

        # L1192-1204：cispo/sapo/real 等 loss 变体的 per_token_loss（非 GRPO 主线，略）
        # loss_type 属于 GRPO 一族时的 PPO-clip：
        elif self.loss_type in ['grpo', 'bnpo', 'dr_grpo', 'dapo', 'fipo']:  # L1205
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)  # L1209：PPO 式非对称裁剪（DAPO 的 clip-higher）
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)  # L1213
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)  # L1214
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)  # L1215：悲观取 min，即 PPO-clip 目标
        if per_token_kl is not None:  # L1220
            per_token_loss = per_token_loss + self.beta * per_token_kl  # L1221：KL 惩罚加进 loss
        # L1224-1238：rollout IS 权重相乘、off-policy sequence mask 并入 completion_mask（可选项，略）

        # 归一化：loss_type 决定分母（序列内 mask-mean 再对样本平均，或除以全局 token 数）：
        if self.loss_type in ['grpo', 'sapo']:  # L1240
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()  # L1242
        elif self.loss_type == 'bnpo':  # L1243
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)  # L1244
        elif self.loss_type == 'dr_grpo':  # L1245
            loss = (per_token_loss * completion_mask).sum() / (batch_size * self.max_completion_length)  # L1247
        # L1248-1280：loss_type='real' 分支（略）
        elif self.loss_type in ['cispo', 'dapo', 'fipo']:  # L1281
            normalizer = inputs['num_items_in_batch'] / self.accelerator.num_processes  # L1283：全局 completion token 数
            loss = (per_token_loss * completion_mask).sum() / normalizer  # L1284
        # L1288-1351：KL/clip 比例等 metrics_data 组装（logging，略）；L1352-1353：CHORD loss 混合（略）
        return loss, metrics_data  # L1355

    # L1357-1549：_update_metrics/_compute_loss_chunked/_aggregate_and_update_metrics（metrics 与分块杂项，略）
    # L1551-1674：_unpad_logps_and_entropies/_get_logps_via_sp（padding_free 还原与序列并行前向，略）
    # L1676-1718：_get_logps_via_local_forward：真正的前向：logits 截尾 → 除以 temperature → selective_log_softmax

    @profiling_decorator  # L1720
    def _get_per_token_logps_and_entropies(self, model, inputs, compute_entropy=False):  # L1721
        # 统一入口：按 super()/SP/本地前向三条路径算 per-token logps，必要时分块省显存；
        # 被 _prepare_batch_inputs（old/ref logps）与 _compute_loss_and_metrics（当前策略 logps）共用

    # L1740-1945：logps 的 single/chunked 实现、Liger、evaluation_loop/training_step 等（略）

    def old_policy(self):  # L1947
        # 判断是否存在真 off-policy：num_iterations>1，或梯度累积步数不能整除 steps_per_generation 时，
        # 同一批 rollout 会被多次更新，策略已漂移
        # （注：本版本中该方法定义而未被任何地方调用，old_per_token_logps 一律在 _prepare_batch_inputs 预先算好）
        return (self.num_iterations > 1 or self.args.gradient_accumulation_steps % self.args.steps_per_generation != 0)  # L1949-1950

    # L1955-2190：显存 offload、wandb/swanlab completion 表格、分布式辅助（略）
    # L2192-2238：_prepare_liger_loss/_prepare_metrics/_collect_config_info（初始化杂项，略）

    def _prepare_algorithm_params(self):  # L2240
        # 把 GRPOConfig 展开成算法属性：num_iterations（GRPO 论文的 μ）、epsilon_low/high（DAPO 非对称裁剪）、
        # importance_sampling_level（GSPO）、advantage_estimator（grpo/rloo/reinforce_plus_plus）、
        # kl_in_reward、rollout_importance_sampling_mode、off_policy_sequence_mask_delta 等

    def _prepare_rewards(self, reward_funcs, reward_model=None, reward_templates=None):  # L2297
        # 把字符串名解析成 swift.rewards.orms 里的 reward 类、给 reward model 挂 plugin/template、
        # 组装 reward_weights，并为异步 reward 函数启动独立事件循环（L2301-2381）

    # L2383-2531：动态采样数据迭代器、rollout IS 校正四种模式、off-policy 序列屏蔽（可选机制，略）
    # L2533-2732：off-policy 诊断指标、_prepare_model_inputs、_get_eval_sampler（略）
```

三点导读：

- **生成一次、复用多步**（`_prepare_inputs`，L187）：GRPO 不是每个梯度小步都 rollout，而是每 `steps_per_generation × num_iterations` 个小步 rollout 一次，结果缓存在 `_buffered_inputs` 里逐步取用。训练时看到“loss 在动而 completions 没变”，就是这个机制在工作。
- **一次 rollout 的五个子步骤**（`_generate_and_score_completions`，L234）：生成 → 打分 → 动态重采样（可选）→ 编码成张量 → 组内优势。reward 函数在第 ② 步被调用，能看到 completion 全文、数据集自定义列与整条 trajectory；第 ⑤ 步的 `advantage = reward − 组均值`（再按配置除以 std）就是 GRPO 的“G”：同 prompt 的 `num_generations` 条 rollout 互为 baseline，省掉 PPO 的 critic。
- **loss 全在一个函数里**（`_compute_loss_and_metrics`，L1090）：骨架里已能看到 KL 的 k3 估计、token/sequence 两级重要性采样比与 PPO-clip；公式层面的逐段讲解见下文“GRPO loss 公式”一节。

自定义组件的注入点骨架里也已标出：reward 走 `_prepare_rewards`（L2297，查 `orms` 注册表，命令行 `--reward_funcs 名字`）；多轮调度走 `prepare_rollout` 里的 `_prepare_scheduler`（定义在 rollout_mixin.py，查 `multi_turns` 注册表，命令行 `--multi_turn_scheduler 名字 --max_turns N`）。改前向、改 loss、改梯度的入口，见下文“改 forward pass”与“改学习算法与手动改梯度”两节。

---

## 第 5 层：RolloutTrainerMixin 基建骨架（`rollout_mixin.py`，1654 行）

GRPOTrainer 的第一个基类，训练循环之外的一切脏活都在这里：vLLM 的装配、colocate/server 两种模式的分发、训练权重到 vLLM 的同步、多轮推理的驱动。外层的训练大循环本身继承自 HF `Trainer.train()`：每个梯度小步进 `training_step`，它开头的 `_prepare_inputs`（见上一节骨架）触发 rollout，随后 `compute_loss` → `loss.backward()` → `optimizer.step()`，没有任何 GRPO 特殊逻辑。

```python
# swift/rlhf_trainers/rollout_mixin.py（共 1654 行）：GRPO rollout 基建骨架
# —— L53-81：DataCache／AsyncGenerateCallback／SyncRefModelCallback，异步生成的结果容器与两个 TrainerCallback，细节略 ——

class RolloutTrainerMixin(RLHFTrainerMixin):  # L84
    # rollout 类 RLHF trainer（GRPO／GKD）的公共基建：vLLM 采样（colocate／server 两模式）、多轮对话、异步生成。

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)  # L95：async_generate 的后台单线程池

    def __init__(self, model=None, ref_model=None, *_args, **kwargs):  # L97
        # 只做 FSDP 版本检查（仅支持 FSDP2）；vLLM 相关初始化全部推迟到 prepare_rollout()。

    def prepare_rollout(self):  # L108：GRPOTrainer 初始化末尾调用的总装配入口
        self._prepare_rollout_params()  # L163：把采样超参打包成 self.request_config
        self._prepare_scheduler()       # L1315：装配多轮调度器 self.multi_turn_scheduler
        self._prepare_vllm()            # L195：装配 vLLM 引擎（colocate）或与 rollout server 握手
        self._prepare_async_generate()  # L1466：建异步生成队列并注册回调
        self.parameter_groups, self.parameter_groups_no_lora = self.split_batches()  # L341：权重同步分桶

    # —— L115-161：_split_data_by_steps／split_by_mini_batches，按 steps_per_generation 均分 mini-batch（SP 分支略）——

    def _prepare_rollout_params(self):  # L163
        # 读取 num_generations／temperature／top_p／top_k／max_completion_length 等，构造
        # self.request_config = RequestConfig(n=1, ..., logprobs=args.use_vllm)：rollout logprobs 供重要性采样用。

    def _prepare_vllm(self):  # L195
        # server 模式：主进程用 self.vllm_client（VLLMClient，由 GRPOTrainer 注入的训练侧 HTTP 客户端）调
        #   get_engine_type() 握手，读回 use_async_engine／enable_multi_turn／enable_lora／use_gym_env，
        #   broadcast 给所有 rank；是否多轮由 rollout server 的启动参数决定，trainer 只读标志。
        # colocate 模式：按 vllm_tensor_parallel_size 建 tp_group，self.engine = self._prepare_vllm_engine()；
        #   若 sleep_level>0，先让引擎 sleep 把显存还给训练。
        # 末尾初始化 base_sync_done／_last_loaded_step 等权重同步状态位。

    def _prepare_vllm_engine(self):  # L266：仅 colocate
        # 与训练进程同卡构造 GRPOVllmEngine（L313）：distributed_executor_backend='external_launcher'、
        # enable_sleep_mode=args.sleep_level>0、logprobs_mode='processed_logprobs'；（LoRA／量化／MoE 旁路，略）。

    def split_batches(self):  # L341
        # 把参数名按 move_model_batches 分桶（LLM 层按层号、embedding 与多模态组件各自成桶），
        # 返回 (parameter_groups, parameter_groups_no_lora)：之后权重分批同步到 vLLM，压低同步时的峰值显存。

    # ============ 权重同步：训练出的新策略 → vLLM ============

    @profiling_decorator
    def _move_model_to_vllm(self, skip_async_check=False):  # L449-450
        # 同步入口，由 _fast_infer 在 global_step 变化时触发（L459 分支）：full 微调、base 未同步、
        # sleep_level==2 或未开 vllm_enable_lora → _move_full_model_to_vllm()；仅当 LoRA＋base 已同步＋
        # vllm_enable_lora=True 且 sleep_level≠2 时才走 _move_adapter_to_vllm()；
        # 最后 _reset_vllm_cache()（L466）清 prefix cache，防止旧权重的缓存污染新采样。

    def _move_adapter_to_vllm(self):  # L478
        # 只传 LoRA 增量：逐 parameter_group gather（ZeRO-3）并抽出 lora 权重，
        # server → vllm_client.update_adapter_flattened_param(...)；colocate → engine.engine.add_lora(TensorLoRARequest)。

    def _load_state_dict_to_vllm(self, state_dict):  # L558
        # 真正写入权重：server 模式把参数摊平分 bucket 传给 rollout server；
        # colocate 直接 self.engine.inner_model.load_weights(state_dict.items())。

    # —— L581-771：参数名清洗、FSDP2 下 LoRA 张量级合并、按组收集 state_dict（DTensor→Tensor），细节略 ——

    def _move_full_model_to_vllm(self):  # L773
        # 逐 parameter_group：gather_if_zero3 → merge_adapter → 收集 state_dict → _load_state_dict_to_vllm → unmerge；
        # 全部组加载完后统一 process_weights_after_loading；PEFT 时置 base_sync_done=True。

    @torch.no_grad()
    def _sync_ref_model_weights(self, alpha: float) -> None:  # L827-828
        # ref ← (1-α)·ref + α·policy 的 EMA 混合，由 SyncRefModelCallback 每 ref_model_sync_steps 步触发，略。

    # ============ 采样调用链：_fast_infer → _infer_single_or_multi_turn → _rollout ============

    def _rollout(self, inputs, request_config, is_global_inputs=False) -> List[RolloutOutput]:  # L867
        # 单轮采样分发：vllm_mode 为 server → _server_rollout，否则 → _colocate_rollout。

    # L879 _get_request_config：colocate TP>1 时按 TP 组设种子；L894 _set_inputs_system：补默认 system，略。

    def _infer_single_or_multi_turn(self, inputs, request_config, is_global_inputs=False) -> List[DataType]:  # L906
        # 先 _rollout 拿首轮输出；若无 scheduler、或多轮已由 server 端负责（enable_server_multi_turn）→ 直接后处理返回；
        # 否则进入 _colocate_multi_turn_infer，在 trainer 侧继续多轮。

    def _colocate_multi_turn_infer(self, inputs, first_turn_rollout_outputs, request_config):  # L919
        # trainer 侧多轮驱动：调 swift/rollout/agent_loop.py 的 run_multi_turn（L923），
        # 它逐轮调 scheduler.step() 决定是否续写，以 self._rollout 为 rollout_fn 生成下一轮，直到所有轨迹结束；
        # 异步的 MultiTurnScheduler.run 不在此路径，只在 rollout server 的 AsyncEngine 内部被驱动。

    def _fast_infer(self, inputs: DataType) -> DataType:  # L934：GRPO 采样主入口
        # ① colocate 且 sleep_level>0 且引擎在睡：wake_up(tags=['weights'])，先只唤醒权重部分；
        # ② global_step != _last_loaded_step 或 sleep_level==2 → _move_model_to_vllm()：训练一步后同步新权重；
        # ③ 再 wake_up(tags=['kv_cache'])；async_generate 时 gather 全局输入提交后台线程、从 _queue 取上一批结果
        #    （一步 off-policy），否则在 multi_turn_completion_length_context() 内同步走 _infer_single_or_multi_turn；
        # ④ colocate 收尾：reset_prefix_cache + engine.sleep(level=args.sleep_level)，显存归还训练。

    # L984-1011：_preprocess_inputs／_add_prompt_id_to_inputs，清旧 response、发全局 prompt_id／request_id，略。

    def _server_rollout(self, inputs, request_config, is_global_inputs) -> List[RolloutOutput]:  # L1013
        # inputs2requests → gather_object 汇总到主进程 → 仅主进程 _engine_infer（vllm_client.infer 走 HTTP，
        # 只传请求体与 RequestConfig，不传 scheduler 名：多轮由 swift rollout 启动时的 --multi_turn_scheduler
        # 在 server 侧完成）→ broadcast 回所有 rank，各自按请求归属切片取回输出；（动态样本数兼容分支，略）。

    def _colocate_rollout(self, inputs, request_config) -> List[RolloutOutput]:  # L1069
        # TP>1 时先在 tp_group 内 all_gather 输入，各 rank 用本地 engine 跑 _engine_infer，再切回自己那段；
        # 本函数只做单轮生成，多轮由上层 _colocate_multi_turn_infer 反复调用它来推进。

    def _engine_infer(self, infer_requests, request_config=None, *, use_tqdm=False):  # L1091
        # 统一推理出口：server → self.vllm_client.infer(...)；colocate → self.engine.infer(...)；结果统一包成 RolloutOutput。

    def _postprocess_rollout_outputs(self, inputs, outputs) -> DataType:  # L1128
        # 把 RolloutOutput 合并回训练样本：messages／response_token_ids／response_loss_mask／
        # rollout_logprobs／finish_reason，GRPO 后续的 reward 计算与 loss 编码都消费这个结构。

    # —— L1200-1313：offload_model／load_optimizer 等，colocate 下训练态与 vLLM 态之间的显存腾挪，工程细节，略 ——

    def _prepare_scheduler(self):  # L1315
        # 多轮调度器装配（目前仅 GRPO 支持该参数）：args.multi_turn_scheduler 为字符串时在 multi_turns 注册表查类，
        # 以 max_turns=args.max_turns、tokenizer=self.processing_class（有 gym_env 时再加 gym_env）为 kwargs 实例化；
        # 已是 MultiTurnScheduler 实例则直接采用；未配置则 self.multi_turn_scheduler = None。

    @contextmanager
    def multi_turn_completion_length_context(self):  # L1339-1340
        # colocate 多轮时临时 patch engine.set_default_max_tokens：让 max_completion_length 限制整条轨迹总长而非每轮，略。

    def inputs2requests(self, inputs) -> List[RolloutInferRequest]:  # L1375
        # dict 样本 → RolloutInferRequest；配了 scheduler 或 vllm_server_pass_dataset 时，
        # 把数据集额外列打包进 data_dict，随请求传给调度器（reward 函数也靠它拿 ground truth）。

    # ============ async generate（一步 off-policy 的异步采样机制）============
    def async_generate_rollout(self, all_inputs):  # L1443：把 _infer_single_or_multi_turn 提交到 executor，完成后结果入 _queue
    def _prepare_async_generate(self):  # L1466：建 train_queue／eval_queue；async_generate 时注册 AsyncGenerateCallback
    # L1475-1503：队列选择、等待、按 request_id 排序、训练开始时的首批预取，略。

    # —— L1508-1654：sequence-parallel 开关、模板上下文、编码失败重采样等防御性工程，与 GRPO 主线弱相关，略 ——
```

两点导读：

- **权重同步是 RL 接 vLLM 的独有麻烦**：vLLM 持有自己一份权重，训练每走一个 global step 都要经 `_move_model_to_vllm()`（L449）把新权重推过去（full 微调推全量；LoRA 默认也推合并后的全量，仅在 `--vllm_enable_lora true` 且 base 已同步时只推增量），并清 prefix cache。魔改后 reward 突然崩掉，先查这条链有没有断。
- **多轮的两种驱动方式**：colocate 模式下多轮由 trainer 侧驱动，`_colocate_multi_turn_infer`（L919）反复调 `agent_loop.run_multi_turn` 与 `scheduler.step()`；server 模式下 trainer 只发请求（L1013，注意请求里不带 scheduler 名字），多轮由 `swift rollout` 启动参数指定的 scheduler 在 server 侧的 `MultiTurnScheduler.run()` 里完成。这就是“动态多轮推荐 server 模式”的原因。

---

## 第 6 层：MultiTurnScheduler 骨架（`swift/rollout/multi_turn.py`，832 行）

多轮 rollout 的编排层。文件骨架如下：`run()` 的默认 while 循环、`step()`/`check_finished()` 的契约、三个内置示例和注册表都在里面。

```python
# swift/rollout/multi_turn.py（共 832 行）
# GRPO 多轮 rollout 的调度器层：GRPOTrainer（server mode）把一批 prompt 发到 rollout 服务，
# 服务端用这里的 scheduler 驱动“模型推理 → 环境/规则反馈 → 再推理”的循环，最终返回 RolloutOutput。

# —— L1-18: imports ——
from swift.infer_engine.protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, RequestConfig,
                                         RolloutInferRequest, RolloutOutput)   # L8-9
# （gym_env、TYPE_CHECKING 下的 GRPOVllmEngine 延迟导入：避免 use_vllm=False 时硬依赖 vllm，略）


class RolloutScheduler(ABC):                                                   # L20：单轮调度器基类
    def __init__(self, infer_engine=None, max_turns=None, *args, **kwargs):    # L22-26
        # 状态只有三个：infer_engine（vLLM 异步引擎）、_tokenizer（colocate 模式下可显式传入）、max_turns

    async def on_trajectory_start(self, requests):                             # L42
        # 首轮推理前的钩子：原地修改 requests（如注入 env 初始观测）。默认 no-op。
        # server mode 的 run() 与 colocate mode 的 run_multi_turn() 都会调它，子类不必重写整个 run()。

    async def on_turn_end(self, infer_request, response_choice, current_turn) -> Dict:   # L50-51
        # 每轮 assistant 消息 append 之后、check_finished 之前的钩子（如 env.step）。
        # 返回 dict：'done' 可覆盖 check_finished 的结果，'rollout_infos' 会 merge 进轨迹级 infos。默认返回 {}。

    async def async_infer(self, infer_requests, request_config, *, use_tqdm=None, **kwargs):   # L65-70
        # GRPO 训练侧的总入口：断言 request_config.n == 1（每条请求只采一条），
        # 把 dict 请求转成 RolloutInferRequest，对每条请求并发地 await self.run(...)，
        # 经 infer_engine._batch_infer_stream 收集后拍平成 List[RolloutOutput] 返回给 trainer。

    async def run(self, infer_request, request_config, **kwargs) -> 'RolloutOutput':   # L137-138
        response = await self.infer_engine.infer_async(infer_request, request_config, **kwargs)   # L139
        # 单轮默认实现：取 choices[0].token_ids，loss_mask 全 1，直接包成 RolloutOutput 返回：
        return RolloutOutput(response=response, messages=infer_request.messages,
                             response_token_ids=[response_token_ids],
                             response_loss_mask=[response_loss_mask],
                             rollout_infos={'num_turns': 1})                   # L143-148

    # （L150-177: __getattr__ 把未知属性委托给 infer_engine，engine/tokenizer property，属胶水代码，略）


class MultiTurnScheduler(RolloutScheduler, ABC):                               # L180：多轮调度器基类
    # 两种定制路线：①整体重写 run()；②只实现 step()（可选重写 check_finished），复用默认 run() 骨架。
    # 关键约定：若返回 response_token_ids，trainer 可跳过对 completion 的重新 encode（规避
    # encode/decode 不对称导致的训练不一致）；response_loss_mask 与其等长对齐，逐 token 控制 loss。

    async def run(self, infer_request, request_config, **kwargs):              # L222-223
        # —— L261-267: 初始化轨迹状态 ——
        current_request = infer_request
        await self.on_trajectory_start([current_request])                      # L262
        # current_turn=1；累积器：rollout_infos、total_response_ids、total_response_loss_mask、total_rollout_logprobs
        while True:                                                            # L268
            # （L270-273: 首轮 remove_response 清掉数据集中残留的 response，略）

            # —— L276-278: 本轮推理 ——
            response = await self.infer_engine.infer_async(current_request, request_config, **kwargs)   # L276
            response_choice = response.choices[0]                              # L278
            # （L280-285: 多轮续写时先移除上一轮留下的 dummy assistant 消息，防御性代码，略）

            # —— L287-294: 更新对话历史 ——
            # 若末尾已是 assistant 则拼接续写（is_continuation=True），否则 append 新消息：
            messages.append({'role': 'assistant', 'content': completion})      # L294

            # —— L296-306: 判定是否终止 ——
            turn_result = await self.on_turn_end(current_request, response_choice, current_turn)   # L297
            # turn_result['rollout_infos'] merge 进累积 infos；turn_result['done'] 优先级最高
            should_stop = self.check_finished(current_request, response_choice, current_turn)      # L300
            # 兜底：即使子类忘了判 max_turns，这里也强制 should_stop |= (current_turn >= self.max_turns)

            if should_stop:                                                    # L308
                # —— L309-347: 收尾 —— 把最后一轮的 token_ids/loss_mask/logprobs 并入累积器
                # （续写则 extend 上一段，首轮即停则初始化）；再校验 logprobs 完整性：
                # logprobs 总数必须等于 loss_mask==1 的 token 数（对应 trainer 里 labels != -100 的
                # completion_mask），不匹配则整体清空，禁用 rollout importance sampling 校正。
                return RolloutOutput(                                          # L349
                    response=response,
                    messages=messages,
                    response_token_ids=total_response_ids,        # 每轮一段 List[List[int]]，trainer 免重新 tokenize
                    response_loss_mask=total_response_loss_mask,  # 与上面逐段等长，0 的 token 不进 loss
                    rollout_infos={**rollout_infos, 'num_turns': current_turn},
                    rollout_logprobs=final_rollout_logprobs,      # 供 rollout 端与训练端的 IS 校正
                )                                                              # L358

            # —— L360-399: 未终止，准备下一轮 ——
            ret = self.step(current_request, response_choice, current_turn)    # L361
            current_request = ret['infer_request']                             # L362
            # 累积 ret 里的 'response_token_ids'（续写 extend / 否则 append 新段）；
            # 'response_loss_mask' 必须与 token_ids 同时返回且等长（assert 检查）；
            # 'rollout_infos' 直接 update 覆盖；logprobs 优先用 step() 返回的（截断/改写后对齐），
            # 否则回退 _extract_logprobs_from_choice(response_choice)。
            current_turn += 1                                                  # L399

    def step(self, infer_request, response_choice, current_turn) -> Dict:      # L401-402
        # 契约：轮间转移逻辑，子类必须实现（否则 raise NotImplementedError，L422）。
        # 返回 dict：infer_request（必填，下一轮的请求）；可选 response_token_ids /
        # response_loss_mask（等长）/ rollout_logprobs（改写 completion 后避免错位）/ rollout_infos。

    def check_finished(self, infer_request, response_choice, current_turn) -> bool:   # L425-426
        # 契约：默认终止条件，两条 —— finish_reason == 'length'（L448）或
        # current_turn >= max_turns（L450）；子类 override 后通常仍 super() 兜底。

    @staticmethod
    def _extract_logprobs_from_choice(response_choice) -> List[float]:         # L454-455
        # 从 response_choice.logprobs['content'] 抽出逐 token logprob 列表，缺失则返回 []。


class ThinkingModelTipsScheduler(MultiTurnScheduler):                          # L471
    # 演示“重写 run() 返回 List[RolloutOutput]”路线：每轮独立产出一个 RolloutOutput（配合
    # --loss_scale last_round 只训最后一轮），历史里只保留最后一轮的 think 内容（_build_messages，L576）。
    # 答错时 step()（L563）append 一条 tips user 消息促使模型重查；check_finished（L547）用 MathAccuracy 判对错。


class MathTipsScheduler(MultiTurnScheduler):                                   # L625
    # 演示“step() + token 级 loss_mask”路线：答错时把 completion 截断到 <answer>/</think> 之前，
    # 拼上 tips_prompt；tips 的 token 记 loss_mask=0（不训练、也不带 logprobs），原 token 记 1（L700-703）。
    # check_finished（L642）：tips 只给一次，答对（MathAccuracy==1）即停，否则 super() 兜底。


class GYMScheduler(MultiTurnScheduler):                                        # L725
    # 演示“universal hooks”路线：on_trajectory_start（L757）按 uuid 建 env 并 env.reset 注入初始观测；
    # on_turn_end（L782）执行 env.step，累积 reward 并经 rollout_infos（total_reward/step_rewards）回传给
    # reward 函数；step()（L809）把 next_obs 作为下一轮 user 消息 append。


# —— L828-832: 注册表 —— 模块级 dict，--multi_turn_scheduler 参数按名字在此查找；
# 自定义 scheduler 通过 external_plugins 向该 dict 添加条目即可接入。
multi_turns = {
    'math_tip_trick': MathTipsScheduler,
    'gym_scheduler': GYMScheduler,
    'thinking_tips_scheduler': ThinkingModelTipsScheduler,
}
```

调度器最终产出的 `RolloutOutput`（定义在 `swift/infer_engine/protocol.py` L484）主要字段：`response`（最后一轮原始响应）、`messages`（完整对话历史）、`response_token_ids`（每轮一段 `List[List[int]]`，提供则 trainer 跳过重新 tokenize）、`response_loss_mask`（与之逐段等长，0 的 token 不进 loss）、`rollout_infos`（轨迹元信息，须可 JSON 序列化，reward 函数可读）、`rollout_logprobs`（每轮 logprobs，用于 rollout IS 校正）；另有 `prompt_logprobs`（引擎侧填充，scheduler 通常不设置）。

### `step()` 返回 dict 字段（完整列表）

```python
{
    'infer_request': RolloutInferRequest,     # 必填：下轮 input
    'response_token_ids': List[int],          # 可选：本轮 response token (override 模型 output)
    'response_loss_mask': List[int],          # 可选：per-token loss mask（0=不参与 loss）
    'rollout_logprobs': List[float],          # 可选：importance sampling correction 用
    'rollout_infos': Dict,                     # 可选：per-step metadata
}
```

**关键**：`response_loss_mask` 让你**选择性 mask 哪些 token 不参与 loss**。常见用途如：在 tool-use / agent 多轮场景里，把“工具返回结果”等非模型生成的 token mask 掉，只对模型真正生成的 token 算 loss，避免把环境注入的内容也当成被优化的对象。

### 定制的三条路线（各对应一个内置示例）

- **只实现 `step()`**（可选 `check_finished()`），复用默认 `run()`：`MathTipsScheduler` 路线，适合“答错给提示再试”这类线性多轮，配合 token 级 `response_loss_mask` 把提示内容排除在 loss 外；
- **重写整个 `run()`**：`ThinkingModelTipsScheduler` 路线，一条轨迹可以产出多个 `RolloutOutput`（配合 `--loss_scale last_round` 只训最后一轮）；
- **只挂 universal hooks**（`on_trajectory_start` / `on_turn_end`），不碰推理循环：`GYMScheduler` 路线，环境交互（`env.reset` / `env.step`）全在钩子里完成，reward 经 `rollout_infos` 回传。

---

## `_compute_loss_and_metrics`：GRPO loss 公式

**`grpo_trainer.py:1090`** 是自定义 loss 修改的核心入口，也就是上文主线骨架里 L1090-L1355 那一段的放大。逐段拆：

### 取 per-token log-prob

```python
per_token_logps, entropies = self._get_per_token_logps_and_entropies(
    model, inputs, compute_entropy=self.compute_entropy)
# Shape: [batch_size, seq_len]
```

### 算 KL

```python
if self.beta != 0.0 and not self.kl_in_reward:
    ref_per_token_logps = inputs['ref_per_token_logps']
    safe_ratio = torch.clamp(ref_per_token_logps - per_token_logps, min=-20, max=20)
    per_token_kl = torch.clamp(torch.exp(safe_ratio) - safe_ratio - 1, min=-10, max=10)
```

### Importance sampling ratio

```python
advantages = inputs['advantages']
old_per_token_logps = inputs['old_per_token_logps']  # 生成时策略的 logps，_prepare_batch_inputs 已预先算好
log_ratio = per_token_logps - old_per_token_logps

if importance_sampling_level == 'token':
    log_importance_weights = log_ratio
elif importance_sampling_level == 'sequence':
    log_importance_weights = ((log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1)).unsqueeze(-1)
elif importance_sampling_level == 'sequence_token':  # GSPO-token
    seq_level_log_weight = ((log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1)).unsqueeze(-1).detach()
    log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weight

coef_1 = torch.exp(log_importance_weights)  # ratio π/π_old
```

### Loss 公式（`loss_type` 选 `grpo`）

```python
coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
per_token_loss1 = coef_1 * advantages.unsqueeze(1)
per_token_loss2 = coef_2 * advantages.unsqueeze(1)
per_token_loss = -torch.min(per_token_loss1, per_token_loss2)  # PPO-clip
if per_token_kl is not None:
    per_token_loss = per_token_loss + self.beta * per_token_kl

# Group normalization
loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
```

用数学符号写出来，单个 token 的 PPO-clip 目标是：

$$
\mathcal{L}_{\text{clip}} = -\min\!\Big( \rho \cdot A,\; \text{clip}(\rho,\, 1-\epsilon_{\text{low}},\, 1+\epsilon_{\text{high}}) \cdot A \Big)
$$

其中 $\rho = \pi\_\theta / \pi\_{\text{old}}$ 是 importance sampling ratio，$A$ 是 advantage。再加上 KL penalty 项 $\beta \cdot D\_{\text{KL}}(\pi\_\theta \mid\mid \pi\_{\text{ref}})$，并在序列维度上用 `completion_mask` 做 token 平均，最后对 batch 取均值。

→ GRPO = PPO-clip + group-relative advantage + KL penalty。

---

## 改 forward pass：从哪里下手

从这里开始进入“动手改”的部分。全部修改都遵循同一个模式：**subclass `GRPOTrainer`、override 目标方法、经 plugin 注册**（注册机制见下一节末尾，一个文件搞定）。先看前向。

按你要改的东西，从浅到深有四个 override 点：

| 想改什么 | override 谁 | 说明 |
| --- | --- | --- |
| 给模型喂额外输入 / 过滤输入 | `_prepare_model_inputs`（L2716） | 决定哪些 key 能进 `model(**inputs)`；在这里塞自定义张量最省事 |
| 给 batch 注入额外张量（随数据流动到 loss） | `_prepare_batch_inputs`（L795） | 先 `super()` 拿到标准张量字典，再往每个 mini-batch dict 加你的 key；loss 里用 `inputs['你的key']` 取 |
| 前向本身（额外输出头、改 logp 计算、不同的温度处理） | `_get_per_token_logps_and_entropies`（L1721） | **推荐入口**：三种 logps（old / ref / current）都从这里走，覆盖它就覆盖了全部前向 |
| 只改路径③的裸前向 | `_get_logps_via_local_forward`（L1676） | 注意陷阱：纯文本模型常走路径①（trl 基类实现），根本不经过这里 |

最常见的坑就是最后一行：`_single`（L1740）里 `can_use_super` 为真时（纯文本 + 非 padding_free + 无序列并行 + 模型支持 `logits_to_keep`），前向直接走 trl 基类，`_get_logps_via_local_forward` 不被调用。所以**要保证自己的前向逻辑必然生效，override 总入口 `_get_per_token_logps_and_entropies`**，在里面自己决定怎么算：

```python
class MyGRPOTrainer(GRPOTrainer):

    def _get_per_token_logps_and_entropies(self, model, inputs, compute_entropy=False):
        # 提醒：old / ref / current 三种 logps 都经过这里。
        # 只想改"带梯度的那次前向"（current），用 torch.is_grad_enabled() 区分：
        if not torch.is_grad_enabled():
            return super()._get_per_token_logps_and_entropies(model, inputs, compute_entropy)
        model_inputs = self._prepare_model_inputs(inputs)
        outputs = model(**model_inputs, output_hidden_states=True)   # 例：额外要 hidden states
        logits = outputs.logits[:, -(inputs['logits_to_keep'] + 1):-1, :] / self.temperature
        completion_ids = inputs['input_ids'][:, -inputs['logits_to_keep']:]
        per_token_logps = selective_log_softmax(logits, completion_ids)
        self._my_hidden = outputs.hidden_states[-1]   # 存给 loss 用（同一小步内 compute_loss 会读到）
        return per_token_logps, None
```

两条纪律：其一，改动后 `per_token_logps` 的形状必须仍是 `[batch, logits_to_keep]`（padding_free 时注意 `_unpad_logps_and_entropies` 的还原约定，L1551-1580）；其二，**温度除法别丢**，丢了 ratio 的分子分母就不同标度了（上文前向一节第 2 条）。

---

## 改学习算法与手动改梯度

### 第一层：换公式（不碰梯度机制）

- **改 loss**：override `_compute_loss_and_metrics`（L1090）。它拿到的 `inputs` 里已经有 `advantages`、`old/ref/rollout_per_token_logps`、`completion_mask`、`truncated_mask`、`num_items_in_batch`，返回 `(loss, metrics_data)` 即可。最省事的写法是照抄原函数再改目标那几行；想在原 loss 上加项，也可以 `loss, m = super()._compute_loss_and_metrics(model, inputs)` 后加。
- **改 advantage / baseline**：override `_compute_advantages`（L412），输入 `rewards_per_func`（全局 gather 后的 `[N, num_funcs]`），返回 `[N]` 的优势张量。组均值、RLOO 留一、std 归一化都在原函数里，可对照改。
- **只是想在现成开关里选**：先翻 `_prepare_algorithm_params`（L2240），`loss_type` / `importance_sampling_level` / `advantage_estimator` / `epsilon_high` 这些命令行参数可能已经覆盖了你要的变体（GSPO、RLOO、DAPO 的件都在）。

### 第二层：手动修改梯度

需要“值是 A、梯度是 B”或者直接篡改梯度时，按优先级有四种机制。**前两种在计算图层面动手，任何分布式包装（DDP / ZeRO / FSDP）下都成立，优先用**；后两种碰 `.grad` 张量，有分布式陷阱。

**机制 1：detach 代数（官方满仓库都在用的手法）。** 原理：`x.detach()` 值不变、梯度断流，于是可以拼出任意的值/梯度组合。通式（straight-through 风格）：

```python
y = value_term.detach() + grad_term - grad_term.detach()
# forward:  y 的数值 == value_term
# backward: dy/dθ == d(grad_term)/dθ
```

这不是理论玩具，`_compute_loss_and_metrics` 里就有五个现成范例，改梯度之前先看它们怎么写：

| 范例 | 行号 | 值/梯度效果 |
| --- | --- | --- |
| GSPO-token：`per_token_logps - per_token_logps.detach() + seq_level_log_weights.detach()` | L1181-1183 | 值＝sequence 级 ratio，梯度＝逐 token 的 $\partial \log \pi$：教科书级的值/梯度分离 |
| CISPO：`torch.clamp(coef_1, max=eps_high).detach()` 乘 `per_token_logps` | L1192-1194 | 裁剪过的 ratio 变成无梯度常数权重，梯度只走 logps：clip 不再产生零梯度死区 |
| on-policy 兜底：`per_token_logps.detach()` 当分母 | L1141-1142 | ratio 值恒 1，但梯度只从分子流：PPO 的标准小技巧 |
| entropy mask / off-policy 序列屏蔽 | L1218-1219 / L1228-1238 | 布尔比较天然截断梯度，乘上去即“值不变、被选中的 token 梯度清零” |
| FIPO：`influence_weight.detach()` 乘进 per_token_loss | L1072、L1216-1217 | 任何“用模型自己算出来的量当权重”都必须 detach，否则权重本身会被优化 |

**机制 2：`torch.autograd.Function` 或 tensor hook。** detach 代数拼不出来的任意梯度变换（比如反向时对梯度做投影、旋转、量化），在 `_compute_loss_and_metrics` 里对中间张量下手：

```python
def _compute_loss_and_metrics(self, model, inputs):
    per_token_logps, entropies = self._get_per_token_logps_and_entropies(model, inputs, ...)
    # 方式 A：hook——正向值不动，反向时梯度经过你的函数
    per_token_logps.register_hook(lambda g: my_transform(g))
    # 方式 B：自定义 Function——正反向完全自主
    per_token_logps = MyGradSurgery.apply(per_token_logps, some_aux)
    ...  # 照常组装 loss

class MyGradSurgery(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, aux):
        ctx.save_for_backward(aux)
        return x                      # 值原样通过
    @staticmethod
    def backward(ctx, grad_out):
        (aux,) = ctx.saved_tensors
        return my_transform(grad_out, aux), None   # 梯度按你的规则改写
```

这一层的修改随计算图走，与 ZeRO/FSDP 的参数切分无关，是**做算法研究时改梯度的正解**。

**机制 3：backward 之后直接改 `param.grad`。** 想在参数粒度上动手（逐层缩放、梯度手术式投影），override `training_step`：

```python
def training_step(self, model, inputs, num_items_in_batch=None):
    loss = super().training_step(model, inputs, num_items_in_batch)   # 返回时 backward 已完成
    if self.accelerator.sync_gradients:            # 只在累积的同步步动手（此时梯度已 all-reduce）
        for p in model.parameters():
            if p.grad is not None:
                p.grad = my_grad_edit(p.grad)
    return loss
```

三个分布式陷阱，务必先对号入座：(i) 混合精度下此刻的 grad 还是 **scaled** 的（unscale 发生在 clip 前），改写要么尺度不变、要么自己处理 scaler；(ii) **DeepSpeed ZeRO stage ≥ 2 下 `param.grad` 是 None**（梯度被 engine 分片管理），要用 `deepspeed.utils.safe_get_full_grad` 一类 API；(iii) **FSDP 下参数是 flat shard**，`.grad` 不能按参数直接寻址。ms-swift 的 GRPO 示例默认配 ZeRO，所以这条路只在单卡 / 纯 DDP 下省心；能用机制 1/2 表达的，别用机制 3。

**机制 4：换 optimizer（在 unscale + clip 之后动手）。** swift 有现成注册表：`--optimizer 名字` 查 `swift/optimizers/mapping.py` 的 `optimizers_map`，plugin 里注册一个 `OptimizerCallback`，返回自定义 `Optimizer`，其 `step()` 在梯度已 unscale、已裁剪之后执行，这是唯一能看到“最终真实梯度”的位置。另有 `--callbacks` 查 `callbacks_map`（swift/callbacks/mapping.py）注册 HF `TrainerCallback` 做旁路监控。

### 注册：让 `swift rlhf` 用上你的 trainer

后文 Plugin 一节会展开 reward / scheduler 的注册表；custom trainer 没有注册表，但**同一个 `--external_plugins` 文件就能完成注册**，机制已逐行核实：

1. plugin 文件在 `BaseArguments.__post_init__` 里被 import（base_args.py L175），**早于** `TrainerFactory.get_training_args`（sft_args.py L232），更早于 `get_trainer_cls`（sft.py L175），所以 plugin 里改 `TRAINER_MAPPING` 一定生效；
2. `import_external_file`（swift/utils/utils.py L401-406）把 plugin 所在目录插进 `sys.path`，并把它 **import 成一个真正的顶层模块**（`my_plugin.py` → 模块 `my_plugin`），所以 plugin 里定义的类有合法的 import 路径；
3. `TrainerFactory.get_cls`（trainer_factory.py L53-55）对映射值做 `rsplit('.', 1)` 再 `importlib.import_module`，所以**注册值必须是字符串路径，不能塞类对象**。

三点拼起来就是完整配方，一个文件全搞定：

```python
# my_plugin.py —— swift rlhf --rlhf_type grpo --external_plugins /path/to/my_plugin.py 即可启用
import torch
from trl.trainer.utils import selective_log_softmax
from swift.rewards import ORM, orms
from swift.rlhf_trainers import GRPOTrainer
from swift.trainers import TrainerFactory

class MyReward(ORM):
    def __call__(self, completions, **kwargs):
        return [float(len(c) < 2000) for c in completions]

orms['my_reward'] = MyReward                       # reward：注册表直接赋值

class MyGRPOTrainer(GRPOTrainer):
    def _compute_loss_and_metrics(self, model, inputs):   # 改学习算法
        ...
    def _get_per_token_logps_and_entropies(self, model, inputs, compute_entropy=False):  # 改前向
        ...

# trainer：没有注册表，改 Factory 的映射；值必须是"可 import 的字符串路径"，
# 因为本文件已被 import 成模块 my_plugin，这个字符串保证可解析：
TrainerFactory.TRAINER_MAPPING['grpo'] = 'my_plugin.MyGRPOTrainer'
```

顺带一提：rollout server（`swift rollout`）与 deploy 进程同样会 import `--external_plugins`（rollout.py L615、deploy.py L264），所以 server 模式下 scheduler 的注册也走同一个文件，两边传同一个路径即可。

不想走 CLI 装配的话，还有更直接的路：自己写 `train.py`，手动构造 `MyGRPOTrainer(model=..., reward_funcs=[...], args=grpo_config, ...)` 然后 `trainer.train()`；代价是 SwiftRLHF 里模型/模板/数据集的准备逻辑要自己搬，一般没必要。


---

## Plugin System：怎么注入 custom 组件

**`examples/train/grpo/plugin/plugin.py`** 是官方示例。**机制**：

### 注册点（4 种）

```python
# 在你自己的 plugin.py 顶部 import 这些 registry（与官方 plugin.py 的写法一致）
from swift.rewards import ORM, AsyncORM, orms, rm_plugins            # 1/3. Reward 函数与 RM plugin
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns  # 2. Multi-turn scheduler
from swift.rollout.gym_env import Env, envs                           # 4. Gym env
```

### 注册示例

```python
# Reward 函数
class MyReward(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        ...
orms['my_reward'] = MyReward

# Multi-turn scheduler
class MyScheduler(MultiTurnScheduler):
    def step(self, ...): ...
    def check_finished(self, ...): ...
multi_turns['my_scheduler'] = MyScheduler
```

### 命令行启用

```bash
swift rlhf \
    --rlhf_type grpo \
    --external_plugins /path/to/my_plugin.py \  # ← 把 plugin.py import 进去
    --reward_funcs my_reward \                   # ← 用注册名
    --multi_turn_scheduler my_scheduler \
    ...
```

`--external_plugins` 在 `args` 初始化时 import 这个 py 文件，触发 `orms[...] = ...` 等注册。

### 内置 reward 一览（`swift/rewards/orm.py`，464 行）

写自定义 reward 之前先看内置的有哪些、契约长什么样：

```python
# swift/rewards/orm.py：GRPO 内置 outcome reward（ORM）的实现与注册表

class ORM:
    # 所有同步 reward 函数的基类；GRPOTrainer 按 --reward_funcs 里的名字实例化后逐 batch 调用
    def __init__(self, args=None, **kwargs):
        self.args = args  # 注入训练参数，reward 超参（如 cosine_max_len）从这里读
    def __call__(self, **kwargs) -> List[float]:
        # 契约：kwargs 含 completions（本组全部 rollout 文本）、数据集列（solution 等按列名注入）、
        # 以及 response_token_ids / trainer_state 等运行时信息；必须返回与 completions 等长的 List[float]
        raise NotImplementedError

class AsyncORM:
    # 异步版基类：__call__ 为 async def，trainer 用 asyncio.gather 并行执行，适合调外部 API 打分
    async def __call__(self, **kwargs) -> List[float]: ...

class MathAccuracy(ORM):
    # 数学正确性：math_verify 解析 <answer>…</answer> 或 \boxed{} 里的 LaTeX，与 solution 做符号等价校验，对 1 错 0
    def __call__(self, completions, solution, **kwargs) -> List[float]: ...

class Format(ORM):
    # 格式奖励：整条 completion 严格形如 <think>…</think><answer>…</answer> 给 1，否则 0
    def __call__(self, completions, **kwargs) -> List[float]: ...

class ReActFormat(ORM):
    # ReAct 格式奖励：<think>…</think> 后接 Action: 与 Action Input: 给 1，否则 0
    def __call__(self, completions, **kwargs) -> List[float]: ...

class CosineReward(ORM):
    # 长度整形（arXiv:2502.03373）：先用 accuracy_orm 判对错，再按 token 长度做余弦插值，答对越短奖励越高，答错越长惩罚越轻
    def __call__(self, completions, solution, **kwargs) -> List[float]: ...

class RepetitionPenalty(ORM):
    # 重复惩罚（同上论文）：按 n-gram 重复比例乘 max_penalty 给非正分，重复越多罚越重
    def __call__(self, completions, **kwargs) -> List[float]: ...

class SoftOverlong(ORM):
    # DAPO 式软超长惩罚：token 数超过 soft_max_length - soft_cache_length 后按超出量线性给负分
    def __call__(self, completions, **kwargs) -> List[float]: ...

class ReactORM(ORM):
    # ToolBench 工具调用打分：解析 Action / Action Input，与 ground truth 逐键比对参数（F1、纯文本用 ROUGE-L）
    ...

class MathORM(ORM):
    # 旧版数学打分：抽取 \boxed{} 后用 sympy（或 opencompass 的 MATHEvaluator）判等价
    ...

# --reward_funcs accuracy format cosine … 查的就是这张表；外部插件往里加键即可扩展
orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
}
```

### 官方插件范本长什么样（`examples/train/grpo/plugin/plugin.py`，1228 行）

这个文件是所有自定义组件的照抄模板，骨架如下（注册名都是真实的，可直接在命令行引用）：

```python
# examples/train/grpo/plugin/plugin.py：官方自定义插件示例，训练或 rollout 时用 --external_plugins 加载
# 文件被 import 的那一刻，下面的注册语句就把自定义组件写进四张注册表：orms / rm_plugins / multi_turns / envs

from swift.rewards import ORM, AsyncORM, orms, rm_plugins
from swift.rewards.rm_plugin import DefaultRMPlugin
from swift.rollout.gym_env import Env, envs
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns

# —— L39-893: 自定义 reward 函数，训练时用 --reward_funcs <注册名> 引用 ——
class CountdownORM(ORM):  # Countdown 数字游戏：校验 <answer> 里的算式恰好用完 nums 且结果等于 target
    def __call__(self, completions, target, nums, **kwargs) -> List[float]: ...  # 数据集列按列名直接进形参
orms['external_countdown'] = CountdownORM

class MultiModalAccuracyORM(ORM):  # 多模态 accuracy：先 math_verify 符号校验，失败再退回 <answer> 字符串比对
orms['external_r1v_acc'] = MultiModalAccuracyORM

class MultiTurnThinkingTips(ORM):  # 多轮拆段示例：按 request_id 取整条 trajectory，只对最后一轮算 accuracy，各段共享同一 reward
orms['thinking_tips'] = MultiTurnThinkingTips

class CodeReward(ORM):  # E2B 云沙箱执行生成的代码，按 verification_info 里测试用例的通过率打分
orms['external_code_reward'] = CodeReward

class CodeFormat(ORM):  # 代码格式：<think>…</think><answer> 内含对应语言代码块给 1，否则 0
orms['external_code_format'] = CodeFormat

class CodeRewardByJudge0(ORM):  # 同样是执行代码判分，改用 Judge0 API（沙箱方案的替代）
orms['external_code_reward_by_judge0'] = CodeRewardByJudge0

class AsyncGenRMReward(AsyncORM):  # 异步 LLM-as-judge：aiohttp 并发调 swift deploy 起的生成式 RM，从回复里抽 [[score]]
orms['async_genrm'] = AsyncGenRMReward

# ToolRL（arXiv:2504.13958）三件套，可用环境变量按 trainer_state.global_step 做两阶段或动态 reward 调度
class ToolUseFormatReward(ORM):  # 格式：按 solution 类型检查 <think> 加 <tool_call>/<response> 标签结构是否恰好匹配
orms['external_tooluse_format_reward'] = ToolUseFormatReward
class ToolUseLengthReward(ORM):  # 长度：<think> 段词数越多分越高，达到 max_reward_len 封顶
orms['external_tooluse_length_reward'] = ToolUseLengthReward
class ToolUseCorrectnessReward(ORM):  # 正确性：解析出的工具名与参数逐项和 ground truth 匹配计分
orms['external_tooluse_correct_reward'] = ToolUseCorrectnessReward

# —— L913-1049: 自定义 reward model plugin，用 --reward_model_plugin <注册名> 引用 ——
class CustomizedRMPlugin:  # 与 DefaultRMPlugin 相同：分类头 RM 前向一次，取 logits[:, 0] 作为分数
class QwenLongPlugin(DefaultRMPlugin):  # QwenLong-L1（arXiv:2505.17667）混合奖励：LLM 判 [[YES]]/[[NO]] 与规则 accuracy 取 max
rm_plugins['my_rmplugin'] = CustomizedRMPlugin
rm_plugins['qwenlong'] = QwenLongPlugin

# —— L1075-1220: 自定义多轮调度器，rollout 侧用 --multi_turn_scheduler <注册名> 引用 ——
class ToolCallScheduler(MultiTurnScheduler):  # 覆写 check_finished/step：解析 ReAct 工具调用并实际执行（内置 calculator）
    def step(self, infer_request, response_choice, current_turn) -> Dict:
        # 工具结果拼回消息与 token_ids，对应 loss_mask 置 0：环境反馈的 token 不参与 GRPO loss
multi_turns['tool_call_scheduler'] = ToolCallScheduler

# —— L1224-1228: 注册 GYM env（占位示例）——
class CustomEnv(Env): pass
envs['custom_env'] = CustomEnv
```

### Custom Trainer 怎么办（没有注册表，但有正规注册路径）

`TRAINER_MAPPING` 是硬编码字典，plugin 注册表里没有 trainer 一项。但 plugin 文件在 Factory 查表**之前**就被 import，且会成为一个可 import 的真模块，所以在 plugin 里直接改映射就是正规做法（值必须是字符串路径）：

```python
from swift.trainers import TrainerFactory
TrainerFactory.TRAINER_MAPPING['grpo'] = 'my_plugin.MyGRPOTrainer'
```

时机与机制的逐行核实、以及单文件完成 reward + trainer 全部注册的配方，见上文“改学习算法与手动改梯度”末尾的注册小节。

---

## vLLM Rollout 数据流（colocate vs server）

### Colocate 模式

```
[GPU 0..7] 每张 GPU 同时运行 training + vLLM engine
   training step → 暂停 → vLLM rollout（用同 GPU）→ 收 result → 继续 training
   优点：省 GPU
   缺点：rollout 和 training 不能并行；多轮由 trainer 侧驱动（agent_loop.run_multi_turn），
         不支持动态 rollout 数等高级多轮特性
```

### Server 模式

```
[Training GPUs 0..3] training loop ──HTTP──→ [Rollout GPUs 4..7] swift rollout 进程
                                                                  │
                                                                  └─ vLLM engine
   优点：rollout 和 training 异步；支持 multi-turn dynamic rollout；MultiTurnScheduler.run() 完整支持
   缺点：占用更多 GPU；通信开销
```

→ 需要动态轮数（每个 trajectory 轮数不定）的多轮 rollout 时，应选 server 模式。

### Constrained decoding（guided JSON）

**`vllm_engine.py:42, 511-519`**：本版本经 `RequestConfig` 暴露的是**正则约束**：`protocol.py` L212 的 `RequestConfig` 只有 `structured_outputs_regex: Optional[str]` 这一个结构化输出字段，`vllm_engine.py` L513-519 读取它并在内部包装成 vLLM 的 structured outputs 参数（vLLM v0.12+ 叫 `structured_outputs`，老版本叫 `guided_decoding`）：

```python
request_config.structured_outputs_regex = r'<answer>(yes|no)</answer>'
```

vLLM 拿到约束后在每步采样时做 logit mask，强制输出匹配该正则。注意：JSON schema 级别的约束在这个版本没有从 `RequestConfig` 打通，需要把结构表达成正则，或自行 patch engine。

---

## 魔改指南：想改什么，就去哪儿

机制讲完了，这一节把“从哪下手”收成一张速查表。

| 想改的东西 | 下手位置 | 方式 | 侵入程度 |
| --- | --- | --- | --- |
| 打分规则 | reward 函数 | 写 `ORM` 子类 → `orms['name']` 注册 → `--reward_funcs name` | 纯 plugin，零改框架 |
| 多轮交互逻辑（何时停、下轮 prompt 怎么拼、如何插入环境反馈） | `MultiTurnScheduler` | 子类实现 `step()` / `check_finished()` → `multi_turns` 注册 | 纯 plugin |
| 哪些 token 参与 loss | `response_loss_mask` | 在 `scheduler.step()` 返回值里逐轮给出 | 纯 plugin |
| loss 公式本身 | `_compute_loss_and_metrics` | 子类化 `GRPOTrainer` 后 override | subclass + plugin 注册 |
| advantage / baseline 的定义 | `_compute_advantages` | 同上 | 同上 |
| 训练前向（额外输入输出、logp 算法） | `_get_per_token_logps_and_entropies` | 子类 override，见“改 forward pass”一节 | subclass + plugin 注册 |
| 梯度本身（值/梯度分离、反向改写） | `_compute_loss_and_metrics` 里的 detach 代数 / hook / autograd.Function | 见“改学习算法与手动改梯度”一节 | subclass |
| 换 trainer（承载以上所有 subclass） | `TrainerFactory.TRAINER_MAPPING` | plugin 里赋值字符串路径 'my_plugin.MyGRPOTrainer' | 纯 plugin |
| 采样行为（温度、n、结构化约束） | `request_config` / `structured_outputs_regex` | 命令行参数，或在 scheduler 里按轮修改 | 命令行 / plugin |
| 推理后端 | engine | `--vllm_mode` 或换 engine 类 | 命令行 |
| 数据集 | dataset | `messages` 列必备；自定义列会原样透传进 reward 函数的 `**kwargs` | 数据侧 |
| 训练期指标 | `rollout_infos` / metrics | scheduler 或 reward 里塞 `rollout_infos`，或 subclass 后加 log | plugin 或 subclass |

**三条纪律**，能省掉大多数排障时间：

1. **别碰上层装配。** CLI / Pipeline / Factory 是装配代码，绝大多数定制不需要动它们；要动 Trainer 也优先 subclass 而不是改源码，否则跟随 ms-swift 升级会很痛苦。
2. **token 对齐高于一切。** 多轮场景一律在 scheduler 里逐轮累积 `response_token_ids` / `response_loss_mask`，不要事后拿文本重新 `encode`（编解码不对称，见 Tokenizer 一节）；凡不是模型生成的内容（环境反馈、工具输出、注入的提示），一律用 mask 排除在 loss 之外。
3. **先小后大。** 先用小模型、单 GPU、colocate（甚至不开 vLLM，让它退回 transformers 后端，慢但可断点调试）把整条链路跑通，再上 vLLM server 加多卡；调试期把 `--logging_steps 1` 打开，先盯 completions 样例和 reward 分布是否符合预期，再去看 loss 曲线。

**新手常见坑**：

- **整除关系**：有效 batch 必须能被 `num_generations` 整除，否则分组直接报错；
- **buffered rollout**：`steps_per_generation > 1` 时，一次 rollout 供多个梯度步复用，看到“loss 在动而 completions 没变”是正常现象，不是 bug；
- **权重没同步**：动过 vLLM 相关代码后 reward 突然崩掉，先检查每次参数更新后 `_move_model_to_vllm()` 是否仍被调用；
- **std = 0 的组**：一组回答 reward 全相同（全对或全错）时 advantage 全为 0、没有梯度信号，可用 `dynamic_sample` 过滤重采；
- **KL 爆炸**：`beta` 非 0 时若 ref model 或 template 配置不一致，KL 会异常巨大，先对齐 template 再怀疑算法本身。

---

## 启动命令完整示例

```bash
# 最小可跑（单机单卡 colocate，自带 accuracy reward，用来确认链路通）
# 参数名以所装版本 swift rlhf --help 为准
CUDA_VISIBLE_DEVICES=0 swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-0.6B \
    --dataset AI-MO/NuminaMath-TIR#1000 \
    --reward_funcs accuracy \
    --num_generations 4 \
    --per_device_train_batch_size 4 \
    --max_completion_length 1024 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --output_dir output/grpo_smoke
```

跑通之后，再换成下面带 plugin、多轮与 server 模式的完整配置。

```bash
# Server 模式 (推荐 multi-turn)
# Terminal 1: 启动 rollout server (GPU 4-7)
CUDA_VISIBLE_DEVICES=4,5,6,7 swift rollout \
    --model Qwen/Qwen3-8B-Instruct \
    --vllm_max_model_len 8192 \
    --vllm_use_async_engine true \
    --vllm_gpu_memory_utilization 0.9 \
    --port 8000

# Terminal 2: 启动 training (GPU 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-8B-Instruct \
    --tuner_type full \
    --external_plugins /path/to/my_plugin.py \
    --reward_funcs my_reward \
    --multi_turn_scheduler my_scheduler \
    --max_turns 10 \
    --vllm_mode server \
    --vllm_server_host localhost \
    --vllm_server_port 8000 \
    --dataset /path/to/dataset \
    --num_generations 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --max_length 8192 \
    --num_train_epochs 1 \
    --logging_steps 5 \
    --output_dir output/grpo
```

---

## 核心结论

ms-swift GRPO 管线 = (1) `SwiftRLHF` pipeline 加载 model/data → (2) `TrainerFactory` 选 `GRPOTrainer` → (3) `GRPOTrainer.__init__` 准备 rollout 与 reward → (4) HF Trainer.train 循环 → (5) `_prepare_inputs` 触发 `_generate_and_score_completions`（rollout + reward + advantage）→ (6) `compute_loss` → `_compute_loss_and_metrics` 算 GRPO PPO-clip loss → (7) backward + optimizer.step。

整个框架的关键扩展点也由此清晰：`MultiTurnScheduler`（控制多轮 rollout）+ `_compute_loss_and_metrics`（自定义 loss）+ reward 函数（plugin 注册），三者组合即可覆盖绝大多数 GRPO 定制需求，而不必改动 Pipeline / Factory / CLI 等上层装配代码。更深的改动（训练前向、梯度本身、整个 learning algorithm）同样有正规入口：subclass `GRPOTrainer`，在计算图层面用 detach 代数或 autograd.Function 做梯度手术，再经 `--external_plugins` 把 `TRAINER_MAPPING` 指向你的类，全程不动 ms-swift 源码一行。
