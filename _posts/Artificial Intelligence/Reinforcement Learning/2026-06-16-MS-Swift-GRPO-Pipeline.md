---
title: "MS-Swift GRPO Pipeline Walkthrough"
date: 2026-06-16 10:00:00 +0800
categories: [Artificial Intelligence, Reinforcement Learning]
tags: [Tech, AI, RL, LLM, MS-Swift, GRPO]
math: True
description: 基于源码 walkthrough,逐层拆解 ms-swift 的 GRPO 训练管线:从 CLI 入口、Pipeline、Factory、Trainer,到 rollout、reward、advantage、GRPO loss 公式与 plugin 扩展机制。
---

> 本文基于 ms-swift v4.4.0-dev 源码,逐层 trace GRPO 训练管线的逻辑流。由 Claude Code (Opus 4.8, ultracode 多智能体编排) 整理,作者审校。
{: .prompt-info }

## 一图概览整个调用链

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
   │   class GRPOTrainer(RolloutTrainerMixin, _GRPOTrainer):
   │      __init__:
   │          prepare_rollout()            ← MultiTurnScheduler 初始化在这里
   │          _prepare_rewards()
   │          _prepare_metrics()
   │      train()  (继承 HF Trainer.train)
   │          for step in steps:
   │              inputs = sampler.next()
   │              inputs = _prepare_inputs(inputs)
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
       支持 guided_decoding (Pydantic schema → vLLM structured_outputs)
```

---

## 上面那些名字到底是什么（概念 + 代码职责）

如果你只看上面那张图会觉得每个方框只是一个英文标签。这一节把每个名字**当代码里的一个角色 / 一个 class / 一个文件**来讲：它**拥有什么状态**、**对外提供什么方法**、**为什么必须独立存在而不合并到上下游**。

### Tokenizer（最底层：文字 ↔ 数字的翻译器，先讲它因为下面全靠它）

- **是什么**：一个把**字符串和整数 ID 序列双向翻译**的对象。神经网络只会算数字，不认字；tokenizer 就是文字和数字之间唯一的桥。每个 model 配套自己的一个 tokenizer，**不能跟别的 model 混用**（像配套的密码本）。
- **核心机制：subword（子词）切分**。它既不是按字符切，也不是按单词切，而是按"高频子词"切（BPE / WordPiece / Unigram / SentencePiece 等算法）。比如 `"signaling"` 可能被切成 `["sign", "aling"]` 两个 token，`"96"` 可能是一个 token，中文一个字常常是一到几个 token。背后有一张固定大小的**词表（vocabulary）**，每个 token 对应一个整数 ID（Qwen3 词表约 15 万）。
  - 为什么用 subword 而不是整词：词表能保持可控大小，又能拼出没见过的词（拆成已知子词），还天然多语言友好。
- **对外提供的两个方法**：
  - `encode(text) -> List[int]`：文字变 token id 序列（喂给模型 forward 之前必做）
  - `decode(List[int]) -> text`：token id 序列变回文字（模型吐 token 后给人看）
- **拥有什么状态**：基本是只读的纯函数式对象（词表 + 切分规则固定）。被 template / engine / trainer / scheduler **共享同一个实例**。它属于 model 的一部分。
- **为什么贯穿所有环节（这才是关键）**：本文后面反复出现的 "token" 单位全是 tokenizer 切出来的：
  - 模型 forward 吃的是 token id，不是文字 → 所有 prompt / completion 都先过 `encode`
  - `per_token_logps` 的 "per_token" 就是 per 这个切分单位，每个 token id 一个 log prob
  - `completion_mask` 标记哪些 token id 参与 loss，单位也是 token
  - scheduler 返回的 `response_token_ids` 就是 `encode` 出来的整数序列
  - vLLM 的 guided decoding 做 logit mask 也是在 token 粒度上屏蔽不合法 token id
- **一个微妙但重要的陷阱**：`encode(decode(ids))` **不保证等于** `ids`（编解码不对称，因为多种 token 切法可能 decode 成同一串文字）。所以 ms-swift 的 scheduler **优先直接累积并返回 `response_token_ids`**，而不是让 trainer 把文本重新 `encode` 一遍，因为后者可能切出不同的 token 边界，导致训练时 log prob 对不齐。这也是自定义 scheduler 应该在每轮逐步累积 `token_ids` 而不是事后重新编码的原因。
- **token 级标签的实现要点**：如果你要给 response 里某些片段（比如 JSON 中某个字段）打 token 级标签，必须用 tokenizer 的 **offset mapping**（每个 token 对应原文哪几个字符），把"JSON 里第几个字符"反查成"第几个 token"。
- **类比**：电报的摩斯密码本。收发双方必须用**同一本**密码本，否则译出来是乱码；model 和它的 tokenizer 就是这种绑定关系。

### CLI（命令行入口）

- **是什么**：一个极薄的 shell-to-Python 适配层。`swift/cli/rlhf.py` 全文 5 行，作用是 `python -m swift.cli.rlhf args...` 时把 `args` 解析后调用 `swift.pipelines.rlhf_main(args)`。
- **拥有什么状态**：基本没有（无状态）；只负责"安装命令、设置好 fork-safe 单设备模式、调 main"。
- **为什么独立**：把 `setup.py` 的 entry point 和真正的训练逻辑解耦。后者可以独立用 Python 直接调（不走 shell）。
- **类比**：Linux 的 `/usr/bin/git` 命令本身只是个 dispatcher，真正的逻辑在 `libgit2`。

### Pipeline（流程编排器）

- **是什么**：一个高阶的"训练任务跑通从头到尾要走哪几步"对象。`SwiftRLHF` 继承自 `SwiftSft`，一个 instance 对应一次训练任务。
- **拥有什么状态**：解析过的 `args` (RLHFArguments)、加载好的 `model` / `ref_model` / `reward_model` / `tokenizer` / `template` / `dataset`。
- **对外提供的方法**：`run()`，里面顺序做这些事：(1) 下载/加载模型权重；(2) 加载 dataset；(3) 拼装 template；(4) 调 `TrainerFactory` 选 trainer 类；(5) 把上面准备好的东西塞进 trainer 构造函数；(6) `trainer.train()`。
- **为什么独立**：训练任务的"准备阶段"（model/data/template 加载）和"训练循环本身"（forward/backward）是两件事。把准备阶段放在 Pipeline 里，trainer 就可以专注训练循环；新增一种 RL 算法只需要写新 trainer，不用重复实现 model 加载。
- **类比**：电影制片厂的"项目经理"角色：选演员、定场地、定预算，然后把开机交给导演（trainer）。

### Factory（工厂）

- **是什么**：一个查表函数，输入是 args（含 `--rlhf_type grpo`），输出是**要 instantiate 的 trainer 类**（不是 instance，是 class 对象）。
- **拥有什么状态**：一个硬编码的字典 `TRAINER_MAPPING = {'grpo': 'swift.rlhf_trainers.GRPOTrainer', 'dpo': '...', 'ppo': '...', ...}`。
- **对外提供的方法**：`TrainerFactory.get_trainer_cls(args)` → 反射 import 对应 class。`get_training_args(args)` 同理返回 trainer 专属的 config dataclass。
- **为什么独立**：一个 CLI (`swift rlhf`) 要支持很多 RL 算法（DPO/PPO/GRPO/KTO/CPO/ORPO/RM/GKD），每个算法用不同 trainer 类。Factory 把 "`--rlhf_type` 字符串 → trainer 类" 的映射集中在一处，避免 Pipeline 里写一堆 `if args.rlhf_type == 'grpo': ...`。
- **类比**：编程语言里的 design pattern "Factory Method"，给名字、返回对象。

### Trainer（训练循环 owner）

- **是什么**：训练循环本身的载体。`GRPOTrainer` 是 `_GRPOTrainer`（继承 HF `Trainer`）+ `RolloutTrainerMixin`。一个 instance 对应一次训练任务。
- **拥有什么状态**：`self.model`、`self.ref_model`、`self.reward_funcs`、`self.optimizer`、`self.lr_scheduler`、`self.train_dataloader`、`self.template`、`self.multi_turn_scheduler`、`self.infer_engine`、`self.accelerator`（DDP / FSDP / DeepSpeed 包装器）、当前 step 计数、buffered rollout 数据 …… 几乎所有训练相关的状态都在这。
- **对外提供的方法**：
  - `train()` → 跑整个 epoch loop，是顶层入口
  - `_prepare_inputs(batch)` → 把 dataset 里的原始 batch 变成 forward 能用的 tensor（GRPO 里这一步会触发 rollout + reward + advantage 计算）
  - `compute_loss(model, inputs)` → 算 loss，HF Trainer 通过这个 hook 调子类
  - `_compute_loss_and_metrics(model, inputs)` → GRPO 自家加的，真正写 GRPO loss 公式的地方
  - `training_step()` / `optimizer_step()` → 继承自 HF，标准 backward+step
- **为什么独立于 Pipeline**：Pipeline 是"装东西的盒子"，Trainer 是"开机后跑循环的状态机"。一个 Pipeline run 里只 instantiate 一个 Trainer，但同样一个 Trainer 类可以被 Pipeline、Jupyter notebook、Ray actor、单测脚本反复 instantiate。
- **类比**：导演 + 整个剧组 + 剧本 + 摄影机的合体。Pipeline 把人和设备凑齐交给它后，喊"Action"开始拍。

### Rollout（采样轨迹 = RL 的训练数据生产过程）

- **概念**："Rollout" 在 RL 里就是**让当前 policy 在环境/任务上跑一遍、产生 (state, action, reward) 序列**。在 LLM RL 里更具体：**用当前 model 对一批 prompt 做 generation，得到 completion（再算 reward）**。
- **为什么 RL 必须有 rollout 而 SFT 不需要**：SFT 的训练数据是预先标注的（prompt + 标准答案），不需要模型自己生成；RL 的训练数据**取决于当前 policy**，每改一次权重就要重新采样。"Rollout" 这个步骤就是 RL 的数据生产环节。
- **代码里在哪**：
  - `GRPOTrainer._generate_and_score_completions()` 是高阶 entry，负责"采一批 + 算 reward + 算 advantage"
  - `_generate_completions()` 负责真正调 inference
  - `RolloutTrainerMixin._fast_infer()` / `_server_rollout()` / `_colocate_rollout()` 负责把 rollout 请求分发给 inference engine
  - `MultiTurnScheduler.run()` 负责"一个 trajectory 内部多轮 inference 怎么串"
- **状态**：rollout 过程不持久化 model 权重（只读 model），但产生一堆 `RolloutOutput` 对象（含 messages / token_ids / loss_mask / logprobs / rollout_infos），这些就是后续 `compute_loss` 的输入。
- **为什么独立于 trainer 主循环**：rollout 用的是**当前 model 权重的快照**，可以批量做、可以放到不同 GPU、可以异步流水线。把它从 backward 路径里分离出来，让两边的并行度独立调优（rollout 用 vLLM tensor parallel，training 用 DDP/FSDP）。
- **类比**：拍电影里的"先拍一段素材，再剪辑"。Rollout 是拍素材，loss/backward 是剪辑。

### Engine（推理后端）

- **是什么**：负责"把一组 prompt → 一组 completion"这件事的执行器。可换实现：`VllmEngine` / `SglangEngine` / `LmdeployEngine` / `TransformersEngine`（HF generate 的薄包装）。
- **拥有什么状态**：加载的 model 权重副本（vLLM 内部有自己的 KV cache + 调度器）、tokenizer、采样参数模板、HTTP 端口（server 模式）。
- **对外提供的方法**：`infer(requests, request_config)` 同步版、`infer_async(...)` 异步版、`_batch_infer_stream(...)` 流式 batch。
- **为什么独立于 trainer**：trainer 不关心是 vLLM 还是 SGLang 在跑 inference；engine 也不关心调用方是 GRPO 还是 PPO。一个清晰的 `infer_async(request) → response` 接口让两边都能换。**vLLM colocate 模式**下 engine 和 trainer 在同一进程同一 GPU；**vLLM server 模式**下 engine 是独立进程，trainer 通过 HTTP 调它（`VllmClient`）。
- **类比**：相机机身 vs 镜头。导演（trainer）下"拍这帧"指令；不同的镜头（engine）能拍但成像方式不同。

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
- **类比**：vLLM 是 Engine 这个"相机机身接口"的一款高性能具体型号；SGLang 是另一款，Transformers 是慢的兜底款。

### MultiTurnScheduler（多轮 rollout 编排）

- **是什么**：一个**状态机抽象基类**，把"一个 trajectory 内部要打几轮 inference、每轮 prompt 怎么变、何时停"这件事独立出来。自定义多轮逻辑就继承它。
- **拥有什么状态**：`self.infer_engine`（指向上面的 engine）、`self.max_turns`、`self.tokenizer`。**注意**：每个 trajectory 的状态（messages 历史、累计 token_ids、当前 turn 计数）放在 `run()` 的局部变量里，不放 instance attribute，因为同一个 scheduler 实例会被并发跑多 trajectory（async）。
- **对外提供的方法**：
  - `run(request, cfg)` → 跑完整个 trajectory，返回 `RolloutOutput`。**默认实现**是一个 while 循环；想完全自定义就 override 这个方法。
  - `step(request, response, turn)` → 给"下一轮的 request 长什么样"。**默认行为是抽象**，子类必须实现。
  - `check_finished(request, response, turn)` → 是否应该停。默认按 `finish_reason='length'` 或 `current_turn >= max_turns`。
- **为什么独立于 trainer**：HF Trainer 假设"一个 input → 一次 generate → 一个 output"（单轮）。多轮场景（tool use、agent 任务、多角色对话等）需要在 generate 之前**重新构造 prompt**、追加 history、决定下一步调谁。这一段逻辑跟 trainer 无关，跟具体任务有关，所以单独拎成一个可插拔的 class。
- **类比**：拍多镜头连续剧每一集的"分镜表"。每一集（trajectory）有几个镜头（turn）、每个镜头拍什么、什么时候切镜头，分镜师（scheduler）说了算；摄影师（engine）只负责按分镜表拍。

### Reward function（奖励函数 / Reward Model）

- **是什么**：一个 callable 或 nn.Module，输入 completion 字符串（和可选的 dataset 列），输出 float reward。自定义 reward 通常继承 `ORM`。
- **为什么必须独立**：reward 的定义和算法（GRPO/PPO/…）正交。同一个 reward 函数（如 "math 答案是否正确"）可被任意 RL 算法复用；同一个 RL 算法可以挂任意 reward。
- **数据流上的位置**：rollout 完成之后、advantage 计算之前。`_score_completions()` 调用 reward function，得到 `rewards_per_func` tensor。
- **类比**：考试的评卷标准。学生（policy）怎么写题是模型自己事；评卷规则（reward）独立于"用什么教学方法（RL 算法）"。

### Advantage（优势 = 归一化后的 reward）

- **是什么**：`advantage = reward - baseline`，作为 policy gradient 公式里 $\nabla \log p \cdot \text{advantage}$ 的标量权重。GRPO 的特色是**用 group mean 当 baseline**：同 prompt 跑 `num_generations` 个 rollout，组内归一化。这样省掉了 PPO 需要的额外 critic network。
- **为什么必须独立**：reward 的绝对值意义模糊（10 分还是 100 分？）、variance 大；归一化后才是稳定的梯度信号。
- **代码里在哪**：`GRPOTrainer._compute_advantages()`，输出 `[batch_size]` 的 tensor 塞进 `inputs['advantages']`。

### per_token_logps / completion_mask / KL（loss 公式的三大原料）

- **per_token_logps**：`[batch_size, seq_len]`，每个 token 在当前 model 下的 log probability。是 forward 跑出来的，loss 公式的核心输入。
- **completion_mask**：`[batch_size, seq_len]`，01 mask，标记哪些 token 是"模型生成的、应该参与 loss"（completion 部分）vs "prompt / system message 部分，不参与 loss"。如果要对 response 内部不同片段做差异化 loss，也是在这一层叠加更细的 mask。
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
                                          (scheduler 是分镜师，engine 是相机)
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

这套设计的好处是：自定义扩展通常只需走"换 scheduler + 换 reward + 子类化 Trainer"路径，没必要碰 Pipeline / Factory / CLI 这些上层装配代码。

---

## 调用栈逐层拆解

### CLI Entry: `swift rlhf`

**`swift/cli/rlhf.py`**（5 行）：

```python
if __name__ == '__main__':
    from swift.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    from swift.pipelines import rlhf_main
    rlhf_main()
```

`swift` 命令通过 `setup.py` 注册 entry point；`swift rlhf <args>` 等价于 `python -m swift.cli.rlhf <args>`。

### Pipeline: `SwiftRLHF`

**`swift/pipelines/train/rlhf.py:245`**：`rlhf_main(args) -> SwiftRLHF(args).run()`

```python
class SwiftRLHF(SwiftSft):  # ← 继承 SFT pipeline，复用 model/data loading
    args_class = RLHFArguments

    def _prepare_single_model(self, key, origin_key, model_type, model_revision):
        # 根据 key 加载 actor / ref / reward / teacher model
        # ref/reward/teacher 模型 .requires_grad_(False).eval()
        ...

    def run(self):
        # 1. 准备 dataset / template / tokenizer
        # 2. 准备 model (actor) + ref_model + reward_model
        # 3. TrainerFactory.get_trainer_cls(args) → GRPOTrainer
        # 4. trainer = GRPOTrainer(model, ref_model, reward_funcs, args=training_args)
        # 5. trainer.train()
```

### Factory: 决定用哪个 Trainer

**`swift/trainers/trainer_factory.py`**：

```python
TRAINER_MAPPING = {
    'dpo': 'swift.rlhf_trainers.DPOTrainer',
    'ppo': 'swift.rlhf_trainers.PPOTrainer',
    'grpo': 'swift.rlhf_trainers.GRPOTrainer',
    'gkd': 'swift.rlhf_trainers.GKDTrainer',
    ...
}

@classmethod
def get_trainer_cls(cls, args):
    return cls.get_cls(args, cls.TRAINER_MAPPING)
    # 根据 args.rlhf_type 反射加载
```

`--rlhf_type grpo` 触发 GRPOTrainer 加载。

---

## GRPOTrainer.__init__ 拆解

**`swift/rlhf_trainers/grpo_trainer.py:87-180`**：

```python
class GRPOTrainer(RolloutTrainerMixin, _GRPOTrainer):
    def __init__(self, model, ref_model=None, reward_model=None, reward_funcs=None, **kwargs):
        args: GRPOConfig = kwargs['args']
        self.args = args
        self.vllm_client = kwargs.pop('vllm_client', None)

        # 1. 算法参数（CHORD, RLOO, GSPO, etc. 的开关）
        self._prepare_algorithm_params()

        # 2. 父类初始化（HF Trainer + 模型 setup）
        super().__init__(model, ref_model, *_args, **kwargs)

        # 3. CHORD SFT dataset（如果用 CHORD loss mixing）
        self._prepare_chord_dataset()

        # 4. ★ Rollout 准备（含 MultiTurnScheduler 初始化）
        self.prepare_rollout()

        # 5. Reward 函数 / Reward model 准备
        self._prepare_rewards(reward_funcs, reward_model, reward_templates)

        # 6. Liger loss / metrics
        self._prepare_liger_loss()
        self._prepare_metrics()

        # 7. Misc: seed, transformers fallback engine, etc.
        ...
```

### `prepare_rollout()`（Mixin 提供）

**`swift/rlhf_trainers/rollout_mixin.py:108`**：

```python
def prepare_rollout(self):
    self._prepare_rollout_params()       # request_config, num_generations, temperature
    self._prepare_scheduler()             # ← MultiTurnScheduler 注入点
    self._prepare_vllm()                  # vLLM engine (colocate or server)
    self._prepare_async_generate()        # async vLLM
    self.parameter_groups, self.parameter_groups_no_lora = self.split_batches()
```

### `_prepare_scheduler()` 解析

**`rollout_mixin.py:1315-1340`**：

```python
def _prepare_scheduler(self):
    self.multi_turn_scheduler = None
    if args.multi_turn_scheduler:  # 字符串名 or MultiTurnScheduler 实例
        tokenizer = getattr(self, 'processing_class', None)
        if isinstance(args.multi_turn_scheduler, str):
            assert args.multi_turn_scheduler in multi_turns  # 全局 registry
            scheduler_kwargs = {'max_turns': args.max_turns, 'tokenizer': tokenizer}
            gym_env = getattr(args, 'gym_env', None)
            if gym_env is not None:
                scheduler_kwargs['gym_env'] = gym_env
            multi_turn_scheduler = multi_turns[args.multi_turn_scheduler](**scheduler_kwargs)
            self.multi_turn_scheduler = multi_turn_scheduler
        else:
            self.multi_turn_scheduler = args.multi_turn_scheduler
```

**自定义 scheduler 注入路径**：
- 在 plugin 里：`multi_turns['my_scheduler'] = MyScheduler`（往 registry 注册）
- 命令行：`--multi_turn_scheduler my_scheduler --max_turns 10`

### `_prepare_rewards()` 解析

接受 `reward_funcs: List[str|Callable]` 和 `reward_model: nn.Module`：

```python
def _prepare_rewards(self, reward_funcs, reward_model, reward_templates):
    # 字符串名查 orms registry: orms['accuracy'] = MathAccuracy, etc.
    # Callable 直接用
    # reward_model nn.Module + reward_template
    # async reward func 单独 await 池
```

**自定义 reward 注入路径**：
- 在 plugin 里：`orms['my_reward'] = MyReward`
- 命令行：`--reward_funcs my_reward`

---

## 训练循环（HF Trainer 标准流 + GRPO 特化）

继承 HF `Trainer.train()` 主循环：

```python
for epoch in range(num_epochs):
    for step, inputs in enumerate(dataloader):
        # ★ HF base: prepare_inputs
        inputs = self._prepare_inputs(inputs)

        # ★ HF base: training_step
        loss = self.training_step(model, inputs)
        loss.backward()
        optimizer.step()
        scheduler.step()
```

GRPOTrainer 重写 `_prepare_inputs` 和 `training_step`（通过 `compute_loss`）。

### `_prepare_inputs(generation_batch)` — line 187

```python
def _prepare_inputs(self, generation_batch):
    # GRPO 关键：一次 rollout 给多个 gradient update steps 用（steps_per_generation）
    # 不是每 step rollout，而是每 N step rollout 一次，前 N-1 step 用 buffered
    if self._step % (self.num_iterations * self.steps_per_generation) == 0:
        # 时候到了：生成新的 rollout batch
        generation_batch = self._generate_and_score_completions(generation_batch)
        self._buffered_inputs = generation_batch
    inputs = self._buffered_inputs[self._step % spg]
    self._step += 1
    return inputs
```

→ **效率**：1 次 rollout 复用 N 次 backward，节省 inference。

### `training_step(model, inputs)` — line 1940

继承 HF `Trainer.training_step`，调 `compute_loss()`，然后 `loss.backward()`。

---

## `_generate_and_score_completions`：rollout → reward → advantage

**`grpo_trainer.py:234`**。这是 GRPO 的**核心 pipeline**，5 个 substeps：

```python
def _generate_and_score_completions(self, inputs):
    # Step 1: Rollout（multi-turn 在这里发生）
    inputs = self._generate_completions(inputs)
       # 每个 input dict 现在含 'messages' (完整多轮对话)
       #                     'response_token_ids', 'response_loss_mask',
       #                     'rollout_infos' (含 num_turns 等)

    # Step 2: Reward 计算
    total_rewards_per_func = self._score_completions(inputs)
       # 调每个 reward_func(completions, **reward_kwargs)
       # 含 async reward func（asyncio.gather）

    # Step 3: Dynamic resampling（如果 std=0 group 需要 resample）
    if self.dynamic_sample:
        inputs, total_rewards_per_func = self._dynamic_sampling(...)

    # Step 4: Tokenize + completion_mask
    batch_encoded_inputs = self._prepare_batch_inputs(inputs)

    # Step 5: Advantage 计算（GRPO group-relative）
    total_advantages = self._compute_advantages(inputs, total_rewards_per_func, batch_encoded_inputs)
       # 默认 grouped: 同 prompt 的 num_generations 个 completion 共享 baseline
       # advantage = (reward - group_mean) / group_std

    # 把 advantages 塞回 batch_encoded
    for batch, batch_encoded in zip(...):
        batch_encoded['advantages'] = ...

    return batch_encoded_inputs  # List[DataType]，每 chunk 一个 mini-batch
```

### `_generate_completions`（line 214）

```python
def _generate_completions(self, inputs):
    inputs = self._preprocess_inputs(inputs)  # 加 prompt_ids
    if self.use_fast_infer:  # vLLM 路径
        results = self._fast_infer(inputs)
    else:
        # 退回 transformers generate（慢）
        results = self._infer_single_or_multi_turn(inputs, self.request_config)
    return results
```

`_fast_infer` 内部根据 `vllm_mode`：
- **colocate**：`_colocate_rollout` → 本地 vLLM engine + multi-turn loop
- **server**：`_server_rollout` → HTTP 到 `swift rollout` 独立进程

### `_compute_rewards_per_func`（line 343）

```python
for i, (reward_func, reward_model_plugin, reward_func_name) in enumerate(...):
    if isinstance(reward_func, nn.Module):
        output = reward_model_plugin(inputs=reward_inputs, **reward_kwargs)
    elif i in self._async_reward_func_indices:
        # 后面用 asyncio.gather 跑
        pass
    else:
        output = reward_func(completions, **reward_kwargs)
        # reward_kwargs 含 trajectory_inputs（multi-turn）, trainer_state, 以及 dataset 列
```

→ reward_func 能看到完整 trajectory + final answer，可自由设计评分逻辑。

### `_compute_advantages`（line 412）

```python
def _compute_advantages(self, inputs, rewards_per_func, batch_encoded_inputs):
    # 默认 grouped mode:
    #   每 prompt 有 num_generations 个 rollout
    #   advantage[i] = (rewards[i] - group_mean) / group_std
    # 或 prompt_id / request_id 模式（multi-turn 动态分组）
```

**这就是 GRPO 的 "G"（Group）**：用同 prompt 多 rollout 算 baseline，省掉 critic。

---

## MultiTurnScheduler 详解

**`swift/rollout/multi_turn.py`**（832 行）。两层 class：

### 基类 `RolloutScheduler`（single-turn）

```python
class RolloutScheduler(ABC):
    def __init__(self, infer_engine=None, max_turns=None, **kwargs):
        self.infer_engine = infer_engine
        self._tokenizer = kwargs.get('tokenizer', None)
        self.max_turns = max_turns

    # Universal async hooks（colocate 和 server 模式都调）
    async def on_trajectory_start(self, requests): pass
    async def on_turn_end(self, infer_request, response_choice, current_turn) -> Dict: return {}

    # 主入口
    async def async_infer(self, infer_requests, request_config, ...) -> List[RolloutOutput]:
        # 并发 batch 推理：每个 request 调 self.run()，asyncio.gather 收集
        ...

    async def run(self, infer_request, request_config, **kwargs) -> RolloutOutput:
        # Default: 单次 inference
        response = await self.infer_engine.infer_async(...)
        return RolloutOutput(response, messages, response_token_ids=[...], ...)
```

### `MultiTurnScheduler` 抽象基类

```python
class MultiTurnScheduler(RolloutScheduler, ABC):
    async def run(self, infer_request, request_config, **kwargs):
        """默认 multi-turn loop（也可全 override）"""
        current_request = infer_request
        await self.on_trajectory_start([current_request])
        current_turn = 1
        total_response_ids, total_response_loss_mask, total_rollout_logprobs = [], [], []

        while True:
            # 1. 推理一轮
            response = await self.infer_engine.infer_async(current_request, request_config)
            response_choice = response.choices[0]

            # 2. 追加 assistant message 到对话历史
            messages = current_request.messages
            messages.append({'role': 'assistant', 'content': response_choice.message.content})

            # 3. 调用 on_turn_end hook（env step / metadata）
            turn_result = await self.on_turn_end(current_request, response_choice, current_turn)

            # 4. 检查终止
            should_stop = self.check_finished(current_request, response_choice, current_turn)
            if turn_result.get('done'):
                should_stop = turn_result['done']
            if self.max_turns:
                should_stop = should_stop or (current_turn >= self.max_turns)

            if should_stop:
                # 收集最后一轮 token_ids + logprobs
                ...
                return RolloutOutput(
                    response=response,
                    messages=messages,
                    response_token_ids=total_response_ids,    # 每轮一个 list
                    response_loss_mask=total_response_loss_mask,  # 每轮一个 list, per-token mask
                    rollout_infos={**rollout_infos, 'num_turns': current_turn},
                    rollout_logprobs=total_rollout_logprobs,
                )

            # 5. step() 准备下一轮
            ret = self.step(current_request, response_choice, current_turn)
            current_request = ret['infer_request']

            # 6. 累加 token_ids + loss_mask（如 step 返回）
            if 'response_token_ids' in ret: total_response_ids.append(...)
            if 'response_loss_mask' in ret: total_response_loss_mask.append(...)
            if 'rollout_logprobs' in ret: total_rollout_logprobs.append(...)

            current_turn += 1

    def step(self, infer_request, response_choice, current_turn) -> Dict:
        """子类必须实现：
           - 解析上轮模型输出
           - 准备下轮 prompt
           - 返回 dict 含 infer_request / response_token_ids / response_loss_mask
        """
        raise NotImplementedError

    def check_finished(self, infer_request, response_choice, current_turn) -> bool:
        """默认：finish_reason='length' 或 current_turn >= max_turns 即终止
           可 override 成基于内容的自定义终止条件"""
        ...
```

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

**关键**：`response_loss_mask` 让你**选择性 mask 哪些 token 不参与 loss**。常见用途如:在 tool-use / agent 多轮场景里,把"工具返回结果"等非模型生成的 token mask 掉,只对模型真正生成的 token 算 loss,避免把环境注入的内容也当成被优化的对象。

### 已有示例：`ThinkingModelTipsScheduler`（line 471）

ms-swift 自带的 multi-turn reasoning scheduler 例子，可直接参考。

---

## `_compute_loss_and_metrics`：GRPO loss 公式

**`grpo_trainer.py:1090`** 是自定义 loss 修改的核心入口。逐段拆：

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
old_per_token_logps = (per_token_logps.detach() if on-policy else inputs['old_per_token_logps'])
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

### Loss 公式（loss_type 选 'grpo'）

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

## Plugin System：怎么注入 custom 组件

**`examples/train/grpo/plugin/plugin.py`** 是官方示例。**机制**：

### 注册点（4 种）

```python
# 在你自己的 plugin.py 顶部 import 这些 registry
from swift.rewards import orms                    # 1. Reward 函数
from swift.rollout.multi_turn import multi_turns   # 2. Multi-turn scheduler
from swift.rewards.rm_plugin import rm_plugins      # 3. Reward model plugin
from swift.rollout.gym_env import envs              # 4. Gym env
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

### Custom Trainer 怎么办（不在 plugin registry 里）

`trainer_factory.py` 的 TRAINER_MAPPING 是硬编码的字典。**不能从 plugin 注入新 trainer**。两条路径：

**路径 A：subclass + monkey-patch**。在 plugin.py 顶部：

```python
from swift.trainers.trainer_factory import TrainerFactory
from my_module import MyCustomTrainer
TrainerFactory.TRAINER_MAPPING['grpo'] = 'my_module.MyCustomTrainer'
# 警告：这会覆盖 GRPOTrainer
```

**路径 B：直接 instantiate 跳过 SwiftRLHF**。写自己的 `train_custom.py`，直接：

```python
trainer = MyCustomTrainer(model, ref_model, reward_funcs=[MyReward()], args=cfg)
trainer.train()
```

路径 B 更干净，但需要 reimplement 一些 SwiftRLHF 的 data/model 准备逻辑。

---

## vLLM Rollout 数据流（colocate vs server）

### Colocate 模式

```
[GPU 0..7] 每张 GPU 同时运行 training + vLLM engine
   training step → 暂停 → vLLM rollout（用同 GPU）→ 收 result → 继续 training
   优点：省 GPU
   缺点：rollout 和 training 不能并行；multi-turn 受限（不支持 dynamic rollout 数）
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

**`vllm_engine.py:42, 511-519`**：vLLM v0.12+ 用 `structured_outputs`，老版本用 `guided_decoding`。给定一个 Pydantic schema：

```python
from pydantic import BaseModel
from vllm.sampling_params import GuidedDecodingParams

class MyOutputSchema(BaseModel):
    answer: str
    done: bool

request_config.structured_outputs = GuidedDecodingParams(json=MyOutputSchema.model_json_schema())
```

vLLM 内部把 schema 编译成 grammar，logit mask 强制输出 valid JSON。需要模型严格按结构输出时直接用这个即可。

---

## 启动命令完整示例

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

ms-swift GRPO 管线 = (1) `SwiftRLHF` pipeline 加载 model/data → (2) `TrainerFactory` 选 `GRPOTrainer` → (3) `GRPOTrainer.__init__` 准备 rollout/reward/metrics → (4) HF Trainer.train 循环 → (5) `_prepare_inputs` 触发 `_generate_and_score_completions`（rollout + reward + advantage）→ (6) `compute_loss` → `_compute_loss_and_metrics` 算 GRPO PPO-clip loss → (7) backward + optimizer.step。

整个框架的关键扩展点也由此清晰：`MultiTurnScheduler`（控制多轮 rollout）+ `_compute_loss_and_metrics`（自定义 loss）+ reward 函数（plugin 注册），三者组合即可覆盖绝大多数 GRPO 定制需求，而不必改动 Pipeline / Factory / CLI 等上层装配代码。
