---
# the default layout is 'page'
layout: post
title: About Me
icon: fa-solid fa-hat-wizard
order: 2
toc: true
---

<div style="display: flex; align-items: center; margin-bottom: 20px;">
  <div style="flex-shrink: 0; width: 130px; height: 130px; background-image: url('/assets/img/photo2_transparent.png'); background-size: cover; background-position: center; margin-right: 30px;" onmouseover="this.style.backgroundImage='url(/assets/img/photo2_cooler_transparent.png)'" onmouseout="this.style.backgroundImage='url(/assets/img/photo2_transparent.png)'">
  </div>
  <div style="flex-grow: 1; display: flex; justify-content: center; align-items: center;">
    <!-- <div style="font-style: italic; color: #555; border-left: 5px solid #eee; padding-left: 20px;"> -->
    <div style="color: #495485; font-size: 22px">
      <!-- <b>The world only makes sense if you force it to. <i>—&nbsp;Batman.</i></b> -->
      <b>"The powerful play goes on...<br>And you may contribute a verse."</b>
    </div>
  </div>
</div>

<!-- > *Answer.*  
> That you are here — that life exists and identity,  
> That the powerful play goes on, and you may contribute a verse.  
> *— Walt Whitman, O Me! O Life!*
{: .prompt-info } -->

{% include intro_shared.html %}

---

## Research Interests

<style>
  .flex-container {
    display: flex;
    justify-content: flex-start;
    align-items: stretch;
  }
  .flex-item {
    /* width: 45%; */
    width: 300px;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .spacer {
    width: 20%;
  }
  .auto-spacer {
    flex-grow: 1;
  }
  @media (max-width: 768px) {
    .flex-container {
      flex-direction: column;
      align-items: center;
    }
    .flex-item, .spacer, .auto-spacer {
      width: 85%;
    }
  }
</style>

<div class="flex-container">
  <div class="flex-item">
    <div>
      <p><strong>Currently</strong></p>
      <ul>
        <!-- <li>Social Dilemma</li> -->
        <!-- <li>Multi-Agent RL: Mixed-Motive Tasks</li> -->
        <li>Multi-Agent Reinforcement Learning</li>
        <li>Game Theory: Information Design</li>
        <!-- <li>Algorithmic Game Theory</li> -->
        <!-- <ul style="margin-left: -20px;">
          <li>Information Design</li>
        </ul> -->
        <li>LLMs for Game Solvers</li>
        <li>Communication Mechanisms</li>
      </ul>
      <p><strong>Formerly</strong></p>
      <ul>
        <li>Redundant Manipulator Control</li>
        <li>Robotic Mechanism Design</li>
      </ul>
    </div>
  </div>
  <div class="spacer"></div>
  <div class="flex-item" style="text-align: center;">
    <img src="/assets/img/about/crow1.png" alt="pic1" style="max-height: 100%; margin: auto;" />
    <p><a href="https://falseknees.com/about.html">"False Knees" by Joshua</a></p>
  </div>
  <div class="auto-spacer"></div>
</div>

<!-- <br clear="all"/> -->

---

## Education & Experience

### Education 

<!-- - <span translate="no">**Shenzhen Loop Area Institute**</span>   -->
  <!-- Joint Ph.D. Student (2025.9 - Present)   -->
- <span translate="no">**The Chinese University of Hong Kong, Shenzhen**</span>  
  Ph.D. Student in Data Science (2024.8 - Present)  
  <!-- Ph.D. Student in Data Science (2024.9 - Present)   -->
<!-- - <span translate="no">**Tiangong University**</span>   -->
- <span translate="no">**Tianjin Polytechnic University**</span>  
  Bachelor of Engineering in Computer Science and Technology (2018.9 - 2022.6)
    - School of Computer Science and Technology (2019.9 - 2022.6)  
        > GPA:  3.89 / 4 (92.22 / 100); Rank: 1 / 127  
        [[Certification]]({{site.baseurl}}/posts/Certification-Rank/)
    - School of Mechanical Engineering (2018.9 - 2019.6)  
        > GPA:  3.90 / 4 (92.00 / 100); Rank: 1 / 60

### Experience

- <span translate="no">**Tencent**</span>  
  Research Intern @ Lightspeed Studios, Shenzhen (2025.12 - 2026.6)
- <span translate="no">**The Chinese University of Hong Kong, Shenzhen**</span>  
  Research Assistant @ School of Data Science (2022.2 - 2024.8)  
  Advisors: Prof. Baoxiang Wang & Prof. Hongyuan Zha (Subsequently my Ph.D. Supervisor and Co-Supervisor)

---

## Publications & Manuscripts

<!-- [[Google Scholar]](https://scholar.google.com/citations?user=fbvQHX4AAAAJ&hl=zh-CN) -->

<div class="pub-view-toggle">
  <button class="pub-view-btn pub-view-btn-active" onclick="switchPubView('category')" id="btn-pub-category">By Category</button>
  <button class="pub-view-btn" onclick="switchPubView('time')" id="btn-pub-time">By Time</button>
</div>

<style>
  .pub-view-toggle { margin: 10px 0 20px 0; }
  .pub-view-btn {
    padding: 6px 16px;
    margin-right: 8px;
    border: 1px solid #c8d3e3;
    background: #f0f4fa;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    font-weight: bold;
    color: #495485;
  }
  .pub-view-btn:hover { background: #e3ebf5; }
  .pub-view-btn-active,
  .pub-view-btn-active:hover {
    background: #495485;
    color: white;
    border-color: #495485;
  }
</style>

<div id="pub-by-time" style="display: none;">
<!-- Auto-populated from #pub-by-category by JS at page load. Single source of truth lives in by-category below. -->
</div>

<div id="pub-by-category" markdown="1">

### Data Science x Game Theory

<!-- #### Mix-Motive MARL Communication x Game Theory -->

- <span translate="no">Information Design in Multi-Agent Reinforcement Learning.</span>  
    <span translate="no">**Yue Lin**, Wenhao Li, Hongyuan Zha, Baoxiang Wang.</span>  
    <span translate="no">*Neural Information Processing Systems (NeurIPS) 2023*.</span>
    > Poster. This is currently my most representative work.  
    [[Paper]](https://arxiv.org/abs/2305.06807) 
    [[Code]](https://github.com/YueLin301/InformationDesignMARL) 
    [[Experiments]](https://wandb.ai/yuelin301/IND+MARL?nw=nwuseryuelin301) 
    [[Blog en]]({{site.baseurl}}/posts/IDMARL/) 
    [[Blog cn]]({{site.baseurl}}/posts/IDMARL-cn/) 
    [[Zhihu cn]](https://zhuanlan.zhihu.com/p/687305626) 
    [[Slides]]({{site.baseurl}}/posts/IDMARL/#slides) 
    [[Talk en]](https://www.youtube.com/watch?v=yhVlpv_1Pg4) 
    [[Talk RLChina]](https://www.bilibili.com/video/BV1t142117Km?vd_source=b3cf9eb7cfe43c730613c5158a38e978) 
    [[Patent]]({{site.baseurl}}/posts/IDMARL_patent/)
    <!-- [[Talk cn]](https://www.bilibili.com/video/BV1e94y177Dj/?share_source=copy_web&vd_source=b3cf9eb7cfe43c730613c5158a38e978&t=2825)   -->
    <details style="margin-top: -10px;">
    <summary>[Click to check the Abstract]</summary>
    <div class="language-plaintext highlighter-rouge">
    <div class="code-header">
    <span data-label-text="Abstract"><i class="fas fa-code fa-fw small"></i></span>
    <span></span>
    </div>
    <pre class="highlight" style="margin-left: 20px; white-space: pre-wrap; word-break: break-word;"><code>Reinforcement learning (RL) is inspired by the way human infants and animals learn from the environment. The setting is somewhat idealized because, in actual tasks, other agents in the environment have their own goals and behave adaptively to the ego agent. To thrive in those environments, the agent needs to influence other agents so their actions become more helpful and less harmful. Research in computational economics distills two ways to influence others directly: by providing tangible goods (mechanism design) and by providing information (information design). This work investigates information design problems for a group of RL agents. The main challenges are two-fold. One is the information provided will immediately affect the transition of the agent trajectories, which introduces additional non-stationarity. The other is the information can be ignored, so the sender must provide information that the receiver is willing to respect. We formulate the Markov signaling game, and develop the notions of signaling gradient and the extended obedience constraints that address these challenges. Our algorithm is efficient on various mixed-motive tasks and provides further insights into computational economics. Our code is publicly available at https://github.com/YueLin301/InformationDesignMARL.</code></pre></div>
    </details>  
    <details>
    <summary>[Click to check the BibTex code]</summary>
    <div class="language-plaintext highlighter-rouge">
    <div class="code-header">
    <span data-label-text="BibTex"><i class="fas fa-code fa-fw small"></i></span>
    <span></span>
    </div>
    <pre class="highlight" style="margin-left: 20px;"><code>@article{lin2023information,
    title={Information design in multi-agent reinforcement learning},
    author={Lin, Yue and Li, Wenhao and Zha, Hongyuan and Wang, Baoxiang},
    journal={Advances in Neural Information Processing Systems},
    volume={36},
    pages={25584--25597},
    year={2023}
}</code></pre></div>
    </details>
- <span translate="no">Verbalized Bayesian Persuasion.</span>  
    <span translate="no">Wenhao Li, **Yue Lin**, Xiangfeng Wang, Bo Jin, Hongyuan Zha, Baoxiang Wang.</span>  
    <span translate="no">*International Conference on Machine Learning (ICML) 2026.*</span>
    > Poster.  
    [[Paper]](https://arxiv.org/abs/2502.01587) 
    <details style="margin-top: -10px;">
    <summary>[Click to check the Abstract]</summary>
    <div class="language-plaintext highlighter-rouge">
    <div class="code-header">
    <span data-label-text="Abstract"><i class="fas fa-code fa-fw small"></i></span>
    <span></span>
    </div>
    <pre class="highlight" style="margin-left: 20px; white-space: pre-wrap; word-break: break-word;"><code>Information design (ID) explores how a sender influence the optimal behavior of receivers to achieve specific objectives. While ID originates from everyday human communication, existing game-theoretic and machine learning methods often model information structures as numbers, which limits many applications to toy games. This work leverages LLMs and proposes a verbalized framework in Bayesian persuasion (BP), which extends classic BP to real-world games involving human dialogues for the first time. Specifically, we map the BP to a verbalized mediator-augmented extensive-form game, where LLMs instantiate the sender and receiver. To efficiently solve the verbalized game, we propose a generalized equilibrium-finding algorithm combining LLM and game solver. The algorithm is reinforced with techniques including verbalized commitment assumptions, verbalized obedience constraints, and information obfuscation. Numerical experiments in dialogue scenarios, such as recommendation letters, courtroom interactions, and law enforcement, validate that our framework can both reproduce theoretical results in classic BP and discover effective persuasion strategies in more complex natural language and multi-stage scenarios.</code></pre></div>
    </details>  
    <details>
    <summary>[Click to check the BibTex code]</summary>
    <div class="language-plaintext highlighter-rouge">
    <div class="code-header">
    <span data-label-text="BibTex"><i class="fas fa-code fa-fw small"></i></span>
    <span></span>
    </div>
    <pre class="highlight" style="margin-left: 20px;"><code>@article{li2025verbalized,
  title={Verbalized Bayesian Persuasion},
  author={Li, Wenhao and Lin, Yue and Wang, Xiangfeng and Jin, Bo and Zha, Hongyuan and Wang, Baoxiang},
  journal={arXiv preprint arXiv:2502.01587},
  year={2025}
}</code></pre></div>
    </details>
- <span translate="no">Policy-Conditioned Policies for Multi-Agent Task Solving.</span>  
    <span translate="no">**Yue Lin**, Shuhui Zhu, Wenhao Li, Dan Qiao, Ang Li, Pascal Poupart, Hongyuan Zha, Baoxiang Wang.</span>  
    <span translate="no">*arXiv preprint. 2025-12-24.*</span>
    > 
    [[Manuscript]](https://arxiv.org/abs/2512.21024) 
    <details style="margin-top: -10px;">
    <summary>[Click to check the Abstract]</summary>
    <div class="language-plaintext highlighter-rouge">
    <div class="code-header">
    <span data-label-text="Abstract"><i class="fas fa-code fa-fw small"></i></span>
    <span></span>
    </div>
    <pre class="highlight" style="margin-left: 20px; white-space: pre-wrap; word-break: break-word;"><code>In multi-agent tasks, the central challenge lies in the dynamic adaptation of strategies. However, directly conditioning on opponents' strategies is intractable in the prevalent deep reinforcement learning paradigm due to a fundamental ``representational bottleneck'': neural policies are opaque, high-dimensional parameter vectors that are incomprehensible to other agents. In this work, we propose a paradigm shift that bridges this gap by representing policies as human-interpretable source code and utilizing Large Language Models (LLMs) as approximate interpreters. This programmatic representation allows us to operationalize the game-theoretic concept of Program Equilibrium. We reformulate the learning problem by utilizing LLMs to perform optimization directly in the space of programmatic policies. The LLM functions as a point-wise best-response operator that iteratively synthesizes and refines the ego agent's policy code to respond to the opponent's strategy. We formalize this process as Programmatic Iterated Best Response (PIBR), an algorithm where the policy code is optimized by textual gradients, using structured feedback derived from game utility and runtime unit tests. We demonstrate that this approach effectively solves several standard coordination matrix games and a cooperative Level-Based Foraging environment.</code></pre></div>
    </details>  
    <details>
    <summary>[Click to check the BibTex code]</summary>
    <div class="language-plaintext highlighter-rouge">
    <div class="code-header">
    <span data-label-text="BibTex"><i class="fas fa-code fa-fw small"></i></span>
    <span></span>
    </div>
    <pre class="highlight" style="margin-left: 20px;"><code>@article{lin2025policy,
  title={Policy-Conditioned Policies for Multi-Agent Task Solving},
  author={Lin, Yue and Zhu, Shuhui and Li, Wenhao and Li, Ang and Qiao, Dan and Poupart, Pascal and Zha, Hongyuan and Wang, Baoxiang},
  journal={arXiv preprint arXiv:2512.21024},
  year={2025}
}</code></pre></div>
    </details>


### Data Science x Game Theory x Social Science


- <span translate="no">The Reciprocity Gradient.</span>  
    <span translate="no">**Yue Lin**, Pascal Poupart, Shuhui Zhu, Dan Qiao, Wenhao Li, Yuan Liu, Hongyuan Zha, Baoxiang Wang.</span>  
    <span translate="no">*arXiv preprint. 2026-05-08.*</span>
    > 
    [[Manuscript]](https://arxiv.org/abs/2605.08323) 
    <details style="margin-top: -10px;">
    <summary>[Click to check the Abstract]</summary>
    <div class="language-plaintext highlighter-rouge">
    <div class="code-header">
    <span data-label-text="Abstract"><i class="fas fa-code fa-fw small"></i></span>
    <span></span>
    </div>
    <pre class="highlight" style="margin-left: 20px; white-space: pre-wrap; word-break: break-word;"><code>Communication is fundamental to sustaining reciprocity and cooperation in strategic interactions. We identify and formulate the influence attribution problem as the central optimization difficulty inherent in such dynamics for a learning agent: any action or signal the agent emits reshapes the reputations of many third parties along combinatorially branching paths before feeding back into its own future rewards, forcing the agent to account for all of these indirect channels at once when choosing every action. To address this, we introduce the reciprocity gradient, which explicitly backpropagates reward gradients through private estimators of opponents' policies trained from public observations. The gradient flows through the reputation chain itself analytically, rather than being estimated from sampled returns. It jointly optimizes actions and evaluative signals without intrinsic rewards or reward shaping. Empirically, the method recovers near-optimal context-sensitive policies, while sample-based baselines collapse into constant-output policies.</code></pre></div>
    </details>
- <span translate="no">Talk, Judge, Cooperate: Gossip-Driven Indirect Reciprocity in Self-Interested LLM Agents.</span>  
    <span translate="no">Shuhui Zhu, **Yue Lin**, Shriya Kaistha, Wenhao Li, Baoxiang Wang, Hongyuan Zha, Gillian K Hadfield, Pascal Poupart.</span>  
    <span translate="no">*International Conference on Machine Learning (ICML) 2026.*</span>
    > Poster.  
    [[Paper]](https://arxiv.org/abs/2602.07777) 
    [[Code]](https://github.com/shuhui-zhu/ALIGN)
    <details style="margin-top: -10px;">
    <summary>[Click to check the Abstract]</summary>
    <div class="language-plaintext highlighter-rouge">
    <div class="code-header">
    <span data-label-text="Abstract"><i class="fas fa-code fa-fw small"></i></span>
    <span></span>
    </div>
    <pre class="highlight" style="margin-left: 20px; white-space: pre-wrap; word-break: break-word;"><code>Indirect reciprocity, which means helping those who help others, is difficult to sustain among decentralized, self-interested LLM agents without reliable reputation systems. We introduce Agentic Linguistic Gossip Network (ALIGN), an automated framework where agents strategically share open-ended gossip using hierarchical tones to evaluate trustworthiness and coordinate social norms. We demonstrate that ALIGN consistently improves indirect reciprocity and resists malicious entrants by identifying and ostracizing defectors without changing intrinsic incentives. Notably, we find that stronger reasoning capabilities in LLMs lead to more incentive-aligned cooperation, whereas chat models often over-cooperate even when strategically suboptimal. These results suggest that leveraging LLM reasoning through decentralized gossip is a promising path for maintaining social welfare in agentic ecosystems. Our code is available at https://github.com/shuhui-zhu/ALIGN.</code></pre></div>
    </details>  


### Game Theory

- <span translate="no">Information Bargaining: Bilateral Commitment in Bayesian Persuasion.</span>  
    <span translate="no">**Yue Lin**, Shuhui Zhu, William A Cunningham, Wenhao Li, Pascal Poupart, Hongyuan Zha, Baoxiang Wang.</span>  
    <span translate="no">*EC 2025 Workshop: Information Economics and Large Language Models.*</span>  
    > The title of an alternative version: Bayesian Persuasion as a Bargaining Game.  
    [[Paper]](https://arxiv.org/abs/2506.05876) 
    [[Code & Experiments]](https://github.com/YueLin301/InformationBargaining) 
    <details style="margin-top: -10px;">
    <summary>[Click to check the Abstract]</summary>
    <div class="language-plaintext highlighter-rouge">
    <div class="code-header">
    <span data-label-text="Abstract"><i class="fas fa-code fa-fw small"></i></span>
    <span></span>
    </div>
    <pre class="highlight" style="margin-left: 20px; white-space: pre-wrap; word-break: break-word;"><code>Bayesian persuasion, an extension of cheap-talk communication, involves an informed sender committing to a signaling scheme to influence a receiver’s actions. Compared to cheap talk, this sender’s commitment enables the receiver to verify the incentive compatibility of signals beforehand, facilitating cooperation. While effective in one-shot scenarios, Bayesian persuasion faces computational complexity (NP-hardness) when extended to long-term interactions, where the receiver may adopt dynamic strategies conditional on past outcomes and future expectations. To address this complexity, we introduce the bargaining perspective, which allows: (1) a unified framework and well-structured solution concept for long-term persuasion, with desirable properties such as fairness and Pareto efficiency; (2) a clear distinction between two previously conflated advantages: the sender’s informational advantage and first-proposer advantage. With only modest modifications to the standard setting, this perspective makes explicit the common knowledge of the game structure and grants the receiver comparable commitment capabilities, thereby reinterpreting classic one-sided persuasion as a balanced information bargaining framework. The framework is validated through a two-stage validationand-inference paradigm: We first demonstrate that GPT-o3 and DeepSeek-R1, out of publicly available LLMs, reliably handle standard tasks; We then apply them to persuasion scenarios to test that the outcomes align with what our informationbargaining framework suggests. All code, results, and terminal logs are publicly available at https://github.com/YueLin301/InformationBargaining.</code></pre></div>
    </details>  
    <details>
    <summary>[Click to check the BibTex code]</summary>
    <div class="language-plaintext highlighter-rouge">
    <div class="code-header">
    <span data-label-text="BibTex"><i class="fas fa-code fa-fw small"></i></span>
    <span></span>
    </div>
    <pre class="highlight" style="margin-left: 20px;"><code>@article{lin2025bayesian,
  title={Bayesian Persuasion as a Bargaining Game},
  author={Lin, Yue and Zhu, Shuhui and Cunningham, William A and Li, Wenhao and Poupart, Pascal and Zha, Hongyuan and Wang, Baoxiang},
  journal={arXiv preprint arXiv:2506.05876},
  year={2025}
}</code></pre></div>
    </details>









<!-- > Of the empty and useless years of the rest, with the rest me intertwined,  
> The question, O me! so sad, recurring — What good amid these, O me, O life?  
> *— Walt Whitman, O Me! O Life!*
{: .prompt-tip } -->


### Robotics

- <span translate="no">Innovative Design and Simulation of a Transformable Robot with Flexibility and Versatility, RHex-T3.</span>  
    <span translate="no">**Yue Lin**, Yujia Tian, Yongjiang Xue, Shujun Han, Huaiyu Zhang, Wenxin Lai, Xuan Xiao.</span>  
    <span translate="no">*International Conference on Robotics and Automation (ICRA) 2021*.</span>
    > Oral. Delivered a presentation at the Xi'an conference venue.  
    [[Paper]](https://ieeexplore.ieee.org/abstract/document/9561060) 
    [[Blog]]({{site.baseurl}}/posts/RHex-T3/) 
    [[Demo Videos]]({{site.baseurl}}/posts/RHex-T3/#videos)  
    <details style="margin-top: -10px;">
    <summary>[Click to check the Abstract]</summary>
    <div class="language-plaintext highlighter-rouge">
    <div class="code-header">
    <span data-label-text="Abstract"><i class="fas fa-code fa-fw small"></i></span>
    <span></span>
    </div>
    <pre class="highlight" style="margin-left: 20px; white-space: pre-wrap; word-break: break-word;"><code>This paper presents a transformable RHex-inspired robot, RHex-T3, with high energy efficiency, excellent flexibility and versatility. By using the innovative 2-DoF transformable structure, RHex-T3 inherits most of RHex’s mobility, and can also switch to other 4 modes for handling various missions. The wheel-mode improves the efficiency of RHex-T3, and the leg-mode helps to generate a smooth locomotion when RHex-T3 is overcoming obstacles. In addition, RHex-T3 can switch to the claw-mode for transportation missions, and even climb ladders by using the hook-mode. The simulation model is conducted based on the mechanical structure, and thus the properties in different modes are verified and analyzed through numerical simulations.</code></pre></div>
    </details>  
    <details>
    <summary>[Click to check the BibTex Code]</summary>
    <div class="language-plaintext highlighter-rouge">
    <div class="code-header">
    <span data-label-text="BibTex"><i class="fas fa-code fa-fw small"></i></span>
    <span></span>
    </div>
    <pre class="highlight" style="margin-left: 20px;"><code>@inproceedings{lin2021innovative,
    title={Innovative design and simulation of a transformable robot with flexibility and versatility, RHex-T3},
    author={Lin, Yue and Tian, Yujia and Xue, Yongjiang and Han, Shujun and Zhang, Huaiyu and Lai, Wenxin and Xiao, Xuan},
    booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
    pages={6992--6998},
    year={2021},
    organization={IEEE}
}</code></pre></div>
    </details>


- <span translate="no">A snake-inspired path planning algorithm based on reinforcement learning and self-motion for hyper-redundant manipulators.</span>  
    <span translate="no">**Yue Lin**, Jianming Wang, Xuan Xiao, Ji Qu, Fatao Qin.</span>  
    <span translate="no">*International Journal of Advanced Robotic Systems (IJARS) 2022*.</span>  
    > [[Paper]](https://journals.sagepub.com/doi/full/10.1177/17298806221110022) 
    [[Code]](https://github.com/YueLin301/Swinging-Search-Crawling-Control) 
    [[Blog]]({{site.baseurl}}/posts/SSCC/) 
    [[Demo Video]]({{site.baseurl}}/posts/SSCC/#videos)
    <details style="margin-top: -10px;">
    <summary>[Click to check the Abstract]</summary>
    <div class="language-plaintext highlighter-rouge">
    <div class="code-header">
    <span data-label-text="Abstract"><i class="fas fa-code fa-fw small"></i></span>
    <span></span>
    </div>
    <pre class="highlight" style="margin-left: 20px; white-space: pre-wrap; word-break: break-word;"><code>Redundant manipulators are flexible enough to adapt to complex environments, but their controller is also required to be specific for their extra degrees of freedom. Inspired by the morphology of snakes, we propose a path planning algorithm named Swinging Search and Crawling Control, which allows the snake-like redundant manipulators to explore in complex pipeline environments without collision. The proposed algorithm consists of the Swinging Search and the Crawling Control. In Swinging Search, a collision-free manipulator configuration that of the end-effector in the target point is found by applying reinforcement learning to self-motion, instead of designing joint motion. The self-motion narrows the search space to the null space, and the reinforcement learning makes the algorithm use the information of the environment, instead of blindly searching. Then in Crawling Control, the manipulator is controlled to crawl to the target point like a snake along the collision-free configuration. It only needs to search for a collision-free configuration for the manipulator, instead of searching collision-free configurations throughout the process of path planning. Simulation experiments show that the algorithm can complete path planning tasks of hyper-redundant manipulators in complex environments. The 16 DoFs and 24 DoFs manipulators can achieve 83.3% and 96.7% success rates in the pipe, respectively. In the concentric pipe, the 24 DoFs manipulator has a success rate of 96.1%.</code></pre></div>
    </details>  
    <details>
    <summary>[Click to check the BibTex code]</summary>
    <div class="language-plaintext highlighter-rouge">
    <div class="code-header">
    <span data-label-text="BibTex"><i class="fas fa-code fa-fw small"></i></span>
    <span></span>
    </div>
    <pre class="highlight" style="margin-left: 20px;"><code>@article{lin2022snake,
    title={A snake-inspired path planning algorithm based on reinforcement learning and self-motion for hyper-redundant manipulators},
    author={Lin, Yue and Wang, Jianming and Xiao, Xuan and Qu, Ji and Qin, Fatao},
    journal={International Journal of Advanced Robotic Systems},
    volume={19},
    number={4},
    pages={17298806221110022},
    year={2022},
    publisher={SAGE Publications Sage UK: London, England}
}</code></pre></div>
    </details>

<!-- - <span translate="no">Self-Adaptive Walking Speed Control on Underactuated Rimless Wheel.</span>  
    <span translate="no">Wenxin Lai, Yujia Tian, Shujun Han, **Yue Lin**, Yongiiang Xue, Juezhu Lai.</span>  
    <span translate="no">*IEEE International Conference on Mechatronics and Automation (ICMA) 2020*.</span>  
    > [[Paper]](https://ieeexplore.ieee.org/abstract/document/9233853)   -->

<!-- - Innovative Design and Simulation of a Transformable Robot with Flexibility and Versatility, RHex-T3.  
    **Yue Lin**, Yujia Tian, Yongjiang Xue, Shujun Han, Huaiyu Zhang, Wenxin Lai, Xuan Xiao.  
    *International Conference on Robotics and Automation (ICRA) 2021*.
    > Oral. Delivered a presentation at the Xi'an conference venue.  
    [[Paper]](https://ieeexplore.ieee.org/abstract/document/9561060) 
    [[Blog]]({{site.baseurl}}/posts/RHex-T3/)  
- A snake-inspired path planning algorithm based on reinforcement learning and self-motion for hyper-redundant manipulators.  
    **Yue Lin**, Jianming Wang, Xuan Xiao, Ji Qu, Fatao Qin.  
    *International Journal of Advanced Robotic Systems (IJARS) 2022*.  
    > [[Paper]](https://journals.sagepub.com/doi/full/10.1177/17298806221110022) [[Blog]]({{site.baseurl}}/posts/SSCC/) 
    [[Code]](https://github.com/YueLin301/Swinging-Search-Crawling-Control)  
- Self-Adaptive Walking Speed Control on Underactuated Rimless Wheel.  
    Wenxin Lai, Yujia Tian, Shujun Han, **Yue Lin**, Yongiiang Xue, Juezhu Lai.  
    *IEEE International Conference on Mechatronics and Automation (ICMA) 2020*.  
    > [[Paper]](https://ieeexplore.ieee.org/abstract/document/9233853)   -->

<!-- ---

## Honors & Awards
- First Prize in the 16th Tianjin "The Challenge Cup" Competition - 2021.6
- First Prize of the President's Scholarship (Top: 3%), Tiangong University - 2020.12
- Second Prize of the President's Scholarship (Top: 10%), Tiangong University - 2018.12 & 2019.12
- Third Prize in the 15th Tianjin "The Projection Mapping Contest" Competition - 2019.5

---

## Campus Involvement
- Assisted classmates in preparing for final exams during the pandemic. Check out my [mind map notes](https://github.com/YueLin301/MindMap-bakcup).
- Initiated and organized a weekly seminar on Advanced Mathematics for classmates, promoting a harmonious learning environment.

The high-scoring courses at the undergraduate level include the following: 
  - **Mathematics:** Advanced Mathematics (100), Discrete Mathematics (96);
  - **Computer Science:** Compiler Theory (99), Embedded System Design (99), Software Engineering (98), Operating System (95), Computer Networks (96), Curriculum Design on Database (95), Application Development Practice (97), Internet of Things Technology (99), Software Development Practice (95), Distributed and Cloud Computing (95);
  - **Artificial Intelligence:** Digital Image Processing (98), Speech Recognition and Natural Language Understanding (96), Computerized Vision (96), Machine Learning (95);
  - **Robotics:** Intelligent Mobile Robot (99), Electrical and Electronic Technology (98), College Physics (97). -->


<!-- ### Preprints -->

</div>

<script>
function buildPubByTime() {
  var catContainer = document.getElementById('pub-by-category');
  var timeContainer = document.getElementById('pub-by-time');
  if (!catContainer || !timeContainer) return;

  var lis = catContainer.querySelectorAll('ul > li');
  var entries = [];

  lis.forEach(function (li) {
    var clone = li.cloneNode(true);
    clone.querySelectorAll('details').forEach(function (d) { d.remove(); });
    var text = clone.textContent;

    var matches = text.match(/(20\d{2})(?:-(\d{2})(?:-(\d{2}))?)?/g);
    var sortKey = 0;
    var year = null;
    if (matches && matches.length > 0) {
      var last = matches[matches.length - 1];
      var parts = last.match(/(20\d{2})(?:-(\d{2})(?:-(\d{2}))?)?/);
      year = parseInt(parts[1], 10);
      var month = parts[2] ? parseInt(parts[2], 10) : 1;
      var day = parts[3] ? parseInt(parts[3], 10) : 1;
      sortKey = year * 10000 + month * 100 + day;
    }

    entries.push({ li: li, sortKey: sortKey, year: year });
  });

  entries.sort(function (a, b) { return b.sortKey - a.sortKey; });

  timeContainer.innerHTML = '';
  var currentYear = null;
  var currentUl = null;
  entries.forEach(function (e) {
    if (e.year !== currentYear) {
      currentYear = e.year;
      var h3 = document.createElement('h3');
      h3.textContent = currentYear != null ? String(currentYear) : 'Unknown';
      timeContainer.appendChild(h3);
      currentUl = document.createElement('ul');
      timeContainer.appendChild(currentUl);
    }
    currentUl.appendChild(e.li.cloneNode(true));
  });
}

function switchPubView(view) {
  var byCat = document.getElementById('pub-by-category');
  var byTime = document.getElementById('pub-by-time');
  var btnCat = document.getElementById('btn-pub-category');
  var btnTime = document.getElementById('btn-pub-time');
  if (view === 'category') {
    byCat.style.display = 'block';
    byTime.style.display = 'none';
    btnCat.classList.add('pub-view-btn-active');
    btnTime.classList.remove('pub-view-btn-active');
  } else {
    byCat.style.display = 'none';
    byTime.style.display = 'block';
    btnTime.classList.add('pub-view-btn-active');
    btnCat.classList.remove('pub-view-btn-active');
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', buildPubByTime);
} else {
  buildPubByTime();
}
</script>

---

## Professional Services

Service Honors
- NeurIPS 2025 Top Reviewer [[List]](https://neurips.cc/Conferences/2025/ProgramCommittee)
- ICML 2026 Silver Reviewer

<div id="prof-services" markdown="1">

Independent Reviewer
- NeurIPS 2024 [6; 45615], 2025 [5; 32912], 2026 [4; 0]
- ICLR 2025 [3; 21831], 2026 [1; 5994]
- ICML 2025 [6; 32893], 2026 [6; 43185]
- TMLR 2025 [2; 38363], 2026 [2; 10331]

Volunteer
- AAMAS 2024 [3; 9876], 2025 [2; 4792], 2026 [2; 2303]
- ICML (Position) 2025 [2; 5016]

</div>

> Numbers in brackets indicate the number of manuscripts reviewed and the character count of all reviews, respectively. A "0" means the invitation was accepted, but no review assignment has been made yet.  
> Total reviews: <span id="total-reviews">—</span>.

<script>
(function () {
  var section = document.getElementById('prof-services');
  var el = document.getElementById('total-reviews');
  if (!section || !el) return;
  var text = section.textContent;
  var pattern = /\[(\d+)(?:\s*[;,]\s*\d+)?\]/g;
  var match, total = 0;
  while ((match = pattern.exec(text)) !== null) {
    total += parseInt(match[1], 10);
  }
  el.textContent = total;
})();
</script>

---

## Teaching

Teaching Assistant @ The Chinese University of Hong Kong, Shenzhen
-  CSC6021/AIR6001 Artificial Intelligence (2024-25 Term 2).


---

## Patents

- <span translate="no">多智能体强化学习通信方法、终端设备及存储介质</span>  
    <span translate="no">发明人：**林越**、李文浩、查宏远、王趵翔</span>  
    <span translate="no">申请人：香港中文大学（深圳）</span>  
    <span translate="no">类型：发明</span>  
    <span translate="no">状态：已授权</span>  
    > <span translate="no">专利号：ZL 2023 1 0397744.0；授权公告号：CN 116455754 B；授权公告日：2025.9.16</span>  
    [[证书]]({{site.baseurl}}/posts/IDMARL_patent/)
    <!-- <span translate="no">申请公布号：CN116455754A；申请号：2023103977440；申请公布日：2023.07.18；申请日：2023.04.06</span> -->

<!-- http://epub.cnipa.gov.cn/Dxb/IndexQuery -->

---

## Hobbies

<style>
  /* Tame nested-list indentation so it stays reasonable on narrow (mobile) screens */
  .hobbies-list, .hobbies-list ul { padding-left: 1.4em; }
</style>

<ul class="hobbies-list">
<li>
    <details>
    <summary>DC Comics</summary>
    <ul>
    <li>Recommended reads (ordered by how much I love them):
        <ul>
        <li><i>The Riddler: Year One</i></li>
        <li><i>Penguin: Pain &amp; Prejudice</i></li>
        <li><i>Batman: The Killing Joke</i></li>
        <li><i>Batman (2016-): I Am Bane</i></li>
        <li><i>One Bad Day: The Riddler</i></li>
        <li><i>Batman (2016-): The War of Jokes and Riddles</i></li>
        <li><i>Joker's Asylum II: The Riddler</i></li>
        <li><i>Joker: Killer Smile</i></li>
        </ul>
    </li>
    </ul>
    </details>
</li>

<li>
    <details>
    <summary>Video Games</summary>
    <ul>
    <li>王者荣耀 Honor of Kings
        <ul>
        <li>全国第74 诸葛亮，战力13853，胜率57.5%，场次2217</li>
        <li>浙江省第57 嫦娥，战力10725，胜率63.8%，场次298</li>
        <li>浙江省第100 阿古朵，战力10028，胜率60.8%，场次176</li>
        <li>衢州市第5 影，战力10028，胜率54.1%，场次368</li>
        <li>浙江省第42 米莱狄，战力7863，胜率55.2%，场次511</li>
        <li>巅峰赛打野2063分（全国第3892名），中路1849分</li>
        <li>IOS Q区</li>
        </ul>
    </li>
    <li>Overwatch 1 （已退坑）
        <ul>
        <li>Doomfist: 284 hours, 1669 matches, win rate 54%, kill/death 26068/13137</li>
        </ul>
    </li>
    <li>Steam
        <ul>
        <li>Slay the Spire 2, Hollow Knight, Hollow Knight: Silksong, The Stanley Parable, Batman: Arkham Knight, Marvel Rivals...</li>
        <!-- <li>Risk of Rain 2, Hollow Knight, Batman: Arkham Knight, Marvel Rivals, Lost Castle 2, baba is you...</li> -->
        </ul>
    </li>
    <!-- <li>Among all games, the one that has influenced me the most is **The Stanley Parable.**</li> -->
    </ul>
    </details>
</li>

<li>Movies</li>
<!-- <li>Ping-Pong</li> -->
<!-- <li>Psychology</li> -->

</ul>


<!-- ## Personality


<details>
<summary>Big Five Personality Test: [Click to expand]</summary>
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-cly1{text-align:left;vertical-align:middle}
.tg .tg-mwxe{text-align:right;vertical-align:middle}
.tg .tg-yla0{font-weight:bold;text-align:left;vertical-align:middle}
.tg .tg-zt7h{font-weight:bold;text-align:right;vertical-align:middle}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-yla0">Openness To Experience</th>
    <th class="tg-zt7h">107/120</th>
    <th class="tg-yla0">Conscientiousness</th>
    <th class="tg-zt7h">101/120</th>
    <th class="tg-yla0">Extraversion</th>
    <th class="tg-zt7h">95/120</th>
    <th class="tg-yla0">Agreeableness</th>
    <th class="tg-zt7h">82/120</th>
    <th class="tg-yla0">Neuroticism</th>
    <th class="tg-zt7h">64/120</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-cly1">Adventurousness</td>
    <td class="tg-mwxe">17/20</td>
    <td class="tg-cly1">Achievement-Striving</td>
    <td class="tg-mwxe">18/20</td>
    <td class="tg-cly1">Activity Level</td>
    <td class="tg-mwxe">19/20</td>
    <td class="tg-cly1">Altruism</td>
    <td class="tg-mwxe">14/20</td>
    <td class="tg-cly1">Anger</td>
    <td class="tg-mwxe">11/20</td>
  </tr>
  <tr>
    <td class="tg-cly1">Artistic Interests</td>
    <td class="tg-mwxe">19/20</td>
    <td class="tg-cly1">Cautiousness</td>
    <td class="tg-mwxe">16/20</td>
    <td class="tg-cly1">Assertiveness</td>
    <td class="tg-mwxe">15/20</td>
    <td class="tg-cly1">Cooperation</td>
    <td class="tg-mwxe">13/20</td>
    <td class="tg-cly1">Anxiety</td>
    <td class="tg-mwxe">15/20</td>
  </tr>
  <tr>
    <td class="tg-cly1">Emotionality</td>
    <td class="tg-mwxe">18/20</td>
    <td class="tg-cly1">Dutifulness</td>
    <td class="tg-mwxe">14/20</td>
    <td class="tg-cly1">Cheerfulness</td>
    <td class="tg-mwxe">18/20</td>
    <td class="tg-cly1">Modesty</td>
    <td class="tg-mwxe">16/20</td>
    <td class="tg-cly1">Depression</td>
    <td class="tg-mwxe">9/20</td>
  </tr>
  <tr>
    <td class="tg-cly1">Imagination</td>
    <td class="tg-mwxe">20/20</td>
    <td class="tg-cly1">Orderliness</td>
    <td class="tg-mwxe">18/20</td>
    <td class="tg-cly1">Excitement-Seeking</td>
    <td class="tg-mwxe">15/20</td>
    <td class="tg-cly1">Morality</td>
    <td class="tg-mwxe">13/20</td>
    <td class="tg-cly1">Immoderation</td>
    <td class="tg-mwxe">11/20</td>
  </tr>
  <tr>
    <td class="tg-cly1">Intellect</td>
    <td class="tg-mwxe">20/20</td>
    <td class="tg-cly1">Self-Discipline</td>
    <td class="tg-mwxe">18/20</td>
    <td class="tg-cly1">Friendliness</td>
    <td class="tg-mwxe">15/20</td>
    <td class="tg-cly1">Sympathy</td>
    <td class="tg-mwxe">15/20</td>
    <td class="tg-cly1">Self-Consciousness</td>
    <td class="tg-mwxe">10/20</td>
  </tr>
  <tr>
    <td class="tg-cly1">Liberalism</td>
    <td class="tg-mwxe">13/20</td>
    <td class="tg-cly1">Self-Efficacy</td>
    <td class="tg-mwxe">17/20</td>
    <td class="tg-cly1">Gregariousness</td>
    <td class="tg-mwxe">13/20</td>
    <td class="tg-cly1">Trust</td>
    <td class="tg-mwxe">11/20</td>
    <td class="tg-cly1">Vulnerability</td>
    <td class="tg-mwxe">8/20</td>
  </tr>
</tbody></table>
</details>

> The first row is the five factors, and the six columns corresponding to each factor are its sub-dimensions, sorted alphabetically. Detailed descriptions and other personality test results (e.g., MBTI) are available on [my blog]({{site.baseurl}}/posts/Personality).  -->


<!-- ![garage](/assets/img/avatar/garage3c.png)
_Generated by ChatGPT-4._ -->

---

## Contact

{% include contact_shared.html %}

<!-- <details>
<summary>The meanings of my names: [Click to expand]</summary>
<span translate="no">
<ul>
  <li>In Chinese, the phonetic abbreviation for <code>301</code> is <code>sbly,</code> where <code>sb</code> stands for "silly," and <code>ly</code> is my name.</li>
  <li>When applying for the email address, I was worried that others might confuse the number <code>0</code> with the letter <code>o</code>, so I changed <code>301</code> to <code>3h1</code>, but actually, <code>3h1</code> is <code>310</code>. Haha, silly me indeed!</li>
  <li><code>RSVP</code> is an abbreviation for the French phrase "Répondez s'il vous plaît," which translates to "<em>Respond, if you please</em>" in English. It is commonly used on invitations to request that the invitee confirm whether or not they will attend the event.</li>
  <li>Here, <code>R01SVP</code> acts like a cry or a spell from a bystander's perspective, and the entity I am inviting is the objective world.</li>
  <li>"不梦眠" means "Dreamless Sleep" in English, referring to a deep, uninterrupted sleep or metaphorically, to death.</li>
</ul>
</span>
</details> -->



![pic2](/assets/img/about/bird1.png){: width="500" }
_["False Knees" by Joshua](https://falseknees.com/about.html)_
