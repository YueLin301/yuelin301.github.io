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

Hi, this is <span translate="no">Yue Lin</span> (/ˈjuːeɪ lɪn/, or 林越 in Chinese), and welcome to my personal website.
Currently I am a Ph.D. student (data science program) in School of Data Science at The Chinese University of Hong Kong, Shenzhen, fortunately advised by Prof. <span translate="no"><a href="https://bxiangwang.github.io">Baoxiang Wang [Homepage]</a></span> and Prof. <span translate="no"><a href="https://scholar.google.com/citations?user=n1DQMIsAAAAJ&amp;hl=en&amp;oi=ao">Hongyuan Zha [Google Scholar]</a></span>. 
I am also a joint Ph.D. student at the Shenzhen Loop Area Institute.

<ul>
    <li>My research interests lie primarily in designing efficient learning algorithms to guide agents toward better equilibriums in multi-agent tasks. For experts: (1) solving sequential social dilemmas, by designing learning algorithms that incorporate mechanism design, and (2) designing learning methods for mechanism design problems.</li>
    <li>Most of my focus is on sequential mixed-motive multi-agent scenarios, and I am particularly interested in communication mechanisms (keywords: information design, Bayesian persuasion, cheap talk), as well as some other mechanisms that influence others, such as incentivization.</li>
    <li>My expertise in learning methods lies in reinforcement learning (RL), particularly multi-agent hyper-gradient modeling, and in large language models (LLMs).</li>
</ul>

<!-- Here, I will irregularly update my experiences, notes, and computational magic spells that interest me. -->

<!-- <details style="margin-top: -10px; margin-bottom: 10px;">
<summary>My current research interest lies in using computational methods to study mechanisms, to address some social dilemma issues in game theory, in which scenarios everyone being self-interested may lead to the detriment of social welfare. [Click here to see details of my research interests]</summary>
<ul>
  <li>
    <ul>
      <li>The method I use is Multi-Agent Reinforcement Learning (MARL), essentially with the rationality assumption. And I have recently started working with some tools that seem more "human-like", like large language models.</li>
      <li>Specifically I am focusing on the problem of Bayesian persuasion (BP, <a href="{{site.baseurl}}/posts/Information-Design-10min/">[my blog]</a>) in economics: a sender with an informational advantage tries to persuade a receiver, who has different motives, to take actions that are beneficial to the sender. My research is on sequential decision-making situations.</li>
      <li>A representative work on this is <a href="{{site.baseurl}}/posts/IDMARL/">this [my blog]</a> (NeurIPS), where we proposed a general model-free RL algorithm for multi-agent communication (for the cognoscenti: policy gradient for communication), and expanded the constraints in BP so that mixed-motive communication (even between two agents) in MARL is conceivable.</li>
    </ul>
  </li>
  <li>During my undergraduate years (specifically, from 2019 to 2021), I dabbled in robotics, understood kinematics, and played a bit with <a href="https://github.com/YueLin301/robot-dynamics-mynotes">dynamics [my repo]</a>.
    <ul>
      <li>I presented a purely simulation-based robotic mechanism design <a href="{{site.baseurl}}/posts/RHex-T3/">work [my blog]</a> at ICRA. It is about a hybrid leg can transform into various forms (wheel, legs, RHex) to adapt to different terrains, and can even climb ladders.</li>
      <li>Also I <a href="{{site.baseurl}}/posts/SSCC/">have implemented [my blog]</a> a "gimbal" using a hyper-redundant manipulator (purely based on kinematics), allowing it to efficiently reach into barrels.</li>
    </ul>
  </li>
</ul>
</details> -->

<div style="color: #999;">
  And how could one endure being a man, if not also for the possibility to create, guess riddles, and redeem accidents? To redeem those who lived in the past and to recreate all "it was" into "thus I willed it" — that alone should I call redemption. <i>—&nbsp;Friedrich&nbsp;Nietzsche,&nbsp;Thus&nbsp;Spoke&nbsp;Zarathustra.</i>
</div>

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

- <span translate="no">**Shenzhen Loop Area Institute**</span>  
  Joint Ph.D. Student (2025.9 - Present)  
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

- <span translate="no">**The Chinese University of Hong Kong, Shenzhen**</span>  
  Research Assistant in School of Data Science (2022.2 - Present)

---

## Selected Publications

<!-- [[Google Scholar]](https://scholar.google.com/citations?user=fbvQHX4AAAAJ&hl=zh-CN) -->

### Data Science

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

- <span translate="no">Verbalized Bayesian Persuasion.</span>  
    <span translate="no">Wenhao Li, **Yue Lin**, Xiangfeng Wang, Bo Jin, Hongyuan Zha, Baoxiang Wang.</span>  
    <span translate="no">*EC 2025 Workshop: Information Economics and Large Language Models.*</span>
    > 
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




<!-- - Information Design in Multi-Agent Reinforcement Learning.  
    **Yue Lin**, Wenhao Li, Hongyuan Zha, Baoxiang Wang.  
    *Neural Information Processing Systems (NeurIPS) 2023*.
    > Poster. This is currently my most representative work.  
    [[Paper]](https://arxiv.org/abs/2305.06807) 
    [[Code]](https://github.com/YueLin301/InformationDesignMARL) 
    [[Experiments]](https://wandb.ai/yuelin301/IND+MARL?nw=nwuseryuelin301) 
    [[Blog en]]({{site.baseurl}}/posts/IDMARL/) 
    [[Blog cn]]({{site.baseurl}}/posts/IDMARL-cn/) 
    [[Zhihu cn]](https://zhuanlan.zhihu.com/p/687305626) 
    [[Slides]]({{site.baseurl}}/posts/IDMARL/#slides) 
    [[Talk en]](https://www.youtube.com/watch?v=yhVlpv_1Pg4) 
    [[Talk cn]](https://www.bilibili.com/video/BV1e94y177Dj/?share_source=copy_web&vd_source=b3cf9eb7cfe43c730613c5158a38e978&t=2825)   -->

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



---

## Professional Services

Independent Reviewer
- NeurIPS 2024 [6, 45615], 2025 [5, 32912]
- ICLR 2025 [3, 21831], 2026 [0]
- ICML 2025 [6, 32893]
- TMLR 2025 [2, 38363]

Volunteer
- AAMAS 2024 [3, 9876], 2025 [2, 4792]
- ICML (Position) 2025 [2, 5016]

> Numbers in brackets indicate the number of manuscripts reviewed and the character count of all reviews, respectively. A "0" means the invitation was accepted, but no review assignment has been made yet. Total reviews: 29.

---

## Teaching

Teaching Assistant
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

<ul>
<li>
    <details>
    <summary>Video Games</summary>
    <ul>
    <li>王者荣耀 Honor of Kings （已退坑）
        <ul>
        <li>全国第74 诸葛亮，战力13853，胜率57.5%，场次1956</li>
        <li>浙江省第57 嫦娥，战力10725，胜率64.3%，场次210</li>
        <li>浙江省第42 米莱狄，战力7863，胜率54.6%，场次414</li>
        <li>巅峰赛打野2063分（全国第3892名），中路1849分，巅峰场次共800</li>
        </ul>
    </li>
    <li>Overwatch 1 （已退坑）
        <ul>
        <li>Doomfist: 284 hours, 1669 matches, win rate 54%, kill/death 26068/13137</li>
        </ul>
    </li>
    <li>Steam
        <ul>
        <li>Hollow Knight, Hollow Knight: Silksong, The Stanley Parable, Batman: Arkham Knight, Marvel Rivals...</li>
        <!-- <li>Risk of Rain 2, Hollow Knight, Batman: Arkham Knight, Marvel Rivals, Lost Castle 2, baba is you...</li> -->
        </ul>
    </li>
    <!-- <li>Among all games, the one that has influenced me the most is **The Stanley Parable.**</li> -->
    </ul>
    </details>
</li>

<li>Movies</li>
<!-- <li>DC Comics</li> -->
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

<div align="center">

<style type="text/css">
.tg  {border:none;border-collapse:collapse;border-spacing:0;}
.tg td{border-style:solid;border-width:0px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;
  padding:10px 5px;word-break:normal;}
.tg th{border-style:solid;border-width:0px;font-family:Arial, sans-serif;font-size:14px;font-weight:normal;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-wp8o{border-color:#000000;text-align:center;vertical-align:top}
.tg .tg-mqa1{border-color:#000000;font-weight:bold;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-mqa1">Types</th>
    <th class="tg-mqa1">ID</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-mqa1">Email</td>
    <td class="tg-wp8o">linyue3h1@gmail.com</td>
  </tr>
  <tr>
    <td class="tg-mqa1">WeChat</td>
    <td class="tg-wp8o">R01SVP</td>
  </tr>
  <!-- <tr>
    <td class="tg-mqa1">Website</td>
    <td class="tg-wp8o"><a href="{{site.baseurl}}/" target="_blank" rel="noopener noreferrer">yuelin301.github.io</a></td>
  </tr> -->
  <tr>
    <td class="tg-mqa1">GitHub</td>
    <td class="tg-wp8o"><a href="https://github.com/YueLin301" target="_blank" rel="noopener noreferrer">github.com/YueLin301</a></td>
  </tr>
  <tr>
    <td class="tg-mqa1">Google Scholar</td>
    <td class="tg-wp8o"><a href="https://scholar.google.com/citations?user=fbvQHX4AAAAJ" target="_blank" rel="noopener noreferrer"><span translate="no">Yue Lin</span></a></td>
  </tr>
  <!-- <tr>
    <td class="tg-mqa1">LinkedIn</td>
    <td class="tg-wp8o"><a href="https://www.linkedin.com/in/linyue3h1" target="_blank" rel="noopener noreferrer"><span translate="no">linyue3h1</span></a></td>
  </tr> -->
  <tr>
    <td class="tg-mqa1">Zhihu</td>
    <td class="tg-wp8o"><a href="https://www.zhihu.com/people/R01SVP/answers" target="_blank" rel="noopener noreferrer"><span translate="no">R01SVP</span></a></td>
  </tr>
  <tr>
    <td class="tg-mqa1">Bilibili</td>
    <td class="tg-wp8o"><a href="https://space.bilibili.com/36040555" target="_blank" rel="noopener noreferrer"><span translate="no">不梦眠</span></a></td>
  </tr>
</tbody>
</table>

</div>

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
