# Multi-Agent Reinforcement Learning with LLMs

Our code implementation is mostly based on MARTI framework.

## 📦 Installation

```bash
uv venv marllm --python 3.9 && source marllm/bin/activate && uv pip install --upgrade pip

uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install vllm==0.8.5.post1
uv pip install setuptools && uv pip install flash_attn==2.7.4.post1 --no-build-isolation

cd MARLLM
uv pip install -r requirements.txt
```

Follow the setup instructions for dependencies, including OpenRLHF, Ray, and vLLM.

## ⚙️ Usage

### 🔁 Multi-Agent Inference

MARTI supports:
- Built-in DAG-based workflows: debate, mixture-of-agents, chain-of-agents
- Third-party frameworks: AutoGen and CAMEL (Experimental)

Example:

```bash
MODEL_DIR="Path to models, like Qwen2.5-3B"

# See the script for more inference examples
bash scripts/run_test_mas.sh ${MODEL_DIR}
```

### 🏋️ Multi-Agent Training

MARTI supports:
- Rule-based rewards (Reward Shaping)
- Generative reward models (LLM-as-Judge) (Experimental)
- Tree-based AgentPRM (ImplicitPRM) (Experimental)
- Supervised fine-tuning + RL (e.g., PPO, GRPO)

Example:

```bash
# Minimum hardware requirement for training with 3 Qwen2.5-3B agents: approximately 6×80G GPUs

MODEL_DIR="Path to models, like Qwen2.5-3B"
WANDB_KEY="API key of wandb"

# Train Single Agent with GRPO
bash scripts/run_train_grpo.sh ${MODEL_DIR} ${WANDB_KEY}

# Train Multi-Agent Debate with Reinforce++
bash scripts/run_train_mad.sh ${MODEL_DIR} ${WANDB_KEY}
```

### 🔥 Customised Async Step and Workflow

We introduce asynchronous tool use and workflow support for both single-agent and multi-agent RL pipelines. These features make our framework more modular, efficient, and scalable for a variety of RL scenarios.

**Single Agent Tool Use**
- Modular Steps (`marti/worlds/steps`): Each agent's actions are now organized in step files (e.g., `xxx_step.py`), making it easy to customize and extend for new tasks.
- Expanded Toolset (`marti/worlds/tools`): Our agents now have access to a broader range of tools for agentic decision-making, enabling richer interactions and problem-solving capabilities.

```bash
# Multi-turn Code RL
bash scripts/run_train_grpo_code.sh

# Multi-turn Search RL
bash scripts/run_train_grpo_search.sh
```

> Note: You can refer to [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) and [bytedance/SandboxFusion](https://github.com/bytedance/SandboxFusion) separately to set up search and code tool services.

**Multi-Agent Workflow**
- Workflow Orchestration (`marti/worlds/workflows`): We now support orchestrating complex multi-agent environments via modular workflow files (e.g., `xxx_workflow.py`). This allows coordinated interactions between multiple agents in a flexible and easily configurable manner.
- Advanced Processors (`marti/worlds/workflows`): Integrated processors (e.g., `xxx_processor.py`) support advanced reward shaping and custom feedback loops, empowering more sophisticated learning dynamics and agent cooperation/competition.

```bash
# Chain-of-agents (MathChat)
bash scripts/run_train_mathchat_async.sh

# Multi-agent Debate
bash scripts/run_train_mad_async.sh
```

These improvements open up new possibilities for research and deployment in both single-agent and multi-agent RL settings. As always, we're keen for your feedback and contributions!



### 📊 Preliminary Experiments

#### Training Details

We employ the MARTI framework to train both base and reasoning models, specifically `Qwen2.5-3B` and `DeepScaleR-1.5B-Preview`. For `Qwen2.5-3B`, we implement DeepSeek-R1 zero-like reinforcement learning training using Level 3-5 samples from the MATH dataset. The `DeepScaleR-1.5B-Preview` model, which exhibits strong inherent reasoning capabilities but presents training challenges, undergoes [Test-Time Reinforcement Learning (TTRL)](https://github.com/PRIME-RL/TTRL) adaptation on AIME benchmark data. For multi-agent reinforcement learning, we employ a cluster configuration consisting of 3 nodes, each equipped with 8 A800 80GB GPUs, allocating one full node per agent.

#### Benchmark Results
We compare non-reasoning and reasoning models under various configurations and show that majority voting consistently outperforms multi-agent workflows when trained conventionally. This reflects known limitations of current LLM-based agent systems, such as poor role adherence and ineffective inter-agent communication.

To address this, MARTI enhances model reasoning through structured agent interactions. As shown in Figure 2 and Figure 3, our experiments show that:

- MARTI-trained base models outperform standard RL setups and rival instructed models.
- Large reasoning models trained with MARTI using TTRL achieve state-of-the-art results on challenging tasks (e.g., 66.7 AIME score with Multi-Agent Debates).
- Multi-agent RL consistently surpasses single-agent systems in performance under the same compute budget.

<p align="center">
  <img src="./assert/qwen2.5-3b-base-instruct-avg.jpg" width="800">
</p>
<p align="center"><i>Figure 2: Average scores of Qwen2.5-3B base and instruct models under different budget and settings</i></p>


<p align="center">
  <img src="./assert/ds-1.5-qwen-1.7-avg.jpg" width="800">
</p>
<p align="center"><i>Figure 3: Average scores of reasoning models under different budget and settings</i></p>


#### Training Dynamics

##### Multi-Agents Debate
We conduct multi-agent debate training with `Qwen2.5-3B` The `Qwen2.5-3B` model is trained using REINFORCE++ on Level 3 to 5 samples from the MATH-500 dataset.

<p align="center">
  <img src="./assert/mad-rl-amc.jpg" width="400">
  <img src="./assert/mad-rl-math.jpg" width="400">
</p>
<p align="center"><i>Figure 4: Accuracy of MAD (Qwen2.5-3B, MATH) on AMC and MATH</i></p>


<p align="center">
  <img src="./assert/mad-dynamics.jpg" width="800">
</p>
<p align="center"><i>Figure 5: Training Dynamics of MAD (Qwen2.5-3B, MATH)</i></p>


##### Mixture-of-Agents
We evaluate a mixture-of-agents approach using the `Qwen2.5-3B` model, trained on Levels 3 through 5 of the MATH-500 training dataset.

<p align="center">
  <img src="./assert/moa-rl-amc.jpg" width="400">
  <img src="./assert/moa-rl-math.jpg" width="400">
</p>
<p align="center"><i>Figure 6: Accuracy of MoA (Qwen2.5-3B, MATH) on AMC and MATH</i></p>


<p align="center">
  <img src="./assert/moa-dynamics.jpg" width="800">
</p>
<p align="center"><i>Figure 7: Training Dynamics of MoA (Qwen2.5-3B, MATH)</i></p>


## 📚 Documentation
- [Overview of MARTI](./docs/1-Overview-Of-MARTI.md)
- [Workflows Integration](./docs/2-Workflows-Integration.md)
- [Reward and Training](./docs/3-Reward-And-Training.md)

## 🚩 Roadmap

- [ ] Release MARTI Technical Report
- [ ] Initial support for agentic tasks (e.g., GAIA benchmark)
- [ ] More features are working in progress

## 👏 Acknowledge

MARTI is developed primarily based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). We would like to express our gratitude to the developers of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), as well as to the teams behind [vLLM](https://github.com/vllm-project/vllm), [Ray](https://github.com/ray-project/ray) and [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) for their invaluable contributions.

## 🤝 Core Contributors
- Project Lead: [Kaiyan Zhang](https://iseesaw.github.io/)
- Agent Group: [Runze Liu](https://ryanliu112.github.io/), [Kaiyan Zhang](https://iseesaw.github.io/), [Kai Tian](https://github.com/XiaoTiank), [Guoli Jia](https://github.com/exped1230), [Xingtai Lv](https://github.com/telxt), [Che Jiang](https://github.com/dcdsf321)
- RL Group: [Kaiyan Zhang](https://iseesaw.github.io/), [Xuekai Zhu](https://github.com/Xuekai-Zhu), [Sihang Zeng](https://github.com/zengsihang), [Yuchen Fan](https://github.com/YuchenFan48), [Yuxin Zuo](https://github.com/yuxinzuo)

For the full list of contributors, please refer to the author list in the citation. We are also deeply grateful to everyone who engaged in discussions and provided valuable feedback throughout the development of this project.

## 📬 Contact

For issues or inquiries: 
- Kaiyan Zhang, Tsinghua University (zhang-ky22@mails.tsinghua.edu.cn)
- Biqing Qi, Shanghai AI Lab (qibiqing@pjlab.org.cn)

## 🔬 Citation

If you use MARTI in your research, please cite the project:

```
@misc{marti2025,
  title={MARTI: A Framework for Multi-Agent LLM Systems Reinforced Training and Inference},
  author={Kaiyan Zhang and Runze Liu and Xuekai Zhu and Kai Tian and Sihang Zeng and Guoli Jia and Yuchen Fan and Xingtai Lv and Yuxin Zuo and Che Jiang and Ziyang Liu and Jianyu Wang and Yuru Wang and Ruotong Zhao and Ermo Hua and Yibo Wang and Shijie Wang and Junqi Gao and Xinwei Long and Youbang Sun and Zhiyuan Ma and Ganqu Cui and Lei Bai and Ning Ding and Biqing Qi and Bowen Zhou},
  year={2025},
  institution={Tsinghua University and Shanghai AI Lab},
  url={https://github.com/TsinghuaC3I/MARTI}
}
```

## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TsinghuaC3I/MARTI&type=Date)](https://www.star-history.com/#TsinghuaC3I/MARTI&Date)

