## Installation

```shell
uv venv orch --python 3.11 && source orch/bin/activate && uv pip install --upgrade pip
```

> [!TIP]
> For Hugging Face cluster users, add `export UV_LINK_MODE=copy` to your `.bashrc` to suppress cache warnings from `uv`

Next, install vLLM and FlashAttention:

```shell
uv pip install vllm==0.8.5.post1
uv pip install setuptools && uv pip install flash_attn==2.7.4.post1 --no-build-isolation
```

```shell
uv pip install -r requirements.txt
```

Next, log into your Hugging Face and Weights and Biases accounts as follows:

```shell
huggingface-cli login
wandb login
```

Finally, check whether your system has Git LFS installed so that you can load and push models/datasets to the Hugging Face Hub:

```shell
git-lfs --version
```

If it isn't installed, run:

```shell
sudo apt-get install git-lfs
```

## Implementation

### MATH Train with GRPO (PUB-PRI)

edit `davids/configs/ddp_config.yaml` and `davids/train/pub_pri_train/run_pub_pri_math.sh`
then run below code in terminal:

```shell
# 
cd orchestrator
bash davids/train/pub_pri_train/run_pub_pri_math.sh
```
