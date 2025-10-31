<img src="figures/logo.png" width="300" /></a><br>
<b>The Conductor</b><br>
</h1>

## Weight download from anymouse huggingface repo
[Conductor Weights](https://huggingface.co/AnonSubmission38/Conductor)

## Installation

Tested with Python `python=3.11`

```sh
scripts/install.sh
```

Or:

```sh
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install vllm==0.8.3 tensorboard
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/

python -m pip install --upgrade -r requirements.txt
```

## Conductor evaluation script

Tasks are contained in the eval/ directory. For example to run MMLU:

```sh
sh eval/mmlu/eval_conductor.sh
```

## Where to find key Conductor components

### Conductor
The conductor model is found in trainers/conductor_engine.py as ConductorReward. The main routing logic is implemented in the _multi_step_coordination() method. 

### Routing templates
Routing templates are found in custom_data/routing_question_formats.py. They can be imported for use in custom_data/data_utils.py

### Run config
For the newest task, MMRL (MMLU, RLPR, Livecodebench, Math500), see cfgs/run_cfg/run_conductor_mix_mmrl.yaml. Set vllm_mode = server and launch_with_vllm as per the usage script instructions above.

### Data load in
Tasks are handled through the proj_guf/guf/tasks/ and custom_data/data_utils.py. 
