import logging 
import os 
import hydra 
from omegaconf import DictConfig ,OmegaConf 
from datetime import datetime 
from transformers.trainer_utils import get_last_checkpoint 
import wandb 

os.environ ["HF_HUB_ENABLE_HF_TRANSFER"]="1"

logging.basicConfig(level =logging.INFO )
logger =logging.getLogger(__name__ )

for noisy in (
"httpx",
"httpcore",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger(noisy).propagate =False 


def wandb_init (cfg, run_name : str, group_name : str, log_dir : str):

    config_dict =OmegaConf.to_container(
    cfg,
    resolve=True,
    throw_on_missing=False,
    )
    config_dict["log_dir"]=log_dir 
    config_dict["wandb_run_name"]=run_name 
    config_dict["wandb_group_name"]=group_name 

    wandb_run =wandb.init(
    project=cfg.wandb_project ,
    group=group_name,
    name=run_name,
    config=config_dict ,
    )
    return wandb_run


def get_checkpoint(output_dir):
    if os.path.isdir(output_dir):
        return get_last_checkpoint(output_dir)
    return None 

def get_total_devices ():
    world_size =os.environ.get("WORLD_SIZE")
    if world_size is not None :
        return int(world_size)
    return 1 

def compute_accumulation_steps(train_batch_size,per_device_train_batch_size):
    total_devices=get_total_devices()

    div=per_device_train_batch_size*total_devices 
    steps=train_batch_size /div 
    if not steps.is_integer ():
        raise ValueError (
        "train_batch_size must be divisible by "
        f"per_device_batch*total_devices={div}"
        )
    return int(steps)


@hydra.main(config_path ="cfgs",config_name ="train",version_base=None)
def main(cfg:DictConfig):
    logger.info (f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    if "LOCAL_RANK" in os.environ :
        is_main_process =int(os.environ["LOCAL_RANK"])==0 
    elif "RANK" in os.environ :
        is_main_process =int(os.environ["RANK"])==0 
    else :
        is_main_process =True 


    if OmegaConf.is_missing (cfg ,"gradient_accumulation_steps"):
        accumulation_steps =compute_accumulation_steps(
        train_batch_size =cfg.train_batch_size,
        per_device_train_batch_size =cfg.per_device_train_batch_size)
        cfg.gradient_accumulation_steps=accumulation_steps 


    logger.info(f"Accumulation steps {cfg.gradient_accumulation_steps} ----")

    using_wandb = False 
    if isinstance (cfg.report_to,str):
        using_wandb =cfg.report_to =='wandb'
    elif cfg.report_to is not None :
        for v in cfg.report_to :
            using_wandb = using_wandb or (v =='wandb')

    if using_wandb and is_main_process :
        wandb_run =wandb_init (
        cfg =cfg ,
        group_name =cfg .wandb_group_name ,
        run_name =cfg .wandb_run_name ,
        log_dir =cfg .output_dir ,
        )

    tokenizer =hydra .utils .instantiate (cfg .make_tokenizer_fn )

    datasets =hydra .utils .instantiate (cfg .make_dataset_fn ,tokenizer =tokenizer )

    if cfg .evaluate_only :
        print (f'Running evaluation from checkpoint {cfg.evaluate_only}')
        last_checkpoint =cfg .evaluate_only 
        cfg .trainer .model =last_checkpoint 
        trainer =hydra .utils .instantiate (
        cfg .trainer ,
        **datasets ,
        )

        try :
            from collections import Counter 
            eval_wrapped =datasets ["eval_dataset"]
            base =getattr (eval_wrapped ,"dataset",None )
            if base is not None and "task_type"in base .column_names :
                counts =Counter (base ["task_type"])
                total =len (base )
            else :
                print ("No task_type column found")

            print ("Evaluation dataset summary:")
            print (f"  total: {total}")
            for k in sorted (counts ):
                print (f"  {k}: {counts[k]}")
        except Exception as e :
            print (f"Evaluation dataset summary failed: {e}")

        results =trainer .evaluate ()
        print (results )
        trainer .log_metrics ("eval",results )
        return 

    # Training mode
    last_checkpoint = None
    if cfg .resume_from:
        last_checkpoint = cfg .resume_from
    else:
        last_checkpoint = get_checkpoint(cfg .output_dir)
    
    if last_checkpoint:
        logger .info (f"Resuming training from {last_checkpoint}")
        cfg .trainer .model = last_checkpoint
    
    trainer = hydra .utils .instantiate (
        cfg .trainer ,
        **datasets ,
    )
    
    if last_checkpoint:
        trainer .train (resume_from_checkpoint=last_checkpoint)
    else:
        trainer .train ()











































if __name__ =="__main__":
    main ()
