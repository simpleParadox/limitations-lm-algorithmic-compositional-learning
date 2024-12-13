import hydra
import omegaconf
from omegaconf import DictConfig
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
import importlib
from models import *
from data_modules import *
import torch
from lightning.pytorch.loggers import WandbLogger # type: ignore
import wandb
import os
os.environ["WANDB_MODE"] = "disable"
print("imports done.")


# def set_env():
#         # specific stuff we needed to make it work.
#         LSB_MCPU_HOSTS = os.environ["LSB_MCPU_HOSTS"].split(' ')  # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
#         HOST_LIST = LSB_MCPU_HOSTS[::2]  # Strips the cores per node items in the list
#         LSB_JOBID = os.environ["LSB_JOBID"]  # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
#         os.environ["MASTER_ADDR"] = HOST_LIST[0]  # Sets the MasterNode to thefirst node on the list of hosts
#         os.environ["MASTER_PORT"] = "5" + LSB_JOBID[-5:-1]
#         os.environ["NODE_RANK"] = str(HOST_LIST.index(os.environ["HOSTNAME"]))  # Uses the list index for node rank, master node rank must be 0
#         os.environ["NCCL_SOCKET_IFNAME"] = "ib,bond"  # avoids using docker of loopback interface
#         os.environ["NCCL_IB_CUDA_SUPPORT"] = '1'  # Force use of infiniband
# # --------------------------------------------------------------------------------------------------------------------------------------------

class SafeModelCheckpoint(ModelCheckpoint):
    # because the checkpointing crashed my code regularly, I added this class to catch the error and continue training
    def on_train_epoch_end(self, trainer, pl_module):
        try:
            # Call the original on_train_epoch_end method
            super().on_train_epoch_end(trainer, pl_module)
        except Exception as e:
            # Handle the exception (e.g., log it)
            print(f"An error occurred while saving a checkpoint: {e}")
            # Optionally, you can also add more sophisticated error handling here



import subprocess
def fix_infiniband():
    ibv = subprocess.run('ibv_devinfo', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = ibv.stdout.decode('utf-8').split('\n')
    exclude = ''
    for line in lines:
        if 'hca_id:' in line:
            name = line.split(':')[1].strip()
        if '\tport:' in line:
            port = line.split(':')[1].strip()
        if 'link_layer:' in line and 'Ethernet' in line:
            exclude = exclude + f'{name}:{port},' # type: ignore

    if exclude:
        exclude = '^' + exclude[:-1]
        print(exclude)
        os.environ['NCCL_IB_HCA'] = exclude

@hydra.main(version_base=None, config_path="configs", config_name="config") # This is the path to the 'config' file.
def train(cfg: DictConfig):
    if cfg.n_gpus > 0:
        fix_infiniband()
        # set_env() # DISABLED, because cluster specific.
    print("preparing training.")
    wandb_logger = WandbLogger(
        # log_model="all", 
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        project="simpleparadox_limitations_of_transformers",
        entity="simpleparadox",
    )
    run = wandb_logger.experiment
    display_name = str(run.name)
    print("display_name: ", display_name)
    SEED = 42
    seed_everything(SEED, workers=True)
    import pdb; pdb.set_trace()
    data_module_name = cfg.dataset_config.data_module_name
    dm = eval(data_module_name)(cfg, logger=print)
    dynamic_config = dm.dynamic_cfg
    model_name = cfg.model_config.model_name
    cfg.resume_from_checkpoint = cfg.resume_from_checkpoint
    if cfg.resume_from_checkpoint is not None:
        # we copy the checkpoint, change its name to make sure it won't be deleted.
        checkpoint_path = cfg.resume_from_checkpoint
        checkpoint_folder = os.path.dirname(checkpoint_path)
        all_checkpoints = os.listdir(checkpoint_folder)
        checkpoint_name = os.path.basename(checkpoint_path)
        if checkpoint_name not in all_checkpoints:
            checkpoint_name_base = checkpoint_name.replace("_(1)", "")
            for checkpoint in all_checkpoints:
                if checkpoint.replace("_(1)", "") == checkpoint_name_base:
                    checkpoint_path = os.path.join(checkpoint_folder, checkpoint)
        checkpoint_path_incr = checkpoint_path.replace(".ckpt", "_(1).ckpt")
        while os.path.exists(checkpoint_path_incr):
            checkpoint_path_incr = checkpoint_path_incr.replace(".ckpt", "_(1).ckpt")
        # load the checkpoint and save it with the new name without modifying the checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            torch.save(checkpoint, checkpoint_path_incr)
            del checkpoint # save memory
            print("checkpoint was copied for safety against deletion.")
        else:
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} wasn not found.")
    seed_everything(cfg.model_config.seed, workers=True) # to make sure, model and dataloading are indepent of each other
    model=eval(model_name)(cfg, dynamic_config)
    end_of_epoch_callback = SafeModelCheckpoint(save_top_k=1, dirpath=f"{cfg.checkpoint_path}{display_name}", filename="model-{epoch:02d}",every_n_epochs=1, save_on_train_epoch_end=True)
    seed_everything(SEED+2, workers=True)
    trainer=Trainer(
        profiler="simple",
        strategy='ddp' if cfg.n_gpus > 1 else 'auto',
        fast_dev_run=False,
        deterministic=True,
        devices=cfg.n_gpus if cfg.n_gpus > 0 else 'auto',
        accelerator="gpu" if cfg.n_gpus > 0 else 'cpu',
        callbacks=[end_of_epoch_callback], 
        precision="16-mixed",
        logger=wandb_logger,
        max_steps=cfg.max_steps,
        max_epochs=cfg.n_epochs,
        check_val_every_n_epoch=1,
        val_check_interval=cfg.dataset_config.val_check_interval,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        log_every_n_steps=cfg.log_every_n_steps,
        gradient_clip_val=cfg.model_config.gradient_clip_val,
        )
    trainer.fit(model,datamodule=dm, ckpt_path=cfg.resume_from_checkpoint)




if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ['MASTER_ADDR'] = os.environ.get('LSF_FROM_HOST', 'localhost')
    train()
