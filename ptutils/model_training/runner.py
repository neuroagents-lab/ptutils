import os
import torch.distributed as dist

import torch

# WARNING: only uncomment the line below for debugging purposes
#          as it slows down training significantly, especially on TPU
# torch.autograd.set_detect_anomaly(True)

from ptutils.core.default_dirs import MODEL_SAVE_DIR
from ptutils.core.default_constants import DDP_PORT, TPU_ZONE, USE_TPU
from ptutils.model_training.train_utils import (
    construct_save_dir,
    get_resume_checkpoint_path,
    parse_config,
)

class Runner:
    def train(self, config_file):
        # Note: this can be subclasses how you like, with the rest being unmodified
        # we use normal print functions here, as a further
        # assurance/sanity check that the number of print statements
        # is the number of GPUs/TPU cores
        if config_file["trainer"] == "SupervisedImageNet":
            from ptutils.model_training.supervised_imagenet_trainer import (
                SupervisedImageNetTrainer,
            )
            print("Using Supervised ImageNet Trainer")
            trainer = SupervisedImageNetTrainer(config_file)
        else:
            raise ValueError("Invalid task.")

        trainer.train()

    def gpu_train(self, rank, args):
        # overwrite config file with the gpu id that process is on
        assert isinstance(args, dict)
        args["gpus"] = [args["gpus"][rank]]
        dist.init_process_group(
            backend="nccl", init_method="env://",
            world_size=args["world_size"], rank=rank
        )
        self.train(args)


    def tpu_train(self, rank, args):
        self.train(args)

    def run(self, args):
        config_file = args.config
        if not os.path.isfile(config_file):
            raise ValueError(f"{config_file} is invalid.")
        else:
            config_file = parse_config(config_file)
            config_file["filepath"] = args.config

        if "save_dir" not in config_file.keys():
            if "save_prefix" not in config_file.keys():
                config_file["save_prefix"] = MODEL_SAVE_DIR
            config_file["save_dir"] = construct_save_dir(
                save_prefix=config_file["save_prefix"], config=config_file
            )

        if args.resume_epoch is not None:
            config_file["resume_checkpoint"] = get_resume_checkpoint_path(
                save_dir=config_file["save_dir"],
                resume_epoch=args.resume_epoch
            )

        tpu = config_file.get("tpu", USE_TPU)
        if tpu:
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.xla_multiprocessing as xmp
            from ptutils.model_training.gcloud_utils import configure_tpu

            if "tpu_zone" not in config_file.keys():
                config_file["tpu_zone"] = TPU_ZONE
            configure_tpu(tpu, config_file["tpu_zone"])

            xmp.spawn(self.tpu_train, args=(config_file,), nprocs=None, start_method="fork")
        else:
            import torch.multiprocessing as mp

            # configure address and port to listen to
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = f"{config_file.get('ddp_port', DDP_PORT)}"

            # determine world size
            if not isinstance(config_file["gpus"], list):
                config_file["gpus"] = [config_file["gpus"]]
            # dist.get_world_size() does not work since it is yet to be initialized
            config_file["world_size"] = len(config_file["gpus"])

            mp.spawn(self.gpu_train, args=(config_file,), nprocs=config_file["world_size"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--resume-epoch", type=str, default=None)
    args = parser.parse_args()
    runner = Runner()
    runner.run(args)

