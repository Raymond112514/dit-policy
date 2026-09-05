# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import traceback

import numpy as np
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf

import hydra
from data4robotics import misc, transforms

base_path = os.path.dirname(os.path.abspath(__file__))


def print_green(text):
    _green, _reset = "\033[92m", "\033[0m"
    print(f"{_green}{text}{_reset}")

def step_checkpoint_path(cfg: DictConfig, global_step: int) -> str:
    """Milestone checkpoint, e.g. bc_ac20_..._step10000.ckpt (not overwritten)."""
    stem, ext = os.path.splitext(cfg.checkpoint_path)
    if not ext:
        ext = ".ckpt"
    return f"{stem}_step{global_step}{ext}"


@hydra.main(
    config_path=os.path.join(base_path, "experiments"), config_name="finetune.yaml"
)
def bc_finetune(cfg: DictConfig):
    print_green(cfg)
    try:
        resume_model = misc.init_job(cfg)

        # set random seeds for reproducibility
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed + 1)

        # build agent from hydra configs
        with open("agent_config.yaml", "w") as f:
            agent_yaml = OmegaConf.to_yaml(cfg.agent, resolve=True)
            f.write(agent_yaml)

        agent = hydra.utils.instantiate(cfg.agent)
        trainer = hydra.utils.instantiate(cfg.trainer, model=agent, device_id=0)

        # build task, replay buffer, and dataloader
        task = hydra.utils.instantiate(
            cfg.task, batch_size=cfg.batch_size, num_workers=cfg.num_workers
        )

        # create a gpu train transform (if used)
        gpu_transform = (
            transforms.get_gpu_transform_by_name(cfg.train_transform)
            if "gpu" in cfg.train_transform
            else None
        )

        # restore/save the model as required
        if resume_model is not None:
            misc.GLOBAL_STEP = trainer.load_checkpoint(resume_model)
        elif misc.GLOBAL_STEP == 0:
            trainer.save_checkpoint(cfg.checkpoint_path, misc.GLOBAL_STEP)
        assert misc.GLOBAL_STEP >= 0, "GLOBAL_STEP not loaded correctly!"

        # register checkpoint handler (always writes latest checkpoint_path)
        misc.set_checkpoint_handler(trainer, cfg.checkpoint_path)
        print(f"Starting at Global Step {misc.GLOBAL_STEP}")

        trainer.set_train()
        train_iterator = iter(task.train_loader)
        for itr in (
            pbar := tqdm.tqdm(range(cfg.max_iterations), postfix=dict(Loss=None))
        ):
            if itr < misc.GLOBAL_STEP:
                continue

            # infinitely sample batches until the train loop is finished
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(task.train_loader)
                batch = next(train_iterator)

            # handle the image transform on GPU if specified
            if gpu_transform is not None:
                if len(batch) == 3:
                    (imgs, obs), actions, mask = batch
                    task = None
                else:
                    (imgs, obs), actions, mask, task = batch
                imgs = {k: v.to(trainer.device_id) for k, v in imgs.items()}
                imgs = {k: gpu_transform(v) for k, v in imgs.items()}
                if task is None:
                    batch = ((imgs, obs), actions, mask)
                else:
                    batch = ((imgs, obs), actions, mask, task)

            trainer.optim.zero_grad()
            loss = trainer.training_step(batch, misc.GLOBAL_STEP)
            loss.backward()
            trainer.optim.step()

            pbar.set_postfix(dict(Loss=loss.item()))
            misc.GLOBAL_STEP += 1

            if misc.GLOBAL_STEP % cfg.schedule_freq == 0:
                trainer.step_schedule()

            if misc.GLOBAL_STEP % (cfg.eval_freq + 1000000000) == 0:
                trainer.set_eval()
                task.eval(trainer, misc.GLOBAL_STEP)
                trainer.set_train()

            if misc.GLOBAL_STEP >= cfg.max_iterations:
                final_step_path = step_checkpoint_path(cfg, misc.GLOBAL_STEP)
                print_green(
                    f"Saving final checkpoints at step {misc.GLOBAL_STEP}: "
                    f"{final_step_path} and {cfg.checkpoint_path}"
                )
                trainer.save_checkpoint(final_step_path, misc.GLOBAL_STEP)
                trainer.save_checkpoint(cfg.checkpoint_path, misc.GLOBAL_STEP)
                return
            elif cfg.save_freq > 0 and misc.GLOBAL_STEP % cfg.save_freq == 0:
                step_path = step_checkpoint_path(cfg, misc.GLOBAL_STEP)
                print_green(
                    f"Saving checkpoint at step {misc.GLOBAL_STEP}: "
                    f"{step_path} (latest: {cfg.checkpoint_path})"
                )
                trainer.save_checkpoint(step_path, misc.GLOBAL_STEP)
                trainer.save_checkpoint(cfg.checkpoint_path, misc.GLOBAL_STEP)

    # gracefully handle and log errors
    except Exception:
        traceback.print_exc(file=open("exception.log", "w"))
        with open("exception.log", "r") as f:
            print(f.read())


if __name__ == "__main__":
    bc_finetune()
