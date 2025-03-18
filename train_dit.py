from src.models import load_model
from src.datasets import load_dataset
from src.utils import CalMetrics, one_iter_for_dit
from src.transport import create_transport, Sampler
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import transformers
import diffusers
import torch
import argparse
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import logging
import os
from glob import glob
from time import time
import yaml
import warnings

warnings.filterwarnings("ignore")
logger = get_logger(__name__, log_level="INFO")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_dit.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        update_args = yaml.unsafe_load(f)
    parser.set_defaults(**update_args)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Training currently requires at least one GPU!"
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    model_name = args.model_name
    output_dir = f"{args.output_dir}/experiments-{model_name}"
    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)
        experiment_index = len(glob(f"{output_dir}/*"))
        experiment_dir = f"{output_dir}/{experiment_index:03d}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        visualization_dir = f"{experiment_dir}/visualization_results"
        os.makedirs(visualization_dir, exist_ok=True)
        validation_dir = f"{experiment_dir}/validation_results"
        os.makedirs(validation_dir, exist_ok=True)
        logging.basicConfig(
            format="[\033[34m%(asctime)s\033[0m] - %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{experiment_dir}/log.txt")]
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        logger.info(f"Experiment directory created at {experiment_dir}")

    if args.global_seed is not None:
        set_seed(args.global_seed)

    # load dataset
    local_batch_size = args.dataloader["batch_size"]
    dataset_name = args.dataset_name
    train_dataset = load_dataset(dataset_name, **args.dataset_args[dataset_name])
    train_dataloader = DataLoader(train_dataset, **args.dataloader)
    val_dataset_name = args.val_dataset_name
    val_dataset = load_dataset(val_dataset_name, **args.dataset_args[val_dataset_name])
    val_dataloader = DataLoader(val_dataset, **args.val_dataloader)
    dataset_len = len(train_dataset)
    steps_one_epoch = dataset_len // (local_batch_size * accelerator.num_processes)
    logger.info(f"Dataset {dataset_name} contains {dataset_len:,} triplets, one epoch equals {steps_one_epoch} steps")

    # load model
    model = load_model(model_name, **args.model_args)
    logger.info(f"{model_name} Parameters: {sum(p.numel() for p in model.parameters()):,}")
    best_val_metric = 0
    if args.train_args["resume_from_ckpt"]:
        ckpt = torch.load(args.train_args["resume_from_ckpt"], map_location="cpu")
        model.load_state_dict(ckpt["eden_dit"])
        best_val_metric = ckpt["best_val_metric"]
        del ckpt
    vae = load_model("EDEN_VAE", **args.vae_args)
    vae_ckpt = torch.load(args.train_args["pretrained_vae_path"], map_location="cpu")["eden_vae"]
    vae.load_state_dict(vae_ckpt)
    logger.info(f"Loaded pretrained vae({args.train_args['pretrained_vae_path']}) for training.")
    del vae_ckpt

    # prepare optimizer, lr_scheduler and transport
    lr = args.train_args["base_lr"] * accelerator.num_processes
    train_epochs = args.train_args["epochs"]
    num_warmup_steps = args.train_args["warmup_steps"]
    num_train_steps = steps_one_epoch * train_epochs * accelerator.num_processes
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, **args.train_args["optimizer"])
    lr_scheduler = get_scheduler(args.train_args["lr_scheduler"], optimizer=optimizer,
                                 num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
    transport = create_transport(**args.transport)
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(sampling_method="euler", num_steps=2, atol=1e-6, rtol=1e-3)
    cal_metrics = CalMetrics()

    model, vae, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, vae, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # begin training
    model.train()
    vae.requires_grad_(False)
    vae.eval()
    train_steps, log_steps, running_loss = 0, 0, 0
    vae_shift, vae_scaler = args.vae_shift, args.vae_scaler
    cos_sim_mean, cos_sim_std = args.cos_sim_mean, args.cos_sim_std
    pre_best_ckpt_path = None
    start_time = time()
    total_steps = steps_one_epoch * train_steps
    logger.info(f"Training for {train_epochs} epochs ({total_steps} steps)...")
    for epoch in range(train_epochs):
        logger.info(f"Beginning epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            loss_dict, latent, cond_tokens, denoise_args = one_iter_for_dit(model, vae, batch, transport, sample_fn,
                                                                            vae_shift, vae_scaler, cos_sim_mean,
                                                                            cos_sim_std)
            pred = loss_dict["pred"]
            t = loss_dict["timestep"]
            loss = loss_dict["loss"].mean()
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            running_loss += accelerator.gather(loss.repeat(local_batch_size)).mean().item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.train_args["log_every_steps"] == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                logger.info(
                    f"(step={train_steps:07d}) DiT Loss: {avg_loss:.4f} lr: {optimizer.state_dict()['param_groups'][0]['lr']:.3e} "
                    f"Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss, log_steps = 0, 0
                start_time = time()
            if train_steps % args.train_args["visual_every_steps"] == 0:
                if accelerator.is_local_main_process:
                    frame_0, frame_1, gt = batch[:, 0, ...] / 255., batch[:, 1, ...] / 255., batch[:, 2, ...] / 255.
                    visual_results_num = args.train_args["visual_results_num"]
                    blended_input = \
                    (frame_0 * 0.5 + frame_1 * 0.5).chunk(local_batch_size // visual_results_num, dim=0)[0]
                    with torch.no_grad():
                        noise = torch.randn_like(latent).to(accelerator.device)
                        samples = sample_fn(noise, model.module.forward, **denoise_args)[-1]
                        generated = vae.module.decode(samples / vae_scaler + vae_shift, cond_tokens)
                        denoised = vae.module.decode(pred / vae_scaler + vae_shift, cond_tokens)
                        recon = vae.module.decode(latent / vae_scaler + vae_shift, cond_tokens)
                    generated_v = generated.chunk(local_batch_size // visual_results_num, dim=0)[0]
                    denoised_v = denoised.chunk(local_batch_size // visual_results_num, dim=0)[0]
                    recon_v = recon.chunk(local_batch_size // visual_results_num, dim=0)[0]
                    gt_v = gt.chunk(local_batch_size // visual_results_num, dim=0)[0]
                    gt_recon = torch.cat((blended_input, gt_v, recon_v, generated_v, denoised_v), dim=0)
                    save_image(gt_recon, f"{visualization_dir}/steps{train_steps:07d}.png", nrow=visual_results_num)
                    logger.info(f"Saved visualization results to {visualization_dir}")
            if train_steps % args.train_args["val_every_steps"] == 0 or train_steps == total_steps:
                logger.info("Validating ...")
                results = {"PSNR": 0., "SSIM": 0., "LPIPS": 0., "FloLPIPS": 0., "L1": 0.}
                val_steps = 0
                model.eval()
                val_batch_size = args.val_dataloader["batch_size"]
                for _, val_frames in enumerate(val_dataloader):
                    val_frame_0, val_frame_1, val_gt = (val_frames[:, 0, ...] / 255., val_frames[:, 1, ...] / 255.,
                                                        val_frames[:, 2, ...] / 255.)
                    val_generated = one_iter_for_dit(model, vae, val_frames, transport, sample_fn,
                                                     vae_shift, vae_scaler, cos_sim_mean, cos_sim_std, False)
                    psnr = cal_metrics.cal_psnr(val_generated, val_gt)
                    ssim = cal_metrics.cal_ssim(val_generated, val_gt)
                    lpips = cal_metrics.cal_lpips(val_generated, val_gt)
                    flolpips = cal_metrics.cal_flolpips(val_generated, val_gt, val_frame_0, val_frame_1)
                    l1 = torch.mean(torch.abs(val_generated - val_gt))
                    results["PSNR"] += accelerator.gather(psnr.repeat(val_batch_size)).mean().item()
                    results["SSIM"] += accelerator.gather(ssim.repeat(val_batch_size)).mean().item()
                    results["LPIPS"] += accelerator.gather(lpips.repeat(val_batch_size)).mean().item()
                    results["FloLPIPS"] += accelerator.gather(flolpips.repeat(val_batch_size)).mean().item()
                    results["L1"] += accelerator.gather(l1.repeat(val_batch_size)).mean().item()
                    val_steps += 1
                model.train()
                for key in results.keys():
                    results[key] /= val_steps
                format_results = (
                    f"PSNR: {results['PSNR']:.4f},  SSIM: {results['SSIM']:.4f}, LPIPS: {results['LPIPS']:.4f}, "
                    f"FloLPIPS: {results['FloLPIPS']:.4f}, L1: {results['L1']:.4f}")
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    with open(f"{validation_dir}/val_results.txt", mode="a+", encoding="utf-8") as f:
                        f.write(f"-*- Steps{train_steps:07d} -*- {results}\n")
                logger.info(f"Steps{train_steps:07d} validation results on DAVIS: {format_results}")
                now_val_metric = results[args.train_args["val_metric"]]
            if train_steps % args.train_args["ckpt_every_steps"] == 0 or train_steps == total_steps:
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    if now_val_metric >= best_val_metric:
                        unwrap_model = accelerator.unwrap_model(model)
                        checkpoint = {
                            "eden_dit": unwrap_model.state_dict(),
                            "best_val_metric": best_val_metric,
                            "args": args
                        }
                        now_best_checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, now_best_checkpoint_path)
                        if pre_best_ckpt_path:
                            os.remove(pre_best_ckpt_path)
                        best_val_metric = now_val_metric
                        pre_best_ckpt_path = now_best_checkpoint_path
                        logger.info(f"Saved the best {args.train_args['val_metric']}({best_val_metric:.4f}) checkpoints"
                                    f" in {now_best_checkpoint_path}.")

    accelerator.end_training()
    model.eval()
    logger.info("Done!")


if __name__ == "__main__":
    main()
