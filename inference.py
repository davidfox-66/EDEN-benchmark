from src.models import load_model
from src.utils import InputPadder
from src.transport import create_transport, Sampler
import torchvision
import torch
import argparse
import yaml
import os


def interpolate(frame0, frame1):
    h, w = frame0.shape[2:]
    image_size = [h, w]
    padder = InputPadder(image_size)
    difference = ((torch.mean(torch.cosine_similarity(frame0, frame1),
                              dim=[1, 2]) - args.cos_sim_mean) / args.cos_sim_std).unsqueeze(1).to(device)
    cond_frames = padder.pad(torch.cat((frame0, frame1), dim=0))
    new_h, new_w = cond_frames.shape[2:]
    noise = torch.randn([1, new_h // 32 * new_w // 32, args.model_args["latent_dim"]]).to(device)
    denoise_kwargs = {"cond_frames": cond_frames, "difference": difference}
    samples = sample_fn(noise, eden.denoise, **denoise_kwargs)[-1]
    denoise_latents = samples / args.vae_scaler + args.vae_shift
    generated_frame = eden.decode(denoise_latents)
    generated_frame = padder.unpad(generated_frame.clamp(0., 1.))
    return generated_frame


device = "cuda:0"
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/eval_eden.yaml")
parser.add_argument("--frame_0_path", type=str, default="examples/frame_0.jpg")
parser.add_argument("--frame_1_path", type=str, default="examples/frame_1.jpg")
parser.add_argument("--video_path", type=str, default=None)
parser.add_argument("--interpolated_results_dir", type=str, default="interpolation_outputs")
args = parser.parse_args()
with open(args.config, "r") as f:
    update_args = yaml.unsafe_load(f)
parser.set_defaults(**update_args)
args = parser.parse_args()
model_name = args.model_name
eden = load_model(model_name, **args.model_args)
ckpt = torch.load(args.pretrained_eden_path, map_location="cpu")
eden.load_state_dict(ckpt["eden"])
eden.to(device)
eden.eval()
del ckpt
transport = create_transport("Linear", "velocity")
sampler = Sampler(transport)
sample_fn = sampler.sample_ode(sampling_method="euler", num_steps=2, atol=1e-6, rtol=1e-3)
video_path = args.video_path
interpolated_results_dir = args.interpolated_results_dir
os.makedirs(interpolated_results_dir, exist_ok=True)
frame_0_path, frame_1_path = args.frame_0_path, args.frame_1_path
if video_path:
    print(f"Interpolating Video ({video_path}) ...")
    interpolated_video_save_path = f"{interpolated_results_dir}/interpolated.mp4"
    interpolated_video = []
    video_frames, _, video_info = torchvision.io.read_video(video_path)
    video_frames = video_frames.float().permute(0, 3, 1, 2) / 255.
    fps = video_info["video_fps"]
    frames_num = video_frames.shape[0]
    for i in range(frames_num - 1):
        with torch.no_grad():
            frame_0, frame_1 = video_frames[i].unsqueeze(0).to(device), video_frames[i + 1].unsqueeze(0).to(device)
            interpolated_frame = interpolate(frame_0, frame_1)
            interpolated_video.append(frame_0.cpu())
            interpolated_video.append(interpolated_frame.cpu())
            del frame_0, frame_1, interpolated_frame
            torch.cuda.empty_cache()
    interpolated_video.append(video_frames[-1].unsqueeze(0))
    interpolated_video = (torch.cat(interpolated_video, dim=0).permute(0, 2, 3, 1) * 255.).cpu()
    torchvision.io.write_video(interpolated_video_save_path, interpolated_video, fps=2*fps)
    print(f"Saved interpolated video in {interpolated_video_save_path}.")
elif frame_0_path and frame_1_path:
    print(f"Interpolating Image-pairs {frame_0_path}-{frame_1_path} ...")
    frame_0 = (torchvision.io.read_image(frame_0_path) / 255.).unsqueeze(0).to(device)
    frame_1 = (torchvision.io.read_image(frame_1_path) / 255.).unsqueeze(0).to(device)
    interpolated_frame = interpolate(frame_0, frame_1)
    interpolated_frame_path = f"{interpolated_results_dir}/interpolated.png"
    torchvision.utils.save_image(interpolated_frame, interpolated_frame_path)
    print(f"Saved interpolated image in {interpolated_frame_path}.")
else:
    assert "There are no images or videos to be interpolated!"

