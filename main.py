import argparse
import pathlib

import torch
import tqdm
import matplotlib.pyplot as plt

from utils import *

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")


def parse_args():
    parser = argparse.ArgumentParser(description="Image Rendering and Optimization")

    parser.add_argument(
        "input_image",
        type=pathlib.Path,
        help="Path to the input image",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        default="output",
        help="Directory to save output images",
    )

    parser.add_argument(
        "--source_dir",
        type=pathlib.Path,
        default="images",
        help="Directory containing source images",
    )

    parser.add_argument(
        "--saliency_map",
        type=pathlib.Path,
        default=None,
        help="Path to the saliency map image",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of optimization steps",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for optimization",
    )

    parser.add_argument(
        "--inner_steps",
        type=int,
        default=500,
        help="Number of inner optimization steps per source image",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    input_image = load_image(args.input_image).to(device)
    if args.saliency_map is None:
        saliency_map = torch.ones((1, 1, input_image.shape[2], input_image.shape[3]), device=device)
    else:
        saliency_map = load_image(args.saliency_map, mode="L").to(device)# + 0.5
    saliency_map = torch.clamp(saliency_map, 0.0, 1.0)
    input_image = torch.nan_to_num(input_image, nan=0.0)


    source_path = args.source_dir
    destination_path = args.output_dir
    destination_path.mkdir(parents=True, exist_ok=True)
    source_images = []
    for img_file in source_path.glob("*.png"):
        source_images.append(load_image(str(img_file), mode="LA").to(device))


    dest_image = torch.ones_like(input_image).to(device)

    print("dest_image shape:", dest_image.shape)
    print("source_images count:", len(source_images))
    print("source_image shape:", source_images[0].shape)
    print("source_image range:", source_images[0].min(), source_images[0].max())

    bar = tqdm.tqdm(range(args.steps), desc="Overall Progress", unit="step")
    lr = args.lr
    for i in bar:
        idx = torch.randint(0, len(source_images), (1,)).item()
        source_image = source_images[idx]
        ratio = source_image.shape[2] / source_image.shape[3]
        angle = torch.randn(1, device=device) * 0.5
        scale = (10 * torch.rand(1, device=device)).clamp(1.0, 10.0)
        translation = -0.5 + torch.rand(2, device=device)
        color = torch.rand(4, device=device)
        angle.requires_grad = True
        scale.requires_grad = True
        translation.requires_grad = True
        color.requires_grad = True
        optim = torch.optim.Adam([angle, scale, translation, color], lr=lr)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=75, gamma=0.5)
        optim.zero_grad()

        for step in range(args.inner_steps):
            optim.zero_grad()
            scale_clamped = torch.clamp(scale, 1.0, 10.0)
            theta = get_theta(angle, scale_clamped, translation, ratio=ratio)
            rendered_image = render_image(
                dest_image, source_image, theta, color
            )
            loss_mse = torch.nn.functional.mse_loss(rendered_image, input_image)
            loss1 = mse_weighted(rendered_image, input_image, saliency_map)

            loss = loss1
            loss.backward()
            optim.step()

            bar.set_postfix({"loss": f"{loss.item():.6f}", "MSE": f"{loss_mse.item():.6f}"})

        dest_image = render_image(dest_image, source_image, theta, color)
        dest_image = dest_image.detach()
        if i % 10 == 0:
            save_image(
                        dest_image.clone().cpu(),
                        str(destination_path / f"intermediate_{i:04d}.png"),
                    )

    output_file = destination_path / "rendered_image.png"
    save_image(dest_image.cpu(), str(output_file))


if __name__ == "__main__":

    main()
