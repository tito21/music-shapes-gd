import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image
import matplotlib.pyplot as plt


def get_theta(angle, scale, translation, ratio=1.0):
    angle = torch.fmod(angle + torch.pi, 2 * torch.pi) - torch.pi  # Wrap angle to [-pi, pi]
    s = torch.sin(angle)
    c = torch.cos(angle)
    # translation = torch.clamp(translation, -1.0, 1.0)
    theta = torch.stack([
        torch.stack([scale * ratio * c, -scale * ratio * s, scale * ratio * translation[0:1]]),
        torch.stack([scale * s,  scale * c, scale * translation[1:2]]),
    ])
    theta = theta.to(angle.device)
    # theta = torch.clamp(theta, -1.0, 1.0)
    return theta.unsqueeze(0).squeeze(-1)  # Add batch dimension


def gaussian_kernel(size=5, sigma=1.0):
    """Generates a 2D Gaussian kernel."""
    ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax)
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / torch.sum(kernel)
    return kernel


def blur_image(image, kernel_size=5, sigma=1.0):
    """Applies Gaussian blur to the input image."""
    kernel = gaussian_kernel(kernel_size, sigma).to(image.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, kH, kW)
    channels = image.shape[1]
    kernel = kernel.repeat(channels, 1, 1, 1)  # Shape: (C, 1, kH, kW)
    padding = kernel_size // 2
    blurred = F.conv2d(image, kernel, padding=padding, groups=channels)
    return blurred


def render_image(dest, source, transform_matrix, color):
    grid = F.affine_grid(transform_matrix, dest.size(), align_corners=False)
    # rendered = blur_image(F.grid_sample(source, grid, align_corners=False, mode='bilinear', padding_mode='zeros'), kernel_size=5, sigma=1.0)
    rendered = F.grid_sample(source, grid, align_corners=False, mode='bilinear', padding_mode='zeros')
    rendered = torch.nan_to_num(rendered, nan=0.0)
    rendered_alpha = torch.clamp(color[3] * rendered[:, 1, :, :], 1e-16, 1)

    # Alpha blending
    final_image = torch.lerp(dest[:, :3, :, :], color[0:3].view(1, 3, 1, 1), rendered_alpha)
    final_image = torch.nan_to_num(final_image, nan=0.0)
    return torch.clamp(final_image, 0, 1)


def load_image(path, mode="RGB"):
    img = Image.open(path).convert(mode)
    transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    return transform(img).unsqueeze(0)  # Add batch dimension


def save_image(tensor, path):
    transform = T.ToPILImage()
    img = transform(tensor.squeeze(0))  # Remove batch dimension
    img.save(path)



def mse_weighted(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)


def outside_mask(input, mask):
    mask = mask[:, 0, :, :]  # BHW
    # input2 = input.clone()
    # input2 = input2 * mask
    # # plt.imshow((mask).squeeze(0).detach().cpu())
    # # plt.show()
    # plt.imshow(input2.squeeze(0).detach().cpu())
    # plt.colorbar()
    # plt.show()
    return torch.mean(input * mask.float())