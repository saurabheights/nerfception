"""
This script is pytorch implementation of the experiment done in -
    Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
"""
import imageio
import numpy as np
import os
import skimage
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchviz import make_dot
from tqdm import tqdm


class MLP(nn.Module):

    def __init__(self, input_coordinates_dimensions=2, *args, **kwargs):
        super().__init__()
        self.input_coordinates_dimesnions = input_coordinates_dimensions
        self.Flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_coordinates_dimensions, 256),  # Take x, y location values as input per data sample.
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),  # Predict RGB values for given x, y values.
            nn.Sigmoid(),
        )

    # noinspection PyUnusedFunction
    def forward(self, x):
        flat_x = self.Flatten(x)
        return self.linear_relu_stack(flat_x)


# Fourier feature mapping
def input_mapping(x, B):
    if B is None:
        return x
    else:
        B = B.float()
        x_proj = (2. * torch.pi * x) @ B.T
        return torch.concatenate([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)


def psnr(mse_loss):
    return -10 * np.log10(mse_loss)  # Pytorch has no division by 2


# Train model with given hyperparameters and data
def train_model(B, train_data, test_data, learning_rate=1e-4, iters=2000, device='cpu'):
    # Outputs
    train_psnrs = []
    test_psnrs = []
    pred_imgs = []
    iter_counts = []

    # Prepare train data
    x_train = torch.from_numpy(train_data[0]).float()
    x_train = input_mapping(x_train, B)
    # x represents the pixel grid location to which we apply various transformations.
    x_train = x_train.reshape((-1, x_train.shape[-1])).to(device)  # Convert (M, N, 2/2*B.dim) shape to (M*N, 2/2*B.dim)
    y_train = torch.from_numpy(train_data[1]).float().to(device)
    y_train = y_train.reshape(-1, 3)  # Convert (M, N, 3) shape to (M*N, 3) shape

    # Prepare test data
    M, N, _ = test_data[0].shape  # get image dimensions for saving predictions as 2D-image (and not flattened ones)
    x_test = torch.from_numpy(test_data[0]).float()
    x_test = input_mapping(x_test, B)
    # x represents the pixel grid location to which we apply various transformations.
    x_test = x_test.reshape((-1, x_test.shape[-1])).to(device)  # Convert (M, N, 2/2*B.dim) shape to (M*N, 2/2*B.dim)
    y_test = torch.from_numpy(test_data[1]).float().to(device)
    y_test = y_test.reshape(-1, 3)  # Convert (M, N, 3) shape to (M*N, 3) shape

    model = MLP(input_coordinates_dimensions=x_train.shape[-1]).to(device)
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    val_per_train_iters = 25
    epochs = iters // val_per_train_iters
    current_iter = 0
    pbar = tqdm(range(epochs), desc='train epoch', position=1, leave=False)
    for _ in pbar:
        model.train(True)
        for _ in range(val_per_train_iters):
            model_pred = model(x_train)
            loss = mse_loss(model_pred, y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        current_iter += val_per_train_iters
        # noinspection PyUnboundLocalVariable
        loss = loss.item()

        model.train(False)
        with torch.no_grad():
            train_psnrs.append(psnr(loss))
            model_pred = model(x_test)
            test_loss = mse_loss(model_pred, y_test).cpu().item()
            test_psnrs.append(psnr(test_loss))
            pred_imgs.append(model_pred.detach().cpu().reshape((M, N, 3)).numpy())
            iter_counts.append(current_iter)

    return {
        'train_psnrs': train_psnrs,
        'test_psnrs': test_psnrs,
        'pred_imgs': np.stack(pred_imgs),
        'iter_counts': iter_counts,
    }


def imshow(img: np.ndarray):
    plt.imshow(img)
    plt.show()


def train_image():
    # Fix RNG Seeds to ensure all models are trained with same initial weights.
    model = MLP(input_coordinates_dimensions=2)  # Each data point location has 2 coordinates. X, Y
    x = torch.randn(256 * 256, 2)
    y = model(x)
    dot = make_dot(y.mean(), params=dict(model.named_parameters()))
    model_image_path = "../data/model_nerf_2dimage"
    dot.render(model_image_path, format="png")
    imshow(skimage.io.imread(model_image_path + ".png"))

    # Check available devices
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Load image and take a square crop from the center of size 515x512x3
    image_url = '../data/fox.jpg'
    img = skimage.io.imread(image_url)[..., :3] / 255.
    c = [img.shape[0] // 2, img.shape[1] // 2]
    r = 256
    img = img[c[0] - r:c[0] + r, c[1] - r:c[1] + r]

    # Create input pixel coordinates in the unit square
    coords = np.linspace(0, 1, img.shape[0], endpoint=False)
    x_test = np.stack(np.meshgrid(coords, coords), -1)

    # Training data is evenly sampled points from test data.
    test_data = [x_test, img]
    train_data = [x_test[::2, ::2], img[::2, ::2]]
    # Three different scales of Gaussian Fourier feature mappings
    mapping_size = 256
    B_gauss = torch.randn((mapping_size, 2))
    B_dict = {
        'none': None,  # No mapping inputs before training the model
        'basic': torch.eye(2),  # Basic mapping inputs before training the model, values still multiplied by 2*pi.
        'gauss_1': B_gauss * 1,  # different scales of Gaussian Fourier feature mappings
        'gauss_10': B_gauss * 10,  # different scales of Gaussian Fourier feature mappings
        'gauss_100': B_gauss * 100,  # different scales of Gaussian Fourier feature mappings
    }

    # This should take about 2-3 minutes
    outputs = {}
    for k in tqdm(B_dict, position=0, leave=False):
        outputs[k] = train_model(B_dict[k], train_data, test_data, learning_rate=1e-4, iters=10000, device=device)

    # Show final network outputs
    fig = plt.figure(figsize=(24, 4))
    N = len(outputs)
    for i, k in enumerate(outputs):
        plt.subplot(1, N + 1, i + 1)
        plt.imshow(outputs[k]['pred_imgs'][-1])
        plt.title(k)
    plt.subplot(1, N + 1, N + 1)
    plt.imshow(img)
    plt.title('GT')
    plt.show()
    fig.savefig('../data/learn_image_pred_imgs.png', dpi=fig.dpi)

    # Plot train/test error curves
    fig = plt.figure(figsize=(16, 6))
    plt.subplot(121)
    for i, k in enumerate(outputs):
        plt.plot(outputs[k]['iter_counts'], outputs[k]['train_psnrs'], label=k)
    plt.title('Train error')
    plt.ylabel('PSNR')
    plt.xlabel('Training iter')
    plt.legend()

    plt.subplot(122)
    for i, k in enumerate(outputs):
        plt.plot(outputs[k]['iter_counts'], outputs[k]['test_psnrs'], label=k)
    plt.title('Test error')
    plt.ylabel('PSNR')
    plt.xlabel('Training iter')
    plt.legend()
    plt.show()
    fig.savefig('../data/learn_image_test_train_curve.png', dpi=fig.dpi)

    # Save out video of model output as training goes on.
    all_preds = np.concatenate([outputs[n]['pred_imgs'] for n in outputs], axis=-2)
    data8 = (255 * np.clip(all_preds, 0, 1)).astype(np.uint8)
    f = os.path.join('../data/training_convergence.mp4')
    imageio.mimwrite(f, data8, fps=5)


if __name__ == '__main__':
    train_image()
