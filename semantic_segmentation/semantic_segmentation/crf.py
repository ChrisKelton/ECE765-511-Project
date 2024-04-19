__all__ = ["CrfRnn", "CrfRnnConfig"]

from typing import Any

import cv2
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime

from utils.torch_model import print_number_of_params


class CustomFilterLayer(nn.Module):
    def __init__(self, channels: int, kernel_size: int, sigma: int):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        # self.padding_amount = ceil(kernel_size / 2)
        self.padding_amount = kernel_size // 2
        # pad images with reflection to account for extrapolating pixels at the borders
        # do DepthWise Convolution to apply the gaussian kernels independently per channel
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(self.padding_amount),
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=0, bias=False, groups=channels),
        )

        # not trying to learn these Filters, they are already getting defined weights from scipy or cv2
        self.requires_grad_(requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def _weights_init(self):
        raise RuntimeError(f"_weights_init() function not defined in inherited class!")


class GaussianFilterLayer(CustomFilterLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weights_init()

    def _weights_init(self):
        arr_ = np.zeros((self.kernel_size, self.kernel_size))
        arr_[self.padding_amount, self.padding_amount] = 1
        k = scipy.ndimage.gaussian_filter(arr_, sigma=self.sigma)
        for name, weight in self.named_parameters():
            weight.data.copy_(torch.from_numpy(k))


class BilateralFilterLayer(CustomFilterLayer):
    def __init__(self, channels: int, kernel_size: int, sigma_space: int, sigma_color: int):
        super().__init__(channels, kernel_size, sigma_space)
        # I believe it is similar to the sigma for a normal Gaussian.
        self.sigma_space = sigma_space

        # sigma_color, sigma_r in literature, indicates the deviation away from the color space affecting the weights
        # of differently colored pixels. E.g., if sigma_color = 4, then pixels values > 3*sigma_color away from the
        # pixel being convolved with the bilateral filter will have no impact on the current pixel. If we set
        # sigma_color = (255 / 3), then every pixel will affect every other pixel to some extent, regardless of their
        # pixel value.
        self.sigma_color = sigma_color

        self._weights_init()

    def _weights_init(self):
        arr_ = np.zeros((self.kernel_size, self.kernel_size)).astype(np.float32)
        arr_[self.padding_amount, self.padding_amount] = 1.
        k = cv2.bilateralFilter(arr_, self.kernel_size, self.sigma_color, self.sigma_space)
        for name, weight in self.named_parameters():
            weight.data.copy_(torch.from_numpy(k))


class ConditionalRandomFieldBlock(nn.Module):
    def __init__(
        self,
        n_classes: int,
        channels: int = 3,
        gaussian_filter_kernel_sizes: tuple[int, ...] = (3, 7),
        gaussian_filter_sigmas: tuple[int, ...] = (3, 3),
        bilateral_filter_kernel_sizes: tuple[int, ...] = (5, 9),
        bilateral_filter_sigmas: tuple[tuple[int, float], ...] = ((3, 10.), (3, 30.)),
    ):
        super().__init__()
        self.n_classes = n_classes
        self.channels = channels
        self.gaussian_filter_sigmas = gaussian_filter_sigmas
        self.gaussian_filter_kernel_sizes = gaussian_filter_kernel_sizes

        # ******************** Message Passing ********************
        self.gaussian_filters: nn.ModuleList[GaussianFilterLayer] = nn.ModuleList()
        for kernel_size, sigma in zip(gaussian_filter_sigmas, gaussian_filter_kernel_sizes):
            self.gaussian_filters.append(
                GaussianFilterLayer(
                    channels=channels,
                    kernel_size=kernel_size,
                    sigma=sigma,
                ).requires_grad_(requires_grad=False)
            )

        self.bilateral_filters: nn.ModuleList[BilateralFilterLayer] = nn.ModuleList()
        for kernel_size, sigmas in zip(bilateral_filter_kernel_sizes, bilateral_filter_sigmas):
            self.bilateral_filters.append(
                BilateralFilterLayer(
                    channels=channels,
                    kernel_size=kernel_size,
                    sigma_space=sigmas[0],
                    sigma_color=sigmas[1],
                ).requires_grad_(requires_grad=False)
            )

        self.message_passing_channels_size = channels * (len(self.gaussian_filters) + len(self.bilateral_filters))

        # ******************** Weighting Filter Outputs ********************
        # self.conv_weight_filter_outputs = nn.Conv2d(
        #     self.message_passing_channels_size,
        #     1,
        #     kernel_size=1,
        # )
        self.conv_weight_filter_outputs = nn.Conv3d(
            self.message_passing_channels_size,
            1,
            kernel_size=1,
        )
        self.relu = nn.ReLU()

        # ******************** Compatibility Transform ********************
        self.conv_compatibility = nn.Conv2d(
            n_classes,
            n_classes,
            kernel_size=1,
        )

    @classmethod
    def initialize_simpler_distribution(cls, unary_potentials: torch.Tensor) -> torch.Tensor:
        """

        Args:
            unary_potentials: shape: [Batch size, Num Classes, Height, Width]

        Returns:

        """
        # from https://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf p.4 Section 4:
        #
        #   Q_i(l) <- (1 / Z_i)exp(U_i(l))
        #       U_i(l) denotes the negative of the unary energy, i.e., U_i(l) = -psi_u(X_i = l), where
        #       i = pixel location & l = label
        #
        #       Z_i = Sum_{l}exp(U_i(l))
        #
        # This is equivalent to applying a softmax function over the unary potentials U across all the labels at
        # each pixel.
        return F.softmax(unary_potentials, dim=1)

    def message_passing(self, x: torch.Tensor) -> torch.Tensor:
        # Excerpt from https://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf p.4 Section 4.2
        #
        #   In the dense CRF formulation, message passing is implemented by applying M Gaussian filters on Q values.
        #   Gaussian filter coefficients are derived based on image features such as the pixel locations and RGB values,
        #   which reflect how strongly a pixel is related to other pixels.
        #   ...
        #   Following [29], we use two Gaussian kernels, a spatial kernel and a bilateral kernel.
        x_filtered: list[torch.Tensor] = []
        for gaussian_filter in self.gaussian_filters:
            x_filtered.append(gaussian_filter(x))
        for bilateral_filter in self.bilateral_filters:
            x_filtered.append(bilateral_filter(x))
        return torch.cat(x_filtered, dim=1)

    def forward(self, x: torch.Tensor, unary_potentials: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: input images, shape: [Batch size, Channels, Height, Width]
            unary_potentials: comes from the output of the backbone FCN, shape: [Batch size, Num Classes, Height, Width]

        Returns:

        """
        # Q_init shape: [Batch size, Num Classes, Height, Width]
        # H_1
        Q_init = self.initialize_simpler_distribution(unary_potentials)

        # Q_weighted shape: [Batch size, Num Classes, Height, Width]
        # E.g., if we have 2 Gaussian filters & 2 Bilateral filters, then n_filters = 4
        x_message_passed = self.message_passing(x)

        # iterate over message passing and weighted filter outputs portions of algorithm
        # want to look at each message passed per label's unary potential
        Q_message_passed_per_layer = x_message_passed[:, :, None, :, :] * Q_init[:, None, :, :, :]
        Q_weighted_filter = self.conv_weight_filter_outputs(Q_message_passed_per_layer).reshape(Q_init.shape)
        Q_weighted = self.relu(Q_weighted_filter)

        # Outputs from Q_weighted are shared between the labels to a varied extent, depending on the compatibility
        # between these labels. Compatibility between the two labels l & l' is parameterized by the label compatibility
        # function u(l, l'). The Potts model, given by u(l, l') = [l != l'], where [.] is the Iverson bracket, assigns
        # a fixed penalty if different labels are assigned to pixels with similar properties. A limitation of this
        # model is that it assigns the same penalty for all different pairs of labels. Therefore, the paper concluded
        # to allow this compatibility function to be learned by a 1 x 1 convolutional layer where the number of inputs
        # and outputs is both the Num Classes.
        Q_compatibility = self.conv_compatibility(Q_weighted)

        # Iterative portion where we refine the unary_potentials and return them. This is simply a block in the
        # overall model architecture of the CRF-RNN model, so the output here will be the unary_potentials input in the
        # next iteration.
        # H_2
        Q_w_unary_pot = unary_potentials - Q_compatibility

        return Q_w_unary_pot


class ConditionalRandomFieldBlockTimed(ConditionalRandomFieldBlock):

    @classmethod
    def timer(cls, beg_time, end_time, name):
        time_delta = datetime.timedelta(microseconds=int(round(end_time - beg_time)))
        print(f"{name}: {time_delta}us")

    def forward(self, x: torch.Tensor, unary_potentials: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: input images, shape: [Batch size, Channels, Height, Width]
            unary_potentials: comes from the output of the backbone FCN, shape: [Batch size, Num Classes, Height, Width]

        Returns:

        """
        # Q_init shape: [Batch size, Num Classes, Height, Width]
        # H_1
        time_beg = time.perf_counter_ns()
        Q_init = self.initialize_simpler_distribution(unary_potentials)
        time_end = time.perf_counter_ns()
        self.timer(beg_time=time_beg * 1000, end_time=time_end * 1000, name="initialize_simpler_distribution")

        # Q_weighted shape: [Batch size, Num Classes, Height, Width]
        # E.g., if we have 2 Gaussian filters & 2 Bilateral filters, then n_filters = 4
        time_beg = time.perf_counter_ns()
        x_message_passed = self.message_passing(x)
        time_end = time.perf_counter_ns()
        self.timer(beg_time=time_beg * 1000, end_time=time_end * 1000, name="message_passing")

        # iterate over message passing and weighted filter outputs portions of algorithm
        # want to look at each message passed per label's unary potential
        time_beg = time.perf_counter_ns()
        Q_message_passed_per_layer = x_message_passed[:, :, None, :, :] * Q_init[:, None, :, :, :]
        Q_weighted_filter = self.conv_weight_filter_outputs(Q_message_passed_per_layer).reshape(Q_init.shape)
        Q_weighted = self.relu(Q_weighted_filter)
        time_end = time.perf_counter_ns()
        self.timer(beg_time=time_beg * 1000, end_time=time_end * 1000, name="weighted message passing")

        # Outputs from Q_weighted are shared between the labels to a varied extent, depending on the compatibility
        # between these labels. Compatibility between the two labels l & l' is parameterized by the label compatibility
        # function u(l, l'). The Potts model, given by u(l, l') = [l != l'], where [.] is the Iverson bracket, assigns
        # a fixed penalty if different labels are assigned to pixels with similar properties. A limitation of this
        # model is that it assigns the same penalty for all different pairs of labels. Therefore, the paper concluded
        # to allow this compatibility function to be learned by a 1 x 1 convolutional layer where the number of inputs
        # and outputs is both the Num Classes.
        time_beg = time.perf_counter_ns()
        Q_compatibility = self.conv_compatibility(Q_weighted)
        time_end = time.perf_counter_ns()
        self.timer(beg_time=time_beg * 1000, end_time=time_end * 1000, name="compatibility")

        # Iterative portion where we refine the unary_potentials and return them. This is simply a block in the
        # overall model architecture of the CRF-RNN model, so the output here will be the unary_potentials input in the
        # next iteration.
        # H_2
        time_beg = time.perf_counter_ns()
        Q_w_unary_pot = unary_potentials - Q_compatibility
        time_end = time.perf_counter_ns()
        self.timer(beg_time=time_beg * 1000, end_time=time_end * 1000, name="refine potentials")

        return Q_w_unary_pot



CrfRnnConfig: dict[str, Any] = {
    "gaussian_filter_kernel_sizes": ((3, 7), (3, 7), (3, 7), (3, 7), (3, 7)),
    "gaussian_filter_sigmas": ((3, 3), (3, 3), (3, 3), (3, 3), (3, 3)),
    "bilateral_filter_kernel_sizes": ((5, 9), (5, 9), (5, 9), (5, 9), (5, 9)),
    "bilateral_filter_sigmas": (((3, 10.), (3, 30.)), ((3, 10.), (3, 30.)), ((3, 10.), (3, 30.)), ((3, 10.), (3, 30.)), ((3, 10.), (3, 30.))),
}


class CrfRnn(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_crf_blocks: int = 5,  # it is not advised to increase this value
        channels: int = 3,
        gaussian_filter_kernel_sizes: tuple[tuple[int, ...], ...] = CrfRnnConfig["gaussian_filter_kernel_sizes"],
        gaussian_filter_sigmas: tuple[tuple[int, ...], ...] = CrfRnnConfig["gaussian_filter_sigmas"],
        bilateral_filter_kernel_sizes: tuple[tuple[int, ...], ...] = CrfRnnConfig["bilateral_filter_kernel_sizes"],
        bilateral_filter_sigmas: tuple[tuple[tuple[int, float], ...], ...] = CrfRnnConfig["bilateral_filter_sigmas"],
        return_logits: bool = True,
    ):
        super().__init__()
        self.n_classes = n_classes,
        self.n_crf_blocks = n_crf_blocks
        self.channels = channels
        self.gaussian_filter_kernel_sizes = gaussian_filter_kernel_sizes
        self.gaussian_filter_sigmas = gaussian_filter_sigmas
        self.bilateral_filter_kernel_sizes = bilateral_filter_kernel_sizes
        self.bilateral_filter_sigmas = bilateral_filter_sigmas
        self.return_logits = return_logits

        assert len(self.gaussian_filter_kernel_sizes) == n_crf_blocks, f"Incorrect number of 'gaussian_filter_kernel_sizes' for '{n_crf_blocks}' CRF Blocks"
        assert len(self.gaussian_filter_sigmas) == n_crf_blocks, f"Incorrect number of 'gaussian_filter_sigmas' for '{n_crf_blocks}' CRF Blocks"
        assert len(self.bilateral_filter_kernel_sizes) == n_crf_blocks, f"Incorrect number of 'bilateral_filter_kernel_sizes' for '{n_crf_blocks}' CRF Blocks"
        assert len(self.bilateral_filter_sigmas) == n_crf_blocks, f"Incorrect number of 'bilateral_filter_sigmas' for '{n_crf_blocks}' CRF Blocks"

        self.crf_blocks: nn.ModuleList[ConditionalRandomFieldBlock] = nn.ModuleList()
        for block_idx in range(self.n_crf_blocks):
            self.crf_blocks.append(
                ConditionalRandomFieldBlock(
                    n_classes=n_classes,
                    channels=channels,
                    gaussian_filter_kernel_sizes=self.gaussian_filter_kernel_sizes[block_idx],
                    gaussian_filter_sigmas=self.gaussian_filter_sigmas[block_idx],
                    bilateral_filter_kernel_sizes=self.bilateral_filter_kernel_sizes[block_idx],
                    bilateral_filter_sigmas=self.bilateral_filter_sigmas[block_idx],
                )
            )

    def forward(self, x: torch.Tensor, unary_potentials: torch.Tensor) -> torch.Tensor:
        for crf_block in self.crf_blocks:
            unary_potentials = crf_block(x, unary_potentials)

        if self.return_logits:
            return unary_potentials
        return F.softmax(unary_potentials, dim=1)


class CrfRnnTimed(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_crf_blocks: int = 5,  # it is not advised to increase this value
        channels: int = 3,
        gaussian_filter_kernel_sizes: tuple[tuple[int, ...], ...] = CrfRnnConfig["gaussian_filter_kernel_sizes"],
        gaussian_filter_sigmas: tuple[tuple[int, ...], ...] = CrfRnnConfig["gaussian_filter_sigmas"],
        bilateral_filter_kernel_sizes: tuple[tuple[int, ...], ...] = CrfRnnConfig["bilateral_filter_kernel_sizes"],
        bilateral_filter_sigmas: tuple[tuple[tuple[int, float], ...], ...] = CrfRnnConfig["bilateral_filter_sigmas"],
        return_logits: bool = True,
    ):
        super().__init__()
        self.n_classes = n_classes,
        self.n_crf_blocks = n_crf_blocks
        self.channels = channels
        self.gaussian_filter_kernel_sizes = gaussian_filter_kernel_sizes
        self.gaussian_filter_sigmas = gaussian_filter_sigmas
        self.bilateral_filter_kernel_sizes = bilateral_filter_kernel_sizes
        self.bilateral_filter_sigmas = bilateral_filter_sigmas
        self.return_logits = return_logits

        assert len(self.gaussian_filter_kernel_sizes) == n_crf_blocks, f"Incorrect number of 'gaussian_filter_kernel_sizes' for '{n_crf_blocks}' CRF Blocks"
        assert len(self.gaussian_filter_sigmas) == n_crf_blocks, f"Incorrect number of 'gaussian_filter_sigmas' for '{n_crf_blocks}' CRF Blocks"
        assert len(self.bilateral_filter_kernel_sizes) == n_crf_blocks, f"Incorrect number of 'bilateral_filter_kernel_sizes' for '{n_crf_blocks}' CRF Blocks"
        assert len(self.bilateral_filter_sigmas) == n_crf_blocks, f"Incorrect number of 'bilateral_filter_sigmas' for '{n_crf_blocks}' CRF Blocks"

        self.crf_blocks: nn.ModuleList[ConditionalRandomFieldBlockTimed] = nn.ModuleList()
        for block_idx in range(self.n_crf_blocks):
            self.crf_blocks.append(
                ConditionalRandomFieldBlockTimed(
                    n_classes=n_classes,
                    channels=channels,
                    gaussian_filter_kernel_sizes=self.gaussian_filter_kernel_sizes[block_idx],
                    gaussian_filter_sigmas=self.gaussian_filter_sigmas[block_idx],
                    bilateral_filter_kernel_sizes=self.bilateral_filter_kernel_sizes[block_idx],
                    bilateral_filter_sigmas=self.bilateral_filter_sigmas[block_idx],
                )
            )

    def forward(self, x: torch.Tensor, unary_potentials: torch.Tensor) -> torch.Tensor:
        time_beg_ = time.perf_counter_ns()
        for idx, crf_block in enumerate(self.crf_blocks):
            time_beg = time.perf_counter_ns()
            unary_potentials = crf_block(x, unary_potentials)
            time_end = time.perf_counter_ns()
            ConditionalRandomFieldBlockTimed.timer(beg_time=time_beg * 1000, end_time=time_end * 1000, name=f"crf-block {idx}")
        time_end_ = time.perf_counter_ns()
        ConditionalRandomFieldBlockTimed.timer(beg_time=time_beg_ * 1000, end_time=time_end_ * 1000, name="All crf-blocks")

        if self.return_logits:
            return unary_potentials
        return F.softmax(unary_potentials, dim=1)


def main():
    from backbone import FcnResNet50BackBone
    from data_prep.cityscape_dataset import get_dataset
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    fcn_resnet50 = FcnResNet50BackBone().to(device)
    datasets = get_dataset(batch_size=1)
    imgs, labels = next(iter(datasets.train))
    imgs = imgs.to(device)
    labels = labels.to(device)
    unary_potentials = fcn_resnet50(imgs)

    crf_net = CrfRnnTimed(n_classes=20).to(device)
    print_number_of_params(crf_net)
    out = crf_net(imgs, unary_potentials)
    print(imgs.shape, out.shape)


if __name__ == '__main__':
    main()
