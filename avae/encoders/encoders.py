import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn # add this as required for DecoderC - could find a way to add back to models.py

from avae.encoders.base import AbstractEncoder
from avae.models import dims_after_pooling, set_layer_dim, dims_after_pooling_1D  # added dims_after_pooling_1D


class Encoder(AbstractEncoder):
    """Affinity encoder. Includes optional pose component in the architecture.

    Parameters
    ----------
    input_size: tuple (X, Y) or tuple (X, Y, Z)
        Tuple representing the size of the data for each image
        dimension X, Y and Z.
    latent_dims: int
        Number of bottleneck latent dimensions.
    pose_dims : int
        Number of bottleneck pose dimensions.
    capacity : int (None)
        The capacity of the network - initial number of nodes doubled at each
        depth.
    depth : int (None)
        The depth of the network - number of downsampling layers.
    filters : list : [int] (None)
        List of filter sizes, where len(filters) becomes network depth.
    bnorm : bool (True)
        If True, turns BatchNormalisation on.
    """

    def __init__(
        self,
        input_size: tuple,
        capacity: Optional[int] = None,
        depth: int = 4,
        latent_dims: int = 8,
        pose_dims: int = 0,
        filters: Optional[list[int]] = None,
        bnorm: bool = True,
    ):

        super(Encoder, self).__init__()
        self.filters = []
        if capacity is None and filters is None:
            raise RuntimeError(
                "Pass either capacity or filters when definining avae.Encoder."
            )
        elif filters is not None and len(filters) != 0:
            if 0 in filters:
                raise RuntimeError("Filter list cannot contain zeros.")
            self.filters = filters
            if depth is not None:
                logging.warning(
                    "You've passed 'filters' parameter as well as 'depth'. Filters take"
                    " priority so 'depth' and 'capacity' will be disregarded."
                )
        elif capacity is not None:
            if depth is None:
                raise RuntimeError(
                    "When passing initial 'capacity' parameter in avae.Encoder,"
                    " provide 'depth' parameter too."
                )
            self.filters = [capacity * 2**x for x in range(depth)]
        else:
            raise RuntimeError(
                "You must provide either capacity or filters when definity ave.Encoder."
            )

        assert all(
            [int(x) == x for x in np.array(input_size) / (2**depth)]
        ), (
            "Input size not compatible with --depth. Input must be divisible "
            "by {}.".format(2**depth)
        )

        bottom_dim = tuple(
            [int(i / (2 ** len(self.filters))) for i in input_size]
        )
        self.bnorm = bnorm
        self.pose = not (pose_dims == 0)

        # define layer dimensions
        CONV, TCONV, BNORM = set_layer_dim(len(input_size))

        # iteratively define convolution and batch normalisation layers
        self.conv_enc = torch.nn.ModuleList()
        if self.bnorm:
            self.norm_enc = torch.nn.ModuleList()

        for d in range(len(self.filters)):
            self.conv_enc.append(
                CONV(
                    in_channels=(self.filters[d - 1] if d != 0 else 1),
                    out_channels=self.filters[d],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            if self.bnorm:
                self.norm_enc.append(BNORM(self.filters[d]))

        # define fully connected layers
        ch = 1 if depth == 0 else self.filters[-1]  # allow for no conv layers
        self.fc_mu = torch.nn.Linear(
            in_features=ch * np.prod(bottom_dim),
            out_features=latent_dims,
        )
        self.fc_logvar = torch.nn.Linear(
            in_features=ch * np.prod(bottom_dim),
            out_features=latent_dims,
        )
        if self.pose:
            self.fc_pose = torch.nn.Linear(
                in_features=ch * np.prod(bottom_dim),
                out_features=pose_dims,
            )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Encoder forward pass.

        Parameters
        ----------
        x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of inputs, where N stands for the number of samples in
            the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.

        Returns
        -------
        x_mu : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent means, where N stands
            for the number of samples in the mini-batch
            and 'latent_dims' defines the number of latent dimensions.
        x_logvar : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent log of the variance,
            where N stands for the number of samples in
            the mini-batch and 'latent_dims' defines the number of latent
            dimensions.
        x_pose : torch.Tensor (N, pose_dims)
            Optional return if pose is True. Mini-batch of outputs
            representing pose capturing the within-class
            variance, where N stands for the number of samples in the
            mini-batch and 'pose_dims' defines the number of
            pose dimensions.
        """
        for d in range(len(self.filters)):
            if self.bnorm:
                x = self.norm_enc[d](
                    torch.nn.functional.relu(self.conv_enc[d](x))
                )
            else:
                x = torch.nn.functional.relu(self.conv_enc[d](x))
        x = x.view(x.size(0), -1)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        if self.pose:
            x_pose = self.fc_pose(x)
            return x_mu, x_logvar, x_pose
        else:
            return x_mu, x_logvar


class EncoderA(AbstractEncoder):
    def __init__(
        self,
        input_size: tuple, # input size is the variable dshape in train.py
        capacity: Optional[int] = None,
        depth: int = 4,
        latent_dims: int = 8,
        pose_dims: int = 0,
        bnorm: bool = True,
    ):
        super(EncoderA, self).__init__()
        self.pose = not (pose_dims == 0)
        self.bnorm = bnorm


        if len(input_size) == 1: # added a if else to specify the correct condition for a 1D input (assume fixed kernal size, stride and padding)
            assert int(dims_after_pooling_1D(int(np.array(input_size)[0]), depth)) == dims_after_pooling_1D(int(np.array(input_size)[0]), depth), "Input length not compatible with --depth, kernal, stide and padding"
        else:
            assert all(
                [int(x) == x for x in np.array(input_size) / (2**depth)]
            ), (
                "Input size not compatible with --depth. Input must be divisible "
                "by {}.".format(2**depth)
            )

        filters = [capacity * 2**x for x in range(depth)]       # capacity specified in the config file

        if len(input_size) == 1:                  # added if else - output shape different for 1D vs. 2D inputs. dims_after_pooling_1D in models.py
            unflat_shape = tuple([filters[-1],
                                 dims_after_pooling_1D(int(np.array(input_size)[0]), depth)])

        else:    
            unflat_shape = tuple(                                   
                [
                    filters[-1],
                ]
                + [dims_after_pooling(ax, depth) for ax in input_size]   # dims after pooling is function in models.py
            )
        flat_shape = np.prod(unflat_shape)                        # multiply out last conv. layer to get length of input vector for fully connected layer

        ndim = len(unflat_shape[1:])                              # number of dimensions of image, excluding the number of filters

        conv, _, BNORM = set_layer_dim(ndim)                  # set_layer_dim is a helpfer function that gets the nn.conv models of the right number of dimensions

        self.encoder = torch.nn.Sequential()

        input_channel = 1                                         # hard coded - what is the image is rgb? Is encoder A just for bw images?
        for d in range(len(filters)):                             # recursive adding of convolutional layers, based on the filter sizes given in filters
            self.encoder.append(                                  # note: kernal size, stride and padding set at standard values for image processing
                conv(
                    in_channels=input_channel,
                    out_channels=filters[d],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            if self.bnorm:
                self.encoder.append(BNORM(filters[d]))
            self.encoder.append(torch.nn.ReLU(True))
            input_channel = filters[d]                            # update the number of filters (input for next layer) to output from current layer

        self.encoder.append(torch.nn.Flatten())                   # flatten final convolutional layer
        self.mu = torch.nn.Linear(flat_shape, latent_dims)        # fully connected layer to reduce the dimensionality to the required number of latent dims
        self.log_var = torch.nn.Linear(flat_shape, latent_dims)   
        self.pose_fc = torch.nn.Linear(flat_shape, pose_dims)

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)                                     # mean calculated from the final fully connected layer - mean is a learned parameter
        log_var = self.log_var(encoded)                           # variance calculated in the same way as the mean, but will be different since a learned parameter
        pose = self.pose_fc(encoded)
        return mu, log_var, pose                                   


class EncoderB(AbstractEncoder):
    """Affinity encoder. Includes optional pose component in the architecture.

    Parameters
    ----------
    capacity : int
        The capacity of the network - initial number of nodes doubled at each
        depth.
    depth : int
        The depth of the network - number of downsampling layers.
    latent_dims: int
        Number of bottleneck latent dimensions.
    pose_dims : int
        Number of bottleneck pose dimensions.
    """

    def __init__(
        self,
        input_size: tuple,
        capacity: int = 8,
        depth: int = 4,
        latent_dims: int = 8,
        pose_dims: int = 0,
    ):
        super(EncoderB, self).__init__()

        assert all(
            [int(x) == x for x in np.array(input_size) / (2**depth)]
        ), (
            "Input size not compatible with --depth. Input must be divisible "
            "by {}.".format(2**depth)
        )
        CONV, _, BNORM = set_layer_dim(len(input_size))
        self.bottom_dim = tuple([int(i / (2**depth)) for i in input_size])

        self.depth = depth
        self.pose = not (pose_dims == 0)

        # iteratively define convolution and batch normalisation layers
        self.conv_enc = torch.nn.ModuleList()
        self.norm_enc = torch.nn.ModuleList()
        prev_sh = 1
        for d in range(depth):
            sh = capacity * (d + 1)
            self.conv_enc.append(
                CONV(
                    in_channels=prev_sh,
                    out_channels=sh,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )
            self.norm_enc.append(BNORM(sh))
            prev_sh = sh

        # define fully connected layers
        chf = 1 if depth == 0 else capacity * depth  # allow for no conv layers
        self.fc_mu = torch.nn.Linear(
            in_features=chf * np.prod(self.bottom_dim),
            out_features=latent_dims,
        )
        self.fc_logvar = torch.nn.Linear(
            in_features=chf * np.prod(self.bottom_dim),
            out_features=latent_dims,
        )
        if self.pose:
            self.fc_pose = torch.nn.Linear(
                in_features=chf * np.prod(self.bottom_dim),
                out_features=pose_dims,
            )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Encoder forward pass.

        Parameters
        ----------
        x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of inputs, where N stands for the number of samples in
            the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.

        Returns
        -------
        x_mu : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent means, where N stands
            for the number of samples in the mini-batch
            and 'latent_dims' defines the number of latent dimensions.
        x_logvar : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent log of the variance,
            where N stands for the number of samples in
            the mini-batch and 'latent_dims' defines the number of latent
            dimensions.
        x_pose : torch.Tensor (N, pose_dims)
            Optional return if pose is True. Mini-batch of outputs
            representing pose capturing the within-class
            variance, where N stands for the number of samples in the
            mini-batch and 'pose_dims' defines the number of
            pose dimensions.
        """
        for d in range(self.depth):
            x = self.norm_enc[d](torch.nn.functional.relu(self.conv_enc[d](x)))
        x = x.view(x.size(0), -1)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        if self.pose:
            x_pose = self.fc_pose(x)
            return x_mu, x_logvar, x_pose
        else:
            return x_mu, x_logvar


class EncoderC(AbstractEncoder):
    def __init__(
            self,
            input_size: tuple,
            capacity: Optional[int] = None,
            # reuse capacity (number of channels/ filters in CNN) as the number of nodes in each hidden layer
            depth: int = 2,  # reuse depth as the number of fully connected hidden layers
            latent_dims: int = 8,
            pose_dims: int = 0,
            bnorm: bool = True,
    ):
        super(EncoderC, self).__init__()
        self.pose = not (pose_dims == 0)
        self.bnorm = bnorm

        assert len(input_size) == 1, "Input must be a 1D vector for this Encoder / Decoder"

        # n_hidden = np.repeat(capacity, depth)  # define n_hidden as a function of capacity and depth
        cap_in = capacity
        n_hidden = np.zeros(depth, dtype=int)
        for i in range(depth):
            n_hidden[i] = cap_in
            cap_in = cap_in/2

        self.encoder = torch.nn.Sequential()

        input_features = int(np.array(input_size)[0])  # the dimensions of the input

        for d in range(len(n_hidden)):  # recursive adding of hidden layers, based on the n_hidden in each layer
            self.encoder.append(
                nn.Linear(
                    in_features=input_features,
                    out_features=n_hidden[d]
                )
            )
            self.encoder.append(torch.nn.ReLU(True))  # ToDo: allow for different activation functions -mtn
            input_features = n_hidden[d]  # update the number of nodes in the hidden layer to input to the next layer

        self.mu = torch.nn.Linear(in_features=n_hidden[-1], out_features=latent_dims)  # reduce the dimensionality to the required number of latent dims
        self.log_var = torch.nn.Linear(in_features=n_hidden[-1], out_features=latent_dims)
        self.pose_fc = torch.nn.Linear(in_features=n_hidden[-1], out_features=pose_dims)
    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded) # mean calculated from the final fully connected layer - mean is a learned parameter
        log_var = self.log_var(encoded) # variance calculated in the same way as the mean, but will be different since a learned parameter
        pose = self.pose_fc(encoded) #

        mu = torch.flatten(self.mu(encoded), start_dim = 1)  # REMOVE THE FLATTEN IF SIZE ISSUE FIXED ELSEWHERE...mean calculated from the final fully connected layer - mean is a learned parameter
        log_var = torch.flatten(self.log_var(encoded), start_dim = 1)  # REMOVE THE FLATTEN IF SIZE ISSUE FIXED ELSEWHERE... variance calculated in the same way as the mean, but will be different since a learned parameter
        pose = torch.flatten(self.pose_fc(encoded), start_dim = 1) # REMOVE THE FLATTEN IF SIZE ISSUE FIXED ELSEWHERE...
        if self.pose:
            return mu, log_var, pose
        else:
            return mu, log_var

