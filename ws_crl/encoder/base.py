# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

""" Encoders and decoders """

from torch import nn


class IntractableError(Exception):
    """Exception thrown when quantities are fundamentally intractable"""

    pass


class Encoder(nn.Module):
    """
    Base class for encoders and decoders.
    """

    def __init__(self, input_features=2, output_features=2):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

    def forward(self, inputs, deterministic=False):
        """
        Forward transformation.

        In an encoder: takes as input the observed data x and returns the latent representation z.
        In a decoder: takes as input the latent representation x and returns the reconstructed data
        x.

        Parameters:
        -----------
        inputs : torch.Tensor with shape (batchsize, input_features), dtype torch.float
            Data to be encoded or decoded

        Returns:
        --------
        outputs : torch.Tensor with shape (batchsize, output_features), dtype torch.float
            Encoded or decoded version of the data
        additional_info : None, torch.Tensor, or tuple
            Additional information that depends on the kind of encoder. For flow-style transforms,
            this is the log of the Jacobian determinant. For VAE encoders, this is the log
            likelihood or log posterior. Otherwise, None.
        """

        raise NotImplementedError

    def inverse(self, inputs, deterministic=False):
        """
        Inverse transformation, if tractable (otherwise raises an exception).

        In a decoder: takes as input the observed data x and returns the latent representation z.
        In an encoder: takes as input the latent representation z and returns the reconstructed data
        x.

        Parameters:
        -----------
        inputs : torch.Tensor with shape (batchsize, input_features), dtype torch.float
            Data to be encoded or decoded

        Returns:
        --------
        outputs : torch.Tensor with shape (batchsize, output_features), dtype torch.float
            Encoded or decoded version of the data
        additional_info: None or torch.Tensor
            Additional information that depends on the kind of encoder. For flow-style transforms,
            this is the log of the Jacobian determinant. Otherwise, None.
        """

        raise IntractableError()


class Inverse(Encoder):
    """
    Wrapper class that inverts the forward and inverse direction, e.g. turning an encoder into a
    decoder.
    """

    def __init__(self, base_model):
        super().__init__(
            input_features=base_model.output_features, output_features=base_model.input_features
        )
        self.base_model = base_model

    def forward(self, inputs, deterministic=False):
        """
        Forward transformation.

        In an encoder: takes as input the observed data x and returns the latent representation z.
        In a decoder: takes as input the latent representation x and returns the reconstructed data
        x.

        Parameters:
        -----------
        inputs : torch.Tensor with shape (batchsize, input_features), dtype torch.float
            Data to be encoded or decoded

        Returns:
        --------
        outputs : torch.Tensor with shape (batchsize, output_features), dtype torch.float
            Encoded or decoded version of the data
        additional_info : None, torch.Tensor, or tuple
            Additional information that depends on the kind of encoder. For flow-style transforms,
            this is the log of the Jacobian determinant. For VAE encoders, this is the log
            likelihood or log posterior. Otherwise, None.
        """
        return self.base_model.inverse(inputs)

    def inverse(self, outputs, deterministic=False):
        """
        Inverse transformation, if tractable (otherwise raises an exception).

        In a decoder: takes as input the observed data x and returns the latent representation z.
        In an encoder: takes as input the latent representation z and returns the reconstructed data
        x.

        Parameters:
        -----------
        inputs : torch.Tensor with shape (batchsize, input_features), dtype torch.float
            Data to be encoded or decoded

        Returns:
        --------
        outputs : torch.Tensor with shape (batchsize, output_features), dtype torch.float
            Encoded or decoded version of the data
        additional_info: None or torch.Tensor
            Additional information that depends on the kind of encoder. For flow-style transforms,
            this is the log of the Jacobian determinant. Otherwise, None.
        """
        return self.base_model(outputs)
