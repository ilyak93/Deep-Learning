import torch
import itertools as it
import torch.nn as nn


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes: int, channels: list,
                 pool_every: int, hidden_dims: list):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ReLU)*P -> MaxPool]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs should exist at the end, without a MaxPool after them.
        # ====== YOUR CODE: ======
        kernel_size, padding, stride, pool_padding, pool_kernel, pool_stride = 3, 1, 1, 0, 2, 2
        channel_list = [in_channels] + self.channels

        for i in range(len(channel_list) - 1):
            layers.append(nn.Conv2d(channel_list[i], channel_list[i + 1], kernel_size=kernel_size, stride=stride,
                                    padding=padding))
            layers.append(nn.ReLU())
            if (i + 1) % self.pool_every == 0:
                layers.append(nn.MaxPool2d(pool_kernel, stride=pool_stride, padding=pool_padding))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        #  (Linear -> ReLU)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        kernel_size, padding, stride, pool_padding, pool_kernel, pool_stride = 3, 1, 1, 0, 2, 2
        conv_out_dim = in_h

        for i in range(len(self.channels)):
            conv_out_dim = ((conv_out_dim - kernel_size + 2 * padding) / stride) + 1

            if (i + 1) % self.pool_every == 0:
                conv_out_dim = ((conv_out_dim - pool_kernel + 2 * pool_padding) / pool_stride) + 1

        conv_out_dim = int(conv_out_dim)

        conv_out_dim = (int(conv_out_dim) ** 2) * self.channels[-1]
        self.conv_out_dim = conv_out_dim

        hidden_list = [conv_out_dim] + self.hidden_dims

        for i in range(len(hidden_list) - 1):
            layers.append(nn.Linear(hidden_list[i], hidden_list[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_list[-1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x).view(x.shape[0], self.conv_out_dim)
        out = self.classifier(features)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list,
                 batchnorm=False, dropout=0.):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
        convolution in the block. The length determines the number of
        convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
        be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
        convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
        Zero means don't apply dropout.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order). Should end with a
        #    final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use. This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        channels_list = [in_channels] + channels

        main_path_layers = []
        for i in range(len(channels_list) - 1):
            padding = int((kernel_sizes[i] - 1) / 2)
            main_path_layers.append(
                nn.Conv2d(channels_list[i], channels_list[i + 1], kernel_size=kernel_sizes[i], padding=padding))

            if i < (len(channels_list) - 2):

                if dropout > 0:
                    main_path_layers.append(nn.Dropout2d(dropout))

                if batchnorm:
                    main_path_layers.append(nn.BatchNorm2d(channels_list[i + 1]))

                main_path_layers.append(nn.ReLU())

        self.main_path = nn.Sequential(*main_path_layers)

        if in_channels == channels_list[-1]:
            self.shortcut_path = nn.Sequential()
        else:
            self.shortcut_path = nn.Sequential(nn.Conv2d(in_channels, channels_list[-1], kernel_size=1, bias=False))
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ReLU)*P -> MaxPool]*(N/P)
        #   \------- SKIP ------/
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ReLUs (with a skip over them) should exist at the end,
        #    without a MaxPool after them.
        #  - Use your ResidualBlock implemetation.
        # ====== YOUR CODE: ======
        channel_list = [in_channels] + self.channels

        kernel_size, padding, stride, pool_padding, pool_kernel, pool_stride = 3, 1, 1, 0, 2, 2

        i = 0

        while i < len(channel_list) - 1:

            to = i + self.pool_every
            avoid_pool = False

            if to >= len(channel_list):
                avoid_pool = True
                to = len(channel_list) - 1

            layers.append(
                ResidualBlock(channel_list[i], channel_list[i + 1:to + 1], [kernel_size]*(to - i)))

            if not avoid_pool:
                layers.append(nn.MaxPool2d(pool_kernel, stride=pool_stride, padding=pool_padding))

            i += self.pool_every
        # ========================
        seq = nn.Sequential(*layers)
        return seq




class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        kernel_size, padding, stride, pool_padding, pool_kernel, pool_stride = 3, 1, 1, 2, 4, 2

        layers = []

        channel_list = [in_channels] + self.channels

        i = 0

        while i < len(channel_list) - 1:

            to = i + self.pool_every
            avoid_pool = False

            if to >= len(channel_list):
                avoid_pool = True
                to = len(channel_list) - 1

            layers.append(
                ResidualBlock(channel_list[i], channel_list[i + 1:to + 1], [kernel_size for _ in range(to - i)],
                              dropout=0.4, batchnorm=True))

            if not avoid_pool:
                layers.append(nn.MaxPool2d(pool_kernel, stride=pool_stride, padding=pool_padding))

            i += self.pool_every

        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        #  (Linear -> ReLU)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        kernel_size, padding, stride, pool_padding, pool_kernel, pool_stride = 3, 1, 1, 2, 4, 2
        conv_out_dim = in_h

        for i in range(len(self.channels)):
            conv_out_dim = ((conv_out_dim - kernel_size + 2 * padding) / stride) + 1

            if (i + 1) % self.pool_every == 0:
                conv_out_dim = ((conv_out_dim - pool_kernel + 2 * pool_padding) / pool_stride) + 1

        conv_out_dim = int(conv_out_dim)

        conv_out_dim = (int(conv_out_dim) ** 2) * self.channels[-1]
        self.conv_out_dim = conv_out_dim

        hidden_list = [conv_out_dim] + self.hidden_dims

        for i in range(len(hidden_list) - 1):
            layers.append(nn.Linear(hidden_list[i], hidden_list[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_list[-1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq
    # ========================