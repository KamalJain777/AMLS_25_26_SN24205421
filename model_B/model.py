"""Deep learning model architecture (ResNet-style) for Model B."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for ResNet-style architecture."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Initialize residual block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for convolution
        """
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ModelBNet(nn.Module):
    """
    Small ResNet-style network for BreastMNIST.

    Architecture suitable for 28x28 grayscale images.
    """

    def __init__(
        self,
        num_classes: int = 2,
        base_channels: int = 16,
        num_blocks: int = 2,
        dropout_rate: float = 0.5,
    ):
        """
        Initialize Model B network.

        Args:
            num_classes: Number of output classes (2 for binary classification)
            base_channels: Base number of channels (controls model capacity)
            num_blocks: Number of residual blocks per stage
            dropout_rate: Dropout rate for regularization
        """
        super(ModelBNet, self).__init__()

        self.base_channels = base_channels
        self.num_blocks = num_blocks

        # Initial convolution
        self.conv1 = nn.Conv2d(
            1, base_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_channels)

        # Residual layers
        self.layer1 = self._make_layer(
            base_channels, base_channels, num_blocks, stride=1
        )
        self.layer2 = self._make_layer(
            base_channels, base_channels * 2, num_blocks, stride=2
        )
        self.layer3 = self._make_layer(
            base_channels * 2, base_channels * 4, num_blocks, stride=2
        )

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(base_channels * 4, base_channels * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(base_channels * 2, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """
        Create a layer with residual blocks.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            num_blocks: Number of blocks
            stride: Stride for first block

        Returns:
            Sequential container of blocks
        """
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # Initial conv + BN + ReLU
        out = F.relu(self.bn1(self.conv1(x)))

        # Residual layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Global average pooling
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        # Fully connected layers
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "base_channels": self.base_channels,
                "num_blocks": self.num_blocks,
            },
            filepath,
        )

    @classmethod
    def load_checkpoint(cls, filepath: str, device: str = "cpu"):
        """
        Load model from checkpoint.

        Args:
            filepath: Path to checkpoint file
            device: Device to load model on

        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            base_channels=checkpoint["base_channels"],
            num_blocks=checkpoint["num_blocks"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model
