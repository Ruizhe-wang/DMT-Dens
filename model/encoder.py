import torch
import torch.nn as nn
# from torchvision.models import ResNet, BasicBlock

from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import ResNet

# Define a neural network module with Linear, BatchNorm, and LeakyReLU layers
class NN_FCBNRL_MM(nn.Module):
    """
    Neural network module consisting of Linear, BatchNorm, Dropout, and LeakyReLU layers.
    """

    def __init__(
        self, in_dim, out_dim, channel=8, use_RL=True, use_BN=True, use_DO=True
    ):
        """
        Initializes the module.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            channel (int): Unused parameter.
            use_RL (bool): Whether to use LeakyReLU activation.
            use_BN (bool): Whether to use BatchNorm1d.
            use_DO (bool): Whether to use Dropout.
        """
        super(NN_FCBNRL_MM, self).__init__()
        layers = []
        # Linear layer
        layers.append(nn.Linear(in_dim, out_dim))
        # Optional Dropout
        # if use_DO:
        #     layers.append(nn.Dropout(p=0.02))
        # Optional BatchNorm
        if use_BN:
            layers.append(nn.BatchNorm1d(out_dim))
        # Optional LeakyReLU activation
        if use_RL:
            layers.append(nn.LeakyReLU(0.1))

        # Create the sequential block
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of the module."""
        return self.block(x)


# DMT encoder with a fully connected or ResNet backbone.
class DMTEncoder(nn.Module):
    """
    DMT encoder module.

    The current density-preservation experiments use the fully connected
    backbone by default. The constructor keeps several legacy transformer-style
    argument names for configuration compatibility, but this class is not a
    Transformer encoder.
    """

    def __init__(
        self,
        num_layers=2,
        num_attention_heads=6,
        hidden_size=240,
        intermediate_size=300,
        max_position_embeddings=784,
        num_input_dim=784,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_use_moe=10,
        output_dim=512,
        use_moe=True,
        use_resnet=False
    ):
        """
        Initializes the DMT encoder.

        Args:
            num_layers (int): Number of layers.
            num_attention_heads (int): Legacy compatibility argument; unused by
                the fully connected and ResNet backbones.
            hidden_size (int): Hidden size.
            intermediate_size (int): Intermediate size.
            max_position_embeddings (int): Legacy compatibility argument.
            num_input_dim (int): Input dimension size.
            hidden_dropout_prob (float): Dropout probability for hidden layers.
            attention_probs_dropout_prob (float): Legacy compatibility argument.
            num_use_moe (int): Legacy compatibility argument.
            use_moe (bool): Legacy compatibility argument.
        """
        super(DMTEncoder, self).__init__()
        self.use_moe = use_moe

        # Determine the type of network to use based on input dimension
        # if num_input_dim == 3072:
        #     nn_type = "resnet"
        #     print("Using ResNet")
        if use_resnet:
            nn_type = "resnet"
            print("Using ResNet")
        else:
            nn_type = "nn"
            print("Using fully connected network")

        self.enc = self.network_single(
            num_input_dim,
            hidden_size,
            num_layers,
            nn_type=nn_type,
        )

        self.fc = nn.Sequential(
            NN_FCBNRL_MM(hidden_size, output_dim, use_RL=False),
        )

    def network_single(self, num_input_dim, hidden_size, num_layers, nn_type="nn"):
        """
        Creates a single network (either ResNet or fully connected).

        Args:
            num_input_dim (int): Input dimension.
            hidden_size (int): Hidden size.
            num_layers (int): Number of layers.
            nn_type (str): Type of network ('nn' or 'resnet').

        Returns:
            enc (nn.Module): The network module.
        """
        if nn_type == "resnet":
            # Use ResNet architecture
            enc = ResNet(BasicBlock, [2, 2, 2, 2], 3)
        else:
            # Build fully connected network
            layers = []
            layers.append(NN_FCBNRL_MM(num_input_dim, hidden_size))
            for _ in range(num_layers):
                layers.append(NN_FCBNRL_MM(hidden_size, hidden_size))
            layers.append(NN_FCBNRL_MM(hidden_size, hidden_size, use_RL=False))
            enc = nn.Sequential(*layers)
        return enc

    def forward(self, input_x):
        """
        Forward pass of the DMT encoder.

        Args:
            input_x (Tensor): Input tensor of shape (batch_size, num_use_moe, ...).

        Returns:
            emb (Tensor): Output embeddings.
        """
        # if self.use_moe:
        #     # If using MoE, apply each expert to the input
        #     emb_all = [self.fc(enc(input_x[:, i, :])) for i, enc in enumerate(self.enc)]
        #     emb = torch.stack(emb_all, dim=1)
        # else:
        # Single encoder
        emb = self.fc(self.enc(input_x))
        return emb
