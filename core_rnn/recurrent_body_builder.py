from .body_builder import BodyBuilder, Flatten

import torch
import torch.nn as nn

import numpy as np


class RecurrentBodyBuilder(BodyBuilder):
    """Patch BodyBuilder Class for Recurrent Layers"""

    def __init__(
        self,
        input_dim,
        layers_info,
        output_act,
        hidden_act,
        dropout=0.0,
        batch_norm=False,
        load_in_path=None,
        learn_constant=False,
        layer_init=None,
        learn_hidden_init=False,
    ):
        BodyBuilder.__init__(
            self,
            input_dim,
            layers_info,
            output_act,
            hidden_act,
            dropout,
            batch_norm,
            learn_constant,
            layer_init,
            load_in_path,
        )
        # Set whether to learn the hidden state initialization
        self.learn_hidden_init = learn_hidden_init
        if self.learn_hidden_init:
            self.init_hidden_h = nn.Parameter(
                torch.randn(
                    self.num_rec_layers * self.num_rec_directions,
                    1,
                    self.num_rec_hidden_units,
                ),
                requires_grad=True,
            )
            if self.recurrence_type == "lstm":
                self.init_hidden_c = nn.Parameter(
                    torch.randn(
                        self.num_rec_layers * self.num_rec_directions,
                        1,
                        self.num_rec_hidden_units,
                    ),
                    requires_grad=True,
                )

    def initialize_valid_activations(self):
        """Dictionary mapping strings to activations."""
        self.str_to_act = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "softplus": nn.Softplus(),
            "tanh": nn.Tanh(),
            "softmax": nn.Softmax(dim=2),
            "log_softmax": nn.LogSoftmax(dim=2),
            "identity": nn.Identity(),
        }

    def build_layer(self, layer_config):
        """Function specifies the specific network layer with RNN layers
        RNN L Ex:  ["rnn", hidden_size, num_layers, bias, batch_first, bidirectional]
        GRU L Ex:  ["gru", hidden_size, num_layers, bias, batch_first, bidirectional]
        LSTM L Ex: ["lstm", hidden_size, num_layers, bias, batch_first, bidirectional]
        """
        layer_name = layer_config[0]
        # Base Layers
        if layer_name == "linear":
            layer = nn.Linear(
                self.input_dims[-1], layer_config[1], bias=layer_config[2]
            )
        elif layer_name == "flatten":
            layer = Flatten()
        elif layer_name == "conv1d":
            layer = nn.Conv1d(
                in_channels=self.input_dims[-1][0],
                out_channels=layer_config[1],
                kernel_size=layer_config[2],
                stride=layer_config[3],
                padding=layer_config[4],
            )
        elif layer_name == "conv2d":
            layer = nn.Conv2d(
                in_channels=self.input_dims[-1][0],
                out_channels=layer_config[1],
                kernel_size=layer_config[2],
                stride=layer_config[3],
                padding=layer_config[4],
            )
        elif layer_name == "lstm":
            layer = nn.LSTM(
                input_size=self.input_dims[-1],
                hidden_size=layer_config[1],
                num_layers=layer_config[2],
                bias=bool(layer_config[3]),
                batch_first=bool(layer_config[4]),
                bidirectional=bool(layer_config[5]),
            )
            # Set recurrent parameters for hidden state init
            self.recurrence_type = "lstm"
            self.num_rec_layers = layer_config[2]
            self.num_rec_hidden_units = layer_config[1]
            self.num_rec_directions = 1 + layer_config[5]
        elif layer_name == "gru":
            layer = nn.GRU(
                input_size=self.input_dims[-1],
                hidden_size=layer_config[1],
                num_layers=layer_config[2],
                bias=bool(layer_config[3]),
                batch_first=bool(layer_config[4]),
                bidirectional=bool(layer_config[5]),
            )
            # Set recurrent parameters for hidden state init
            self.recurrence_type = "gru"
            self.num_rec_layers = layer_config[2]
            self.num_rec_hidden_units = layer_config[1]
            self.num_rec_directions = 1 + layer_config[5]
        elif layer_name == "rnn":
            layer = nn.RNN(
                input_size=self.input_dims[-1],
                hidden_size=layer_config[1],
                num_layers=layer_config[2],
                bias=bool(layer_config[3]),
                batch_first=bool(layer_config[4]),
                bidirectional=bool(layer_config[5]),
            )
            # Set recurrent parameters for hidden state init
            self.recurrence_type = "rnn"
            self.num_rec_layers = layer_config[2]
            self.num_rec_hidden_units = layer_config[1]
            self.num_rec_directions = 1 + layer_config[5]
        # Pooling Techniques
        elif layer_name == "maxpool":
            layer = nn.MaxPool2d(
                kernel_size=layer_config[1],
                stride=layer_config[2],
                padding=layer_config[3],
            )
        elif layer_name == "avgpool":
            layer = nn.AvgPool2d(
                kernel_size=layer_config[1],
                stride=layer_config[2],
                padding=layer_config[3],
            )
        elif layer_name == "adaptivemaxpool":
            layer = nn.AdaptiveMaxPool2d(
                output_size=(layer_config[1], layer_config[2])
            )
        elif layer_name == "adaptiveavgpool":
            layer = nn.AdaptiveAvgPool2d(
                output_size=(layer_config[1], layer_config[2])
            )

        # Update the input dimension for the next layer
        new_dims = self.calc_new_input_dims(self.input_dims[-1], layer_config)
        self.input_dims.append(new_dims)
        return layer

    def init_hidden(self, device=torch.device("cpu"), batch_size=1):
        """Initialize the hidden state of the RNN"""
        # Allow for optimization of the hidden state initialization
        if self.learn_hidden_init:
            # Need to repeat initial hidden for each sample in batch
            batch_h = torch.cat(batch_size * [self.init_hidden_h], 1)
            if self.recurrence_type == "lstm":
                batch_c = torch.cat(batch_size * [self.init_hidden_c], 1)
                self.hidden = (batch_h, batch_c)
            elif self.recurrence_type == "gru" or self.recurrence_type == "rnn":
                self.hidden = batch_h
        # Otherwise simply set hidden state to zeros
        else:
            if self.recurrence_type == "lstm":
                self.hidden = (
                    torch.zeros(
                        self.num_rec_layers * self.num_rec_directions,
                        batch_size,
                        self.num_rec_hidden_units,
                    ).to(device),
                    torch.zeros(
                        self.num_rec_layers * self.num_rec_directions,
                        batch_size,
                        self.num_rec_hidden_units,
                    ).to(device),
                )
            elif self.recurrence_type == "gru" or self.recurrence_type == "rnn":
                self.hidden = torch.zeros(
                    self.num_rec_layers * self.num_rec_directions,
                    batch_size,
                    self.num_rec_hidden_units,
                ).to(device)

    def set_hidden(self, hidden):
        """Function to set the hidden state of RNN - training from buffer!"""
        self.hidden = hidden

    def forward(self, x, return_all_t=False, return_all_hiddens=False):
        """Perform the forward pass through the recurrent network"""
        # Get the sequence length based on size of input tensor
        seq_len = x.size(1)
        x_out, hidden_out = [], []
        # Loop over time steps in to be processed sequence
        for t in range(seq_len):
            # Select input vector for current timestep in sequence
            x_t = x[:, t, ...]
            # Loop over layers in the network
            for l in self.model.modules():
                # Very ugly module check to let hidden state dynamics be hidden
                if not isinstance(l, nn.Sequential):
                    if isinstance(l, (nn.RNN, nn.GRU, nn.LSTM)):
                        x_t = x_t.view(x_t.size(0), 1, x_t.size(1))
                        x_t, self.hidden = l(x_t, self.hidden)
                    else:
                        x_t = l(x_t)
            hidden_out.append(self.hidden[0].data.cpu().numpy())
            x_out.append(x_t)

        # Return the output for all timesteps or only for a single timestep
        if return_all_t and not return_all_hiddens:
            return torch.stack(x_out, dim=1)
        elif not return_all_t and return_all_hiddens:
            return np.stack(hidden_out)
        elif return_all_t and return_all_hiddens:
            return torch.stack(x_out, dim=1), np.stack(hidden_out)
        else:
            if not self.learn_constant:
                return x_t
            else:
                repeat_constant = torch.cat(
                    x_t.size(0) * [self.learned_constant]
                )
                return torch.cat([x_t.squeeze(2), repeat_constant], axis=1)
