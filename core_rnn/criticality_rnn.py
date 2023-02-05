import torch
import torch.nn as nn
from core_rnn.recurrent_body_builder import RecurrentBodyBuilder


class CriticalityRNN(nn.Module):
    def __init__(self, stem_info, num_hidden_units, num_classes):
        """Network class for Criticality Classifier."""
        super(CriticalityRNN, self).__init__()
        self.learn_init_hidden = stem_info["learn_hidden_init"]

        # Define the recurrent stem based on RecurrentBodyBuilder
        self.rnn = RecurrentBodyBuilder(**stem_info)

        # Define two output heads - classification predictions + activity
        self.classification_head = nn.Linear(num_hidden_units, num_classes)
        self.classification_act = nn.LogSoftmax(dim=2)
        # self.activity_head = nn.Linear(num_hidden_units, 2)
        self.activity_act = nn.Sigmoid()

    def forward(self, x):
        """Compute shared activations & predictions."""
        hidden_activation = self.rnn(x)
        class_logits = self.classification_head(hidden_activation)
        class_pred = self.classification_act(class_logits)
        return class_pred

    def init_hidden(self, device, batch_size):
        """Initialize the hidden state of the RNN."""
        return self.rnn.init_hidden(device, batch_size)


class CriticalityLoss(nn.Module):
    """Loss function combining classification & activity."""

    def __init__(self, activity_weight=1):
        super(CriticalityLoss, self).__init__()
        self.activity_weight = activity_weight
        self.cross_entropy_loss = nn.NLLLoss()
        self.activity_loss = nn.MSELoss(reduce=False)

    def forward(self, predictions, targets, return_separate_loss=False):
        """Compute the weighted sum of losses."""
        # Return aggregated loss for training
        if not return_separate_loss:
            ce_l = self.cross_entropy_loss(
                predictions[:, :-2], targets[:, 0].long()
            )
            act_l = self.activity_loss(
                predictions[:, -2:], targets[:, 1:].float()
            ).sum()
            return ce_l + self.activity_weight * act_l
        # Return separated loss for evaluation
        else:
            return self.activity_loss(
                predictions[:, -2:], targets[:, 1:].float()
            )


def load_net_params(network, ckpth_path):
    """Load in the parameters of the network from a checkpoint"""
    checkpoint = torch.load(ckpth_path, map_location="cpu")
    network.load_state_dict(checkpoint)
