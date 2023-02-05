import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ReduceLROnPlateau,
    CyclicLR,
)


def set_optimizer(network, opt_type, l_rate, momentum=None, w_decay=None):
    """
    Set optimizer for network. If applicable set momentum & regularisation.
    """
    if momentum is None:
        momentum = 0.0
    if w_decay is None:
        w_decay = 0.0

    if opt_type == "Adam":
        optimizer = optim.Adam(
            network.parameters(), lr=l_rate, weight_decay=w_decay
        )
    elif opt_type == "AdamW":
        optimizer = optim.AdamW(
            network.parameters(), lr=l_rate, weight_decay=w_decay
        )
    elif opt_type == "RMSprop":
        optimizer = optim.RMSprop(
            network.parameters(),
            lr=l_rate,
            momentum=momentum,
            weight_decay=w_decay,
        )
    elif opt_type == "SGD":
        optimizer = optim.SGD(
            network.parameters(),
            lr=l_rate,
            momentum=momentum,
            weight_decay=w_decay,
        )
    elif opt_type == "AMSGrad":
        optimizer = optim.Adam(
            network.parameters(),
            lr=l_rate,
            amsgrad=momentum,
            weight_decay=w_decay,
        )
    elif opt_type == "Adadelta":
        optimizer = optim.Adadelta(
            network.parameters(), lr=l_rate, weight_decay=w_decay
        )
    else:
        raise ValueError
    return optimizer


def set_lrate_schedule(optimizer, schedule_type, schedule_inputs):
    """Set the learning rate schedule for a specific network"""
    # See pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
    if schedule_type == "multiplicative-decay":
        # step_size - every x-th epoch decrease the lrate by factor gamma
        scheduler = StepLR(
            optimizer,
            step_size=schedule_inputs["step_size"],
            gamma=schedule_inputs["gamma"],
        )
    elif schedule_type == "multi-step":
        scheduler = MultiStepLR(
            optimizer,
            milestones=schedule_inputs["milestones"],
            gamma=schedule_inputs["gamma"],
        )
    elif schedule_type == "reduce-on-plateu":
        # Note: Need to provide metric to check plateu - optimizer.step(metric)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=schedule_inputs["mode"],
            factor=schedule_inputs["factor"],
            patience=schedule_inputs["patience"],
        )
    elif schedule_type == "cyclic":
        scheduler = CyclicLR(
            optimizer,
            base_lr=schedule_inputs["base_lr"],
            max_lr=schedule_inputs["max_lr"],
            step_size_up=schedule_inputs["step_size_up"],
        )
    else:
        raise ValueError
    return scheduler
