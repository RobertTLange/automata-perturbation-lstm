import torch


class RNNDojo:
    """Learning Loop for basic RNN/sequence learning setups"""

    def __init__(
        self,
        network,
        optimizer,
        criterion,
        only_final_pred,
        device,
        problem_type,
        train_loader,
        test_loader=None,
        train_log=None,
        log_batch_interval=None,
        scheduler=None,
        tboard_network_stats=False,
        num_classes=None,
    ):
        # Initialize base dojo
        self.network = network  # Network to train
        self.criterion = criterion  # Loss criterion to minimize
        self.device = device  # Device for tensors
        self.optimizer = optimizer  # PyTorch optimizer for SGD
        self.scheduler = scheduler  # Learning rate scheduler
        self.only_final_pred = only_final_pred  # Only use final pred in loss

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.problem_type = problem_type
        self.num_classes = num_classes
        self.logger = train_log
        self.log_batch_interval = log_batch_interval
        self.batch_processed = 0
        self.tboard_network_stats = tboard_network_stats

    def train(self, num_epochs):
        """Loop over epochs in training & test on all hold-out batches"""
        # Get Initial Performance after Network Initialization
        train_performance = self.get_network_performance(test=False)
        if self.test_loader is not None:
            test_performance = self.get_network_performance(test=True)
        else:
            test_performance = [0, 0]

        # Update the logging instance with the initial random performance
        clock_tick = {"total_batches": 0, "num_epoch": 0, "batch_in_epoch": 0}
        stats_tick = {
            "train_loss": train_performance[0],
            "train_acc": train_performance[1],
            "test_loss": test_performance[0],
            "test_acc": test_performance[1],
        }

        # Save the log & the initial network checkpoint
        self.logger.update(clock_tick, stats_tick, self.network, save=True)

        for epoch_id in range(1, num_epochs + 1):
            # Train the network for a single epoch
            self.train_for_epoch(epoch_id)

            # Update the learning rate using the scheduler (if desired)
            if self.scheduler is not None:
                self.scheduler.step()

        self.logger.save_model(self.network)

    def train_for_epoch(self, epoch_id=0):
        """Perform one epoch of training with train data loader"""
        self.network.train()
        # Loop over batches in the training dataset
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Put data on the device
            data, target = data.to(self.device), target.to(self.device)

            # Clear gradients
            self.optimizer.zero_grad()

            # Initialize the hidden state
            self.network.init_hidden(
                device=self.device, batch_size=data.size(0)
            )

            # If we are training on a classification problem - change datatype!
            target = target.long()

            # Sum over timesteps
            loss = 0
            for t in range(target.size(1)):
                # Perform forward/backward pass
                output = self.network(data[:, t : t + 1, :].float()).squeeze()
                # Accumulate loss from each timestep (if desired)
                if not self.only_final_pred:
                    loss += self.criterion(output, target[:, t]).sum()

            if self.only_final_pred:
                loss += self.criterion(output, target[:, -1]).sum()
            else:
                # Normalize by number of timesteps
                loss /= target.size(1)

            loss.backward()
            self.optimizer.step()

            # Update the batches processed counter
            self.batch_processed += 1

            # Log the performance of the network
            if self.batch_processed % self.log_batch_interval == 0:
                # Get Current Performance after Single Epoch of Training
                train_performance = self.get_network_performance(test=False)
                if self.test_loader is not None:
                    test_performance = self.get_network_performance(test=True)
                else:
                    test_performance = [0, 0]

                # Update the logging instance
                clock_tick = {
                    "total_batches": self.batch_processed,
                    "num_epoch": epoch_id,
                    "batch_in_epoch": batch_idx + 1,
                }
                stats_tick = {
                    "train_loss": train_performance[0],
                    "train_acc": train_performance[1],
                    "test_loss": test_performance[0],
                    "test_acc": test_performance[1],
                }

                self.logger.update(
                    clock_tick, stats_tick, self.network, save=True
                )
                # Set network back into training mode
                self.network.train()
        return

    def get_network_performance(self, test=False):
        """Get the performance of the network"""
        # Log the classifier accuracy if problem is classification
        loader = self.test_loader if test else self.train_loader

        self.network.eval()
        avg_loss, avg_correct = 0, 0

        with torch.no_grad():
            # Loop over batches and get average batch accuracy/loss
            for data, target in loader:
                loss_t, correct_t = 0, 0
                data, target = data.to(self.device), target.to(self.device)
                # Initialize the hidden state
                self.network.init_hidden(
                    device=self.device, batch_size=data.size(0)
                )

                # If we are training on a classification problem - change datatype!
                target = target.long()

                # Sum over timesteps
                for t in range(target.size(1)):
                    # Get loss for batch and preds via forward
                    output = self.network(
                        data[:, t : t + 1, :].float()
                    ).squeeze()
                    pred = output[:, : self.num_classes].argmax(
                        dim=1, keepdim=True
                    )

                    # Accumulate loss/acc from each t (if desired)
                    if not self.only_final_pred:
                        loss_t += (
                            self.criterion(output, target[:, t]).sum().item()
                        )
                        correct_t += (
                            pred.eq(target[:, t].view_as(pred)).sum().item()
                        )

                if self.only_final_pred:
                    loss_t += self.criterion(output, target[:, -1]).sum().item()
                    correct_t += (
                        pred.eq(target[:, -1].view_as(pred)).sum().item()
                    )
                else:
                    # Normalize by number of timesteps
                    loss_t /= target.size(1)
                    correct_t /= target.size(1)

                avg_loss += loss_t
                avg_correct += correct_t

        avg_loss /= len(loader.dataset)
        avg_correct /= len(loader.dataset)
        return [avg_loss, avg_correct]
