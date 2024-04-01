import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
import itertools


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsLogger:
    def __init__(self, train_metric_names, eval_metric_names, writer=None):
        self.train_metric_names = train_metric_names
        self.eval_metric_names = eval_metric_names
        self.total_train_sample = 0
        self.writer = writer

        # define meters to calculate average
        self.train_meters = {
            metric_name: AverageMeter() for metric_name in train_metric_names
        }
        self.eval_meters = {
            metric_name: AverageMeter() for metric_name in eval_metric_names
        }

        # define metrics to track sample-level metrics
        self.train_metrics = {
            "average": {metric_name: [] for metric_name in train_metric_names},
            "per_iteration": {metric_name: [] for metric_name in train_metric_names},
        }
        self.train_metrics["per_iteration_sample"] = []
        self.train_metrics["total_sample"] = []

        self.eval_metrics = {
            "average": {metric_name: [] for metric_name in eval_metric_names},
            "per_iteration": {metric_name: [] for metric_name in eval_metric_names},
        }

        self.eval_metrics["total_sample"] = []

    def update_train_metrics(self, batchsize, **metrics):
        self.total_train_sample += batchsize
        self.train_metrics["per_iteration_sample"].append(batchsize)
        self.train_metrics["total_sample"].append(self.total_train_sample)
        self._update_metrics(metrics, train=True)
        if self.writer is not None:
            self.writer.add_scalar("batchsize", batchsize, self.total_train_sample)

    def update_eval_metrics(self, **metrics):
        self.eval_metrics["total_sample"].append(self.total_train_sample)
        self._update_metrics(metrics, train=False)

    def _update_metrics(self, metrics, train):
        if train:
            for metric_name, metric_value in metrics.items():
                if metric_name not in self.train_metric_names:
                    raise ValueError(f"Metric '{metric_name}' is not defined.")
                self.train_meters[metric_name].update(metric_value)
                self.train_metrics["per_iteration"][metric_name].append(
                    self.train_meters[metric_name].val
                )
                self.train_metrics["average"][metric_name].append(
                    self.train_meters[metric_name].avg
                )
                if self.writer is not None:
                    scalar_name = f"train/{metric_name}"
                    self.writer.add_scalar(
                        scalar_name,
                        self.train_meters[metric_name].val,
                        self.total_train_sample,
                    )
                    scalar_name = f"train/{metric_name}_avg"
                    self.writer.add_scalar(
                        scalar_name,
                        self.train_meters[metric_name].avg,
                        self.total_train_sample,
                    )
        else:
            for metric_name, metric_value in metrics.items():
                if metric_name not in self.eval_metric_names:
                    raise ValueError(f"Metric '{metric_name}' is not defined.")
                self.eval_meters[metric_name].update(metric_value)
                self.eval_metrics["per_iteration"][metric_name].append(
                    self.eval_meters[metric_name].val
                )
                self.eval_metrics["average"][metric_name].append(
                    self.eval_meters[metric_name].avg
                )
                if self.writer is not None:
                    scalar_name = f"full/{metric_name}"
                    self.writer.add_scalar(
                        scalar_name,
                        self.eval_meters[metric_name].val,
                        self.total_train_sample,
                    )
                    scalar_name = f"full/{metric_name}_avg"
                    self.writer.add_scalar(
                        scalar_name,
                        self.eval_meters[metric_name].avg,
                        self.total_train_sample,
                    )


# =======================================================================================#


def get_random_search_space(search_space, num_sample):
    # used for old code, deprecated
    arguments = []
    while len(arguments) < num_sample:
        idx_stepsize = np.random.choice(len(search_space["stepsize"]))
        idx_multiplier = np.random.choice(len(search_space["multiplier"]))
        idx_exponent = np.random.choice(len(search_space["exponent"]))

        arguments.append(
            (
                search_space["stepsize"][idx_stepsize],
                search_space["multiplier"][idx_multiplier],
                search_space["exponent"][idx_exponent],
            )
        )
        # remove same arguments
        arguments = list(set(arguments))
    return arguments


def get_random_combination(search_space, num_selection):
    keys = list(search_space.keys())
    total_combinations = np.prod([len(search_space[key]) for key in keys])
    if num_selection > total_combinations:
        raise ValueError(
            f"num_selection should be less than the total number of combinations in search space. Total number of combinations: {total_combinations}"
        )

    arguments = []

    while len(arguments) < num_selection:
        combination = {}
        for key in keys:
            combination[key] = np.random.choice(search_space[key])
        if combination not in arguments:
            arguments.append(combination)

    return arguments


if __name__ == "__main__":

    def test_MetricsLogger():
        writer = SummaryWriter(log_dir="temp/test_MetricsLogger")

        train_metrics = ["loss", "accuracy"]
        val_metrics = ["val_loss", "val_accuracy"]

        logger = MetricsLogger(train_metrics, val_metrics, writer)

        # Example training loop
        loss = []
        acc = []
        eval_loss = []
        eval_acc = []
        for epoch in range(1, 6):
            train_loss = 0.5 / epoch
            train_accuracy = 0.8 + 0.05 * epoch
            loss.append(train_loss)
            acc.append(train_accuracy)
            # Update training metrics
            logger.update_train_metrics(
                batchsize=10, loss=train_loss, accuracy=train_accuracy
            )

            # Example evaluation
            if epoch % 2 == 0:
                val_loss = 0.2 / epoch
                val_accuracy = 0.9 - 0.05 * epoch

                eval_loss.append(val_loss)
                eval_acc.append(val_accuracy)
                # Update evaluation metrics
                logger.update_eval_metrics(val_loss=val_loss, val_accuracy=val_accuracy)

        # Get average metrics
        assert logger.train_metrics["per_iteration"]["loss"] == loss
        assert logger.train_metrics["per_iteration"]["accuracy"] == acc

        assert logger.eval_metrics["per_iteration"]["val_loss"] == eval_loss
        assert logger.eval_metrics["per_iteration"]["val_accuracy"] == eval_acc
        # Close the SummaryWriter
        writer.close()

    # test_MetricsLogger()

    def test_get_random_search_space():
        search_space = {"st": [2, 3, 5], "2": np.logspace(-3, 1, 5), "batch": [0.1]}
        num_sample = 5
        return get_random_combination(search_space, num_sample)

    print(test_get_random_search_space())
