import copy
import logging, os
import numpy as np
import torch
import torch.nn.functional as F

from torch.nn.utils import parameters_to_vector, vector_to_parameters

from funcs.prox import prox_l1, prox_l12


from funcs.utils import MetricsLogger


class DROalgorithm:
    def __init__(
        self,
        args,
        model,
        dataloader_train,
        dataloader_eval,
        robust_loss,
        robust_loss_eval,
        writer=None,
    ):
        self.args = args
        self.model = model
        self.robust_loss = robust_loss
        self.robust_loss_eval = robust_loss_eval
        self.loader_train = dataloader_train
        self.loader_eval = dataloader_eval

        self.writer = writer

        self.device = torch.device(
            "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.train_metrics = ["loss", "grad_norm"]
        self.val_metrics = ["total_loss", "full_grad_norm", "accuracy"]
        self.logger = MetricsLogger(self.train_metrics, self.val_metrics, self.writer)

    @staticmethod
    def create_algorithm(
        args,
        model,
        dataloader_train,
        dataloader_eval,
        robust_loss,
        robust_loss_eval,
        writer=None,
    ):
        algorithm_name = args.algorithm.lower()
        if algorithm_name == "proxsgd":
            return proxSGD(
                args,
                model,
                dataloader_train,
                dataloader_eval,
                robust_loss,
                robust_loss_eval,
                writer,
            )
        elif algorithm_name == "proxabg":
            return proxABG(
                args,
                model,
                dataloader_train,
                dataloader_eval,
                robust_loss,
                robust_loss_eval,
                writer,
            )
        elif algorithm_name == "abg_storm":
            return ABG_STORM(
                args,
                model,
                dataloader_train,
                dataloader_eval,
                robust_loss,
                robust_loss_eval,
                writer,
            )
        elif algorithm_name == "multistage_storm":
            return Multistage_STORM(
                args,
                model,
                dataloader_train,
                dataloader_eval,
                robust_loss,
                robust_loss_eval,
                writer,
            )
        elif algorithm_name == "multilevel":
            return multilevel(
                args,
                model,
                dataloader_train,
                dataloader_eval,
                robust_loss,
                robust_loss_eval,
                writer,
            )
        elif algorithm_name == "dual":
            return dual(
                args,
                model,
                dataloader_train,
                dataloader_eval,
                robust_loss,
                robust_loss_eval,
                writer,
            )
        elif algorithm_name == "primaldual":
            return primaldual(
                args,
                model,
                dataloader_train,
                dataloader_eval,
                robust_loss,
                robust_loss_eval,
                writer,
            )
        else:
            raise ValueError(f"Invalid algorithm: {algorithm_name}")

    def _proximal_update(self, update=True):
        # do gradient descent step then proximal step
        # if update is False, only calculate generalized gradient and regulizer value in loss function
        with torch.no_grad():
            grad = torch.cat(
                [
                    p.grad.detach().view(-1)
                    for p in filter(lambda p: p.requires_grad, self.model.parameters())
                ]
            )
            param = parameters_to_vector(
                [p for p in filter(lambda p: p.requires_grad, self.model.parameters())]
            )
            pre_param = param.detach().clone()
            param -= self.args.stepsize * grad

            # proximal step
            if self.args.regularizer == "l1":
                param = prox_l1(
                    param, self.args.stepsize * self.args.regularizer_strength
                )
                val_reg = (
                    self.args.regularizer_strength * torch.norm(pre_param, 1).item()
                )
            elif self.args.regularizer == "l12":
                param = prox_l12(
                    param, self.args.stepsize * self.args.regularizer_strength
                )
                val_reg = (
                    0.5
                    * self.args.regularizer_strength
                    * (
                        torch.norm(pre_param, 1).item()
                        + 0.5 * torch.norm(pre_param, 2).item() ** 2
                    )
                )
            elif self.args.regularizer is None:
                val_reg = 0

            grad_norm2 = torch.norm(param - pre_param, 2).pow(2).item() / (
                self.args.stepsize**2
            )

            if update:
                vector_to_parameters(
                    param,
                    [
                        p
                        for p in filter(
                            lambda p: p.requires_grad, self.model.parameters()
                        )
                    ],
                )

        return grad_norm2, val_reg

    def take_grad(self, model, x, y):
        model.zero_grad()
        outputs = model(x)
        per_example_loss = F.cross_entropy(outputs, y, reduction="none")
        loss = self.robust_loss(per_example_loss)
        loss.backward()
        return loss.item()

    def train_step(self, x, y):
        self.model.zero_grad()
        outputs = self.model(x)
        per_example_loss = F.cross_entropy(outputs, y, reduction="none")
        loss = self.robust_loss(per_example_loss)
        loss.backward()

        grad_norm2, reg_val = self._proximal_update()

        return loss.item() + reg_val, grad_norm2

    def train(self):
        self.model.train()

        total_sample = 0
        logged_sample = 0

        while total_sample < self.args.batch_allocation:
            x, y, _, _ = next(iter(self.loader_train))
            total_sample += len(x)
            x, y = x.to(self.device), y.to(self.device)

            train_loss, train_grad_norm2 = self.train_step(x, y)

            # log the results
            self.logger.update_train_metrics(
                batchsize=len(x), loss=train_loss, grad_norm=train_grad_norm2
            )

            # evaluate over the whole training set
            if total_sample - logged_sample >= self.args.log_per_batch:
                logged_sample = total_sample

                # solve the inner function and get p^*
                with torch.no_grad():
                    x, y, _, _ = self.loader_eval.dataset[:]
                    x, y = x.to(self.device), y.to(self.device)
                    output = self.model(x)
                    prediction = output.argmax(dim=1)
                    accuracy = (prediction == y).float().mean()
                    per_example_loss = F.cross_entropy(output, y, reduction="none")
                    p = self.robust_loss_eval.best_response(per_example_loss)

                # calculate the loss and gradient
                total_loss = 0
                self.model.zero_grad()
                for x, y, _, index in self.loader_eval:
                    x, y = x.to(self.device), y.to(self.device)
                    output = self.model(x)
                    per_example_loss = F.cross_entropy(output, y, reduction="none")
                    loss = torch.dot(per_example_loss, p[index])
                    loss.backward()
                    total_loss += loss.item()

                full_grad_norm2, reg_val = self._proximal_update(update=False)
                total_loss += reg_val

                self.logger.update_eval_metrics(
                    total_loss=total_loss,
                    full_grad_norm=full_grad_norm2,
                    accuracy=accuracy.item(),
                )

                logging.info(
                    f"sample [{total_sample} / {self.args.batch_allocation} ({100. * total_sample / self.args.batch_allocation:.0f}%)] accuracy: {100*accuracy:.2f}% train/loss {train_loss:.4f} train/grad_norm {train_grad_norm2:.4f} full loss {total_loss:.4f}\t full grad_norm {full_grad_norm2:.4f}"
                )

        return self.logger


class proxSGD(DROalgorithm):
    def __init__(
        self,
        args,
        model,
        dataloader_train,
        dataloader_eval,
        robust_loss,
        robust_loss_eval,
        writer=None,
    ):
        super(proxSGD, self).__init__(
            args,
            model,
            dataloader_train,
            dataloader_eval,
            robust_loss,
            robust_loss_eval,
            writer,
        )


class proxABG(DROalgorithm):
    def __init__(
        self,
        args,
        model,
        dataloader_train,
        dataloader_eval,
        robust_loss,
        robust_loss_eval,
        writer=None,
    ):
        super(proxABG, self).__init__(
            args,
            model,
            dataloader_train,
            dataloader_eval,
            robust_loss,
            robust_loss_eval,
            writer,
        )

    def train_step(self, x, y):
        loss, grad_norm2 = super(proxABG, self).train_step(x, y)

        # calculate the new batch size
        num_batch = self.args.ABG_multiplier * grad_norm2 ** (self.args.ABG_exponent)
        num_batch = int(np.ceil(num_batch))
        num_batch = np.clip(num_batch, self.args.batch_size, self.args.batch_size_max)
        self.loader_train.batch_sampler.batch_size = num_batch

        return loss, grad_norm2


class ABG_STORM(DROalgorithm):
    def __init__(
        self,
        args,
        model,
        dataloader_train,
        dataloader_eval,
        robust_loss,
        robust_loss_eval,
        writer=None,
    ):
        super(ABG_STORM, self).__init__(
            args,
            model,
            dataloader_train,
            dataloader_eval,
            robust_loss,
            robust_loss_eval,
            writer,
        )
        self.grad_estimator = {}
        self.model_prestep = None

    def train_step(self, x, y):
        loss = self.take_grad(self.model, x, y)
        if not self.grad_estimator:
            # first step
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.grad_estimator[name] = param.grad.detach().clone()
        else:
            # calculate grad at previous parameters
            self.take_grad(self.model_prestep, x, y)
            # update grad estimator
            with torch.no_grad():
                # add gradient in self.model and model_prestep
                for name_param, name_param_pre in zip(
                    self.model.named_parameters(), self.model_prestep.named_parameters()
                ):
                    if name_param[1].grad is not None:
                        self.grad_estimator[name_param[0]] = name_param[
                            1
                        ].grad.detach().clone() + (1 - self.args.STORM_beta) * (
                            self.grad_estimator[name_param[0]]
                            - name_param_pre[1].grad.detach().clone()
                        )

        self.model_prestep = copy.deepcopy(self.model)
        # assign gradient estimator to model
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param.grad.data = self.grad_estimator[name]

        # update parameters
        grad_norm2, reg_val = self._proximal_update()

        # calculate the new batch size
        num_batch = self.args.ABG_multiplier * grad_norm2 ** (self.args.ABG_exponent)
        num_batch = int(np.ceil(num_batch))
        num_batch = np.clip(num_batch, self.args.batch_size, self.args.batch_size_max)
        self.loader_train.batch_sampler.batch_size = num_batch

        return loss + reg_val, grad_norm2


class Multistage_STORM(DROalgorithm):
    def __init__(
        self,
        args,
        model,
        dataloader_train,
        dataloader_eval,
        robust_loss,
        robust_loss_eval,
        writer=None,
    ):
        super(Multistage_STORM, self).__init__(
            args,
            model,
            dataloader_train,
            dataloader_eval,
            robust_loss,
            robust_loss_eval,
            writer,
        )
        self.iteration = 0
        self.num_batch = self.args.batch_size
        self.grad_estimator = {}
        self.model_prestep = None

    def train_step(self, x, y):
        loss = self.take_grad(self.model, x, y)
        if self.iteration % self.args.Multi_STORM_loop == 0:
            # start new stage
            self.grad_estimator = {}
            self.model_prestep = None

            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.grad_estimator[name] = param.grad.detach().clone()
            # increase the batch size
            self.num_batch = self.num_batch * self.args.STORM_multiplier
            self.num_batch = int(self.num_batch)
            self.num_batch = np.clip(
                self.num_batch, self.args.batch_size, self.args.batch_size_max
            )
            self.loader_train.batch_sampler.batch_size = self.num_batch
        else:
            # calculate grad at previous parameters
            self.take_grad(self.model_prestep, x, y)
            # update grad estimator
            with torch.no_grad():
                # add gradient in self.model and model_prestep
                for name_param, name_param_pre in zip(
                    self.model.named_parameters(), self.model_prestep.named_parameters()
                ):
                    if name_param[1].grad is not None:
                        self.grad_estimator[name_param[0]] = name_param[
                            1
                        ].grad.detach().clone() + (1 - self.args.STORM_beta) * (
                            self.grad_estimator[name_param[0]]
                            - name_param_pre[1].grad.detach().clone()
                        )

        self.model_prestep = copy.deepcopy(self.model)
        # assign gradient estimator to model
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param.grad.data = self.grad_estimator[name]

        # update parameters
        grad_norm2, reg_val = self._proximal_update()

        self.iteration += 1
        return loss + reg_val, grad_norm2


class multilevel(DROalgorithm):
    def __init__(
        self,
        args,
        model,
        dataloader_train,
        dataloader_eval,
        robust_loss,
        robust_loss_eval,
        writer=None,
    ):
        super(multilevel, self).__init__(
            args,
            model,
            dataloader_train,
            dataloader_eval,
            robust_loss,
            robust_loss_eval,
            writer,
        )


class dual(DROalgorithm):
    def __init__(
        self,
        args,
        model,
        dataloader_train,
        dataloader_eval,
        robust_loss,
        robust_loss_eval,
        writer=None,
    ):
        super(dual, self).__init__(
            args,
            model,
            dataloader_train,
            dataloader_eval,
            robust_loss,
            robust_loss_eval,
            writer,
        )

    def train_step(self, x, y):
        self.robust_loss.zero_grad()
        self.model.zero_grad()
        outputs = self.model(x)
        per_example_loss = F.cross_entropy(outputs, y, reduction="none")
        loss = self.robust_loss(per_example_loss)
        loss.backward()

        with torch.no_grad():
            for p in self.robust_loss.parameters():
                p -= self.args.stepsize_eta * p.grad

        grad_norm2, reg_val = self._proximal_update()

        return loss.item() + reg_val, grad_norm2


class primaldual(DROalgorithm):
    def __init__(
        self,
        args,
        model,
        dataloader_train,
        dataloader_eval,
        robust_loss,
        robust_loss_eval,
        writer=None,
    ):
        super(primaldual, self).__init__(
            args,
            model,
            dataloader_train,
            dataloader_eval,
            robust_loss,
            robust_loss_eval,
            writer,
        )
