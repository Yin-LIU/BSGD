## Compare to MLMC in Levy'20 paper
# %%
# import sys
import time

# sys.argv = [""]
import logging, os, argparse
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from funcs.dataset import (
    MNISTandTypedFeatures,
    RandomBatchSizeSampler,
    CustomDistributionSampler,
)
from funcs.robust_losses import (
    RobustLoss,
    MultiLevelRobustLoss,
    DualRobustLoss,
    PrimalDualRobustLoss,
)
from funcs.models import Net
from funcs.algorithms import DROalgorithm


# from torch.utils.tensorboard.summary import hparams


def none_or_str(value):
    if value == "None":
        return None
    else:
        return value


parser = argparse.ArgumentParser(description="Train the model using prox-ABG method")

# ---------------- data settings ----------------
parser.add_argument(
    "--data_dir",
    type=str,
    default="data/digit",
    help="directory where datasets are located",
)
parser.add_argument(
    "--target_dir",
    type=str,
    default="results",
    help="directory where results are located",
)

parser.add_argument(
    "--log_dir", default="Logs", type=str, help="directory where logs are located"
)
parser.add_argument(
    "--log_mode", default="w", type=str, help="log mode, w if overwrite; a if append"
)

parser.add_argument("--log_per_batch", default=10000, type=float, help="log per batch")
parser.add_argument("--seed", default=6, type=int, help="random seed")

# ---------------- training settings ----------------
parser.add_argument("--use_cuda", default=True, type=bool, help="to use cuda or not")
parser.add_argument("--num_trail", default=1, type=int, help="number of trails")
parser.add_argument(
    "--algorithm",
    default="multilevel",
    type=str,
    choices=(
        "proxABG",
        "proxSGD",
        "ABG_STORM",
        "Multistage_STORM",
        "multilevel",
        "dual",
        "primaldual",
    ),
    help="algorithm to use",
)
parser.add_argument("--stepsize", default=1e-5, type=float, help="stepsize")

parser.add_argument(
    "--batch_allocation",
    default=60000 * 30,
    type=float,
    help="maximum batch allocation",
)

parser.add_argument(
    "--batch_size",
    default=100,
    type=int,
    help="number of inner batch samples for the first iteration of proxABG and fixed batch for proxSGD",
)

parser.add_argument(
    "--batch_size_max",
    type=int,
    default=10000,
    help="Maximum batch size for multilevel and proxABG (-1 is full batch)",
)

# ---------------- regularizer settings ----------------
parser.add_argument(
    "--regularizer",
    default="l1",
    type=none_or_str,
    choices=("l1", None),
    help="regularizer to use",
)
parser.add_argument(
    "--regularizer_strength", default=1e-4, type=float, help="strength of regularizer"
)
# ---------------- proxABG settings ----------------
parser.add_argument("--ABG_multiplier", default=500, type=float, help="ABG multiplier")
parser.add_argument("--ABG_exponent", default=-1.5, type=float, help="ABG exponent")
parser.add_argument("--max_num_batch", default=10000, type=int, help="max num batch")

# ---------------- ABG_STORM settings ----------------
parser.add_argument("--STORM_beta", default=0.5, type=float, help="ABG_STORM beta")
parser.add_argument(
    "--Multi_STORM_loop", default=1000, type=int, help="Multi_STORM_loop"
)
parser.add_argument("--STORM_multiplier", default=2, type=float, help="STORM exponent")

# ---------------- multilevel settings ----------------

parser.add_argument(
    "--doubling_probability",
    type=float,
    default=0.5,
    help="Doubling probability for multi-level batch size",
)


# ---------------- dual settings ----------------
parser.add_argument(
    "--stepsize_eta",
    type=float,
    default=0.1,
    help="step size for Largrange dual variable",
)

# ---------------- primaldual settings ----------------
parser.add_argument(
    "--stepsize_dual",
    type=float,
    default=1e-3,
    help="step size for dual block in primal-dual methods",
)
parser.add_argument(
    "--clip",
    type=float,
    default=-1,
    help="gradient clipping for p in primal-dual methods",
)

args = parser.parse_args()


if args.use_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    args.use_cuda = False
    device = torch.device("cpu")


def init(args):
    # ------------------ load data ------------------

    dataset_train = MNISTandTypedFeatures(
        args.data_dir, train=True, subsample_to=(-1, 600)
    )
    # dataset_val = MNISTandTypedFeatures(
    #     args.data_dir, train=False, subsample_to=(-1, -1)
    # )

    if args.algorithm == "multilevel":
        train_sampler = RandomBatchSizeSampler(
            dataset_train,
            batch_size_min=args.batch_size,
            batch_size_max=args.batch_size_max,
            doubling_probability=args.doubling_probability,
            replace=False,
            num_batches=None,
        )
        loader_train = DataLoader(dataset_train, batch_sampler=train_sampler)

    elif args.algorithm in (
        "proxABG",
        "proxSGD",
        "ABG_STORM",
        "Multistage_STORM",
        "dual",
    ):
        loader_train = DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True
        )
    elif args.algorithm == "primaldual":
        train_sampler = CustomDistributionSampler(
            dataset_train, batch_size=args.batch_size
        )
        loader_train = DataLoader(dataset_train, batch_sampler=train_sampler)

    else:
        raise ValueError(f"algorithm {args.algorithm} not implemented")

    loader_eval_train = DataLoader(
        dataset_train, batch_size=2048, shuffle=False
    )  # loader for evaluation on full train set

    # ------------------ set loss function ------------------
    if args.algorithm in ("proxABG", "proxSGD", "ABG_STORM", "Multistage_STORM"):
        robust_loss = RobustLoss(1, 0, "chi-square")

    elif args.algorithm == "multilevel":
        robust_layer = RobustLoss(1, 0, "chi-square")
        robust_loss = MultiLevelRobustLoss(
            robust_layer, train_sampler.batch_size_pmf, args.batch_size
        )

    elif args.algorithm == "dual":
        robust_loss = DualRobustLoss(1, 0, "chi-square").to(device)

    elif args.algorithm == "primaldual":
        if args.clip < 0:
            clip = None
        else:
            clip = args.clip

        robust_loss = PrimalDualRobustLoss(
            1, "chi-square", train_sampler, args.stepsize_dual, clip=clip
        )
    else:
        raise ValueError(f"algorithm {args.algorithm} not implemented")
    robust_loss_eval = RobustLoss(1, 0, "chi-square")  # used for full batch evaluation
    return loader_train, loader_eval_train, robust_loss, robust_loss_eval


def main():
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_init = Net()

    loader_train, loader_eval_train, robust_loss, robust_loss_eval = init(args)

    os.makedirs(
        f"{args.target_dir}/main/reg{args.regularizer}{args.regularizer_strength:.0e}/{args.algorithm}",
        exist_ok=True,
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.log_dir, "log_main.log"),
                mode=args.log_mode,
            ),
            # mode="w" will overwrite the log file, mode="a" will append to the log file
            logging.StreamHandler(),
        ],
    )
    for idx in range(args.num_trail):
        model = copy.deepcopy(model_init).to(device)
        np.random.seed(int(time.time()))
        torch.manual_seed(int(time.time()))

        timestr = time.strftime("%m%d-%H%M")

        result_dir = f"main/reg{args.regularizer}{args.regularizer_strength:.0e}/{args.algorithm}/batch{args.batch_size}_step{args.stepsize:.0e}"

        # define directory for saving results
        if args.algorithm == "proxABG":
            result_dir += (
                f"_multiplier{args.ABG_multiplier:.0e}_exponent{args.ABG_exponent}"
            )
        elif args.algorithm == "ABG_STORM":
            result_dir += f"_multiplier{args.ABG_multiplier:.0e}_exponent{args.ABG_exponent}_beta{args.STORM_beta:.0e}"
        elif args.algorithm == "Multistage_STORM":
            result_dir += (
                f"_beta{args.STORM_beta:.0e}_multiplier{args.STORM_multiplier:.0e}"
            )
        elif args.algorithm == "dual":
            result_dir += f"_eta{args.stepsize_eta:.0e}"
        elif args.algorithm == "primaldual":
            result_dir += f"_dual{args.stepsize_dual:.0e}_clip{args.clip:.0e}"

        result_dir += f"_{timestr}"

        writer = SummaryWriter(log_dir=f"{args.target_dir}_tb/{result_dir}")

        algorithm = DROalgorithm.create_algorithm(
            args,
            model,
            loader_train,
            loader_eval_train,
            robust_loss,
            robust_loss_eval,
            writer,
        )

        logger = algorithm.train()
        result = {
            "train_metrics": logger.train_metrics,
            "eval_metrics": logger.eval_metrics,
            "args": vars(args),
        }
        torch.save(result, f"{args.target_dir}/{result_dir}_logger.pt")
        torch.save(model.state_dict(), f"{args.target_dir}/{result_dir}_model.pt")
        writer.close()
        logging.info(
            f"finished trail {idx + 1} of {args.num_trail} trails, saved to {args.target_dir}/{result_dir}_logger.pt"
        )
    return logger


if __name__ == "__main__":
    # logger = main()
    model = Net()
    model.load_state_dict("data/mnist_pretrain.pt")
