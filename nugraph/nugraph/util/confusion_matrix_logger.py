"""Utility class for logging confusion matrices"""
from torch import Tensor
import torchmetrics as tm
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger
import wandb

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sn

class ConfusionMatrixLogger:
    """
    Utility class for logging confusion matrices

    Args:
        classes: List of class names
    """
    def __init__(self, classes: list[str]):
        self.classes = classes


    def log(self, name: str, stage: str, cm: tm.ConfusionMatrix,
            logger: Logger | list[Logger], epoch: int) -> None:
        """
        Log confusion matrix

        Args:
            name: Decoder name
            stage: Stage name (train/test/val)
            cm: Torchmetrics confusion matrix
            logger: PyTorch Lightning logger object(s)
            epoch: Training epoch number
        """

        # create list of loggers
        if not logger:
            loggers = []
        elif isinstance(logger, list):
            loggers = logger
        else:
            loggers = [logger]

        # compute confusion matrices
        mx = cm.compute().cpu()
        mx_recall = mx / mx.sum(dim=1)[:, None]
        mx_precision = mx / mx.sum(dim=0)[None, :]
        mx_f1 = 2 * mx_recall * mx_precision / (mx_precision + mx_recall)
        cm.reset()

        # build matrix dictionary
        mxs = {
            f"{name}/recall-matrix-{stage}": mx_recall,
            f"{name}/precision-matrix-{stage}": mx_precision,
            f"{name}/f1-matrix-{stage}": mx_f1,
        }

        # loop over matrices and log
        for key, val in mxs.items():
            val[~val.isfinite()] = 0
            for logger in loggers:
                if isinstance(logger, TensorBoardLogger):
                    self.log_tensorboard(key, val, logger, epoch)
                if isinstance(logger, WandbLogger):
                    self.log_wandb(key, val)


    def log_tensorboard(self, name: str, matrix: Tensor,
                        logger: TensorBoardLogger, epoch: int) -> None:
        """
        Draw and log confusion matrix with Tensorboard

        Args:
            name: Name of confusion matrix
            matrix: Confusion matrix tensor
            logger: Tensorboard logger
            epoch: Training epoch number
        """
        fig = plt.figure(figsize=[8,6])
        sn.heatmap(matrix,
                   xticklabels=self.classes,
                   yticklabels=self.classes,
                   vmin=0, vmax=1,
                   annot=True)
        plt.ylim(0, len(self.classes))
        plt.xlabel('Assigned label')
        plt.ylabel('True label')
        logger.experiment.add_figure(name, fig, global_step=epoch)


    def log_wandb(self, name: str, matrix: Tensor) -> None:
        """
        Draw and log confusion matrix with Weights and Biases

        Args:
            name: Name of confusion matrix
            matrix: Confusion matrix tensor
        """
        table = wandb.Table(columns=["plotly_figure"])
        fig = px.imshow(
            matrix, zmax=1, text_auto=True,
            labels={"x": "Predicted", "y": "True", "color": label},
            x=self.classes, y=self.classes)
        with tempfile.NamedTemporaryFile() as f:
            fig.write_html(f.name, auto_play=False)
            table.add_data(wandb.Html(f.name))
        wandb.log({name: table})
