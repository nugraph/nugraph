"""Utility class for logging confusion matrices"""
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


    def log(self, name: str, matrix: tm.ConfusionMatrix,
            logger: Logger | list[Logger], epoch: int) -> None:
        """
        Log confusion matrix

        Args:
            name: Name of confusion matrix
            matrix: Torchmetrics confusion matrix
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

        for logger in loggers:
            if isinstance(logger, TensorBoardLogger):
                self.log_tensorboard(name, matrix, logger, epoch)
            if isinstance(logger, WandbLogger):
                self.log_wandb(name, matrix)

        matrix.reset()


    def log_tensorboard(self, name: str, matrix: tm.ConfusionMatrix,
                        logger: TensorBoardLogger, epoch: int) -> None:
        """
        Draw and log confusion matrix with Tensorboard

        Args:
            name: Name of confusion matrix
            matrix: Torchmetrics confusion matrix
            logger: Tensorboard logger
            epoch: Training epoch number
        """
        cm = matrix.compute().cpu()
        fig = plt.figure(figsize=[8,6])
        sn.heatmap(cm,
                   xticklabels=self.classes,
                   yticklabels=self.classes,
                   vmin=0, vmax=1,
                   annot=True)
        plt.ylim(0, len(self.classes))
        plt.xlabel('Assigned label')
        plt.ylabel('True label')
        logger.experiment.add_figure(name, fig, global_step=epoch)


    def log_wandb(self, name: str, matrix: tm.ConfusionMatrix) -> None:
        """
        Draw and log confusion matrix with Weights and Biases

        Args:
            name: Name of confusion matrix
            matrix: Torchmetrics confusion matrix
        """
        cm = matrix.compute().cpu()
        table = wandb.Table(columns=["plotly_figure"])
        fig = px.imshow(
            cm, zmax=1, text_auto=True,
            labels={"x": "Predicted", "y": "True", "color": label},
            x=self.classes, y=self.classes)
        with tempfile.NamedTemporaryFile() as f:
            fig.write_html(f.name, auto_play=False)
            table.add_data(wandb.Html(f.name))
        wandb.log({name: table})
