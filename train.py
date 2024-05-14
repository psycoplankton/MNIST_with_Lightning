import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

@hydra.main(config_path = "configs", config_name="config", version_base=None)
def train(config: DictConfig) -> None:
    data_module = instantiate(config.data_module)
    task = instantiate(config.task)

    #create the logger
    tb_logger = TensorBoardLogger("tb_logs", name='mnist')

    #create the check point callback
    checkpoint_callback = ModelCheckpoint(
        monitor = "validation_accuracy",
        dirpath="checkpoints",
        filename="mnist-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="max",
    )

    #create tbe trainer
    trainer=instantiate(config.trainer, logger=tb_logger, callbacks=[checkpoint_callback])

    #fit the model
    trainer.fit(task, datamodule=data_module)

    #test the model
    trainer.test(data_module)

if __name__ == "__main__":
    train()