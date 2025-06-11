import tempfile

import torch
import pytorch_lightning as pl
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["WANDB_DIR"] = "./"

import utils
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
import torchvision as torchvision
import torchvision.transforms as transforms
from opts import parser
import numpy as np

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloaders import SVHNLoader, MultiMNISTLoader, Dataloader, NORBLoader
import wandb

wandb.login()
#wandb.init(dir=tempfile.gettempdir())

args = parser.parse_args()
# seed_everything(args.seed)

if args.model == 'small_real' :
    from smaller_model import SmallModel as Model
elif args.model == 'small_complex' :
    # from smaller_complex_model_fb import SmallModelWithFeedback as Model
    from smaller_complex_model import SmallModel as Model
elif args.model == 'small_vit':
    from smaller_vit import SmallModel as Model

train_loader, val_loader, test_loader = MultiMNISTLoader("../../data/{}/{}/".format(args.dataset, args.in_repo))

if 'complex' in args.model:
    model = Model(8, args.n_channels, 3, 2, True, args.num_classes, h=13, w=13, epsilon=args.epsilon,
                  lr_kuramoto=args.lr_kuramoto, mean_r=args.mean_r, std_r=args.std_r)
else:
    model = Model(8, args.n_channels, 3, 2, True, args.num_classes)

print(model)

#logger = TensorBoardLogger('tb_logs', name=args.filename)
logger = WandbLogger(project="kuramoto", save_dir="./wandb_logs", name=args.filename, log_model='all')
logger.log_hyperparams(args)
checkpoint_callback = ModelCheckpoint(monitor="Val_acc", mode="max")

trainer = pl.Trainer(
    logger=logger,
    gradient_clip_val=2,
    max_epochs=args.epochs,
    accelerator="auto",
    #gpus=[2],
    devices=1 if torch.cuda.is_available() else None, #[index_1, index_1] for multi gpu
    callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback])

if args.resume:
    path = "./wandb_logs/kuramoto/{}/checkpoints/".format(args.out_repo)
    dir_list = os.listdir(path)

    print("Files and directories in '", path, "' :")

    # prints all files
    print(dir_list)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=path + dir_list[0])

else:
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

trainer.test(model, test_loader)
wandb.finish()
