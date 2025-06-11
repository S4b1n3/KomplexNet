import torch
import pytorch_lightning as pl
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from torch.utils.data import Dataset, DataLoader, TensorDataset
from opts import parser

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
import torchvision as torchvision
import torchvision.transforms as transforms
from dataloaders import SVHNLoader, MultiMNISTLoader, Dataloader, NORBLoader


args = parser.parse_args()
seed_everything(args.seed)

if args.model == 'small_real':
    from smaller_model import SmallModel as Model
elif args.model == 'small_vit':
    from smaller_vit import SmallModel as Model
elif args.model == 'small_complex':
    from smaller_complex_model import SmallModel as Model
elif args.model == 'small_complex_fb':
    from smaller_complex_model_fb import SmallModelWithFeedback as Model


class MaskedDataset(Dataset):
    def __init__(self, images, labels, masks):
        super().__init__()
        self.data = images
        self.labels = labels
        self.masks = masks

    def __getitem__(self, i):
        return self.data[i], self.labels[i], self.masks[i]

    def __len__(self):
        return len(self.data)


train_loader, val_loader, test_loader = MultiMNISTLoader("../../data/{}/{}/".format(args.dataset, args.in_repo))

path = "./wandb_logs/kuramoto/{}/checkpoints/".format(args.out_repo)
dir_list = os.listdir(path)

print("Files and directories in '", path, "' :")

# prints all files
print(dir_list)
device = torch.device("cpu")
if 'complex' in args.model:

    model = Model.load_from_checkpoint(path + dir_list[0], in_channels=8, channels=args.n_channels, kernel_size=3,
                                       stride=2, biases=True, num_classes=args.num_classes, lr=args.lr,
                                       h=args.h, w=args.w, epsilon=args.epsilon, lr_kuramoto=args.lr_kuramoto,
                                       mean_r=args.mean_r, std_r=args.std_r
                                        , k_l2=args.k_l2,
                                       lr_kuramoto_l2=args.lr_kuramoto_l2, lr_kuramoto_l3=args.lr_kuramoto_l3,
                                       lr_kuramoto_l4=args.lr_kuramoto_l4, map_location=device,
                                       strict=False)

else:
    model = Model.load_from_checkpoint(path + dir_list[0], in_channels=8, channels=args.n_channels, kernel_size=3,
                                       stride=2,
                                       biases=True, num_classes=10, lr=args.lr)

logger = WandbLogger(project="kuramoto", save_dir="./wandb_logs_tests", name=args.filename, log_model=True)
logger.log_hyperparams(args)

trainer = pl.Trainer(
    logger=logger,
    track_grad_norm=2,
    auto_scale_batch_size=True,
    gradient_clip_val=2,
    auto_lr_find=True,
    max_epochs=args.epochs,
    accelerator="auto",
    # precision=16,
    # gpus=[2],
    devices=1 if torch.cuda.is_available() else None,  # [index_1, index_1] for multi gpu
    callbacks=[TQDMProgressBar(refresh_rate=10)])

setattr(model, 'test_mode', True)
trainer.test(model, test_loader)
