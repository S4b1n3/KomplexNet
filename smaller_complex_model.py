import torch
import torch.nn as nn
from opts import parser
from model import Model
from utils import Flatten
import torch.nn.functional as F
import complex_functions as complex_functions
import math
import utils
from torchmetrics.functional import accuracy
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.util import random_noise
from torch.nn.functional import one_hot
from torchmetrics.classification import BinaryF1Score

import pytorch_lightning as pl

args = parser.parse_args()


class SmallModel(Model):

    def __init__(self, in_channels=16, channels=32, kernel_size=3, stride=2, biases=True, num_classes=10, h=5, w=5,
                 epsilon=0, lr_kuramoto=1,
                 mean_r=0, std_r=0.5, lr=args.lr, test_mode=False):
        Model.__init__(self, in_channels, channels, kernel_size, lr)

        self.test_mode = test_mode
        self.mean_r = mean_r
        self.epsilon = epsilon
        self.lr_kuramoto = lr_kuramoto
        self.std_r = std_r
        self.h = h
        self.w = h

        self.out_file = None
        self.id = None

        if args.kuramoto_mode == 'input':
            if args.dataset == 'multi_mnist':
                self.downsampling = complex_functions.ComplexConvolution(1, in_channels, kernel_size, 1, self.padding,
                                                                         biases)
            else:
                self.downsampling = complex_functions.ComplexConvolution(3, in_channels, kernel_size, 1, self.padding,
                                                                         biases)
        elif args.kuramoto_mode == 'endtoend':
            if args.dataset == 'multi_mnist' or args.in_repo == 'multi_mnist_cifar_greyscale':

                if args.gabors:
                    self.downsampling = nn.Conv2d(1, in_channels, kernel_size, 1, padding=5 // 2, bias=False)
                    self.downsampling.weight = nn.Parameter(torch.load('../pt_utils/gabors.pt', map_location='cuda'),
                                                            requires_grad=True)

                else:
                    self.downsampling = nn.Conv2d(1, in_channels, kernel_size, 1, padding=3 // 2, bias=False)

            else:
                self.downsampling = nn.Conv2d(3, in_channels, kernel_size, 1, padding=5 // 2, bias=False)
                self.downsampling.weight = nn.Parameter(
                    torch.load('../pt_utils/gabors.pt', map_location='cuda').repeat(1, 3, 1, 1), requires_grad=True)

        else:
            self.downsampling.weight.requires_grad = False

        self.convA = complex_functions.ComplexConvolution(in_channels, channels, kernel_size, stride, self.padding,
                                                          biases)
        self.classif1 = nn.Sequential(Flatten(), complex_functions.ComplexLinear(channels * 16 * 16, 50, biases))

        self.classif2 = complex_functions.ComplexLinear(50, num_classes, biases, last=True)

        if args.kernel == 'gaussian':
            x, y = torch.meshgrid([torch.linspace(-1, 1, self.h), torch.linspace(-1, 1, self.w)])
            dst = torch.sqrt(x ** 2 + y ** 2)
            g = torch.exp(-((dst - mean_r) ** 2 / (2.0 * std_r ** 2)))
            g = g.unsqueeze(0).unsqueeze(0).repeat(8, 8, 1, 1)
        elif args.kernel == 'random':
            g = torch.rand([8, 8, self.h, self.w])
        elif args.kernel == 'learnt':
            if args.kuramoto_mode == 'input':
                if args.bg_kernel:
                    g = torch.load('./pt_utils/learnt_kernel_bg_input_losscor.pt')
                else:
                    g = torch.load('./pt_utils/learnt_kernel_nobg_input_losscor.pt')
            elif args.kuramoto_mode == 'channels':
                if args.dataset == 'multi_mnist_cifar2':
                    g = torch.load('./pt_utils/learnt_kernel_cifarbg_channels_gabor_bg.pt')
                else:
                    if args.gabors:
                        if args.bg_kernel:
                            g = torch.load('./pt_utils/learnt_kernel_bg_channels_gabor_losscor_15ts.pt')
                        else:
                            g = torch.load('./pt_utils/learnt_kernel_nobg_channels_gabor_losscor_15ts.pt')
                    else:
                        g = torch.load('./pt_utils/learnt_kernel_nobg_channels_losscor.pt')
            else:
                print(args.kernel)
                raise NotImplementedError
        else:
            raise NotImplementedError

        if args.kuramoto_mode == 'endtoend':
            self.kernel_kuramoto = nn.Parameter(g, requires_grad=True)
        else:
            self.kernel_kuramoto = nn.Parameter(g, requires_grad=False)
        self.losses = torch.zeros([39, args.timesteps])
        self.accs = torch.zeros([39, args.timesteps])
        self.binary_accs = torch.zeros([39, args.timesteps])
        self.f1_score = torch.zeros([39, args.timesteps])
        self.synchs = torch.zeros([39, args.timesteps])

        self.epsilon = nn.Parameter(torch.Tensor([epsilon]), requires_grad=False)
        self.lr_kuramoto = nn.Parameter(torch.Tensor([lr_kuramoto]), requires_grad=False)

    def forward(self, input, t, phase_last, mask=None):
        input = input.to(torch.float)

        if args.kuramoto_mode == 'channels' or args.kuramoto_mode == 'endtoend':
            amp = torch.relu(self.downsampling(input))
        elif args.kuramoto_mode == 'input':
            amp = input
        else:
            raise NotImplementedError

        if t == 0:
            phases = (torch.rand_like(amp) * 2 * math.pi) - math.pi
        else:
            phases = phase_last

        if args.phase_mode == 'kuramoto':
            phases = phases + self.update_phases(amp, phases)
        elif args.phase_mode == 'ideal':
            nb_objects = mask.shape[1] - 1 if not args.bg_kernel else mask.shape[1]
            phases = (torch.rand(amp.shape) * 2 * math.pi - math.pi).cuda()
            v = (torch.rand(1) * 2 * math.pi / nb_objects).to(self.device)
            v1 = v
            j = 1 if not args.bg_kernel else 0
            for i in range(nb_objects):
                phases = torch.where(mask[:, j] != 0, v, phases)
                v -= 2 * math.pi / nb_objects
                if v > math.pi:
                    v = v - 2 * math.pi
                if v < -math.pi:
                    v = v + 2 * math.pi
                j += 1
            if args.in_repo == 'object_025':
                phases = torch.where(mask[:, 3] != 0, (v1 - v) / 2, phases)

        z_in = complex_functions.get_complex_number(amp, phases)

        if args.kuramoto_mode == 'input':
            z_in = self.downsampling(z_in)

        outA = self.convA(z_in)

        pred1 = self.classif1(outA)

        pred, last = self.classif2(pred1)

        if self.test_mode:
            return pred, z_in, outA, pred1, last, phases
        else:
            return pred, phases, amp

    def update_phases(self, amp, phases):
        b = torch.tanh(amp)

        B_cos = torch.cos(phases) * b
        B_sin = torch.sin(phases) * b

        C_cos = torch.nn.functional.conv2d(B_cos, self.kernel_kuramoto, padding="same")
        C_sin = torch.nn.functional.conv2d(B_sin, self.kernel_kuramoto, padding="same")

        S_cos = torch.sum(B_cos, dim=(1, 2, 3))[:, None, None, None]
        S_sin = torch.sum(B_sin, dim=(1, 2, 3))[:, None, None, None]

        phases_update = torch.cos(phases) * (C_sin - self.epsilon * S_sin) - torch.sin(phases) * (
                C_cos - self.epsilon * S_cos)
        final_phases = self.lr_kuramoto * phases_update

        return final_phases

    def synch_loss(self, phases, masks, amp):
        if args.bg_kernel:
            masks2 = masks
        else:
            masks2 = masks[:, 1:]
        if args.in_repo == 'object_025':
            masks2 = masks2[:, :-1]

        num_groups = masks2.shape[1]
        group_size = masks2.sum((3, 4))
        group_size = torch.where(group_size == 0, torch.ones_like(group_size), group_size)

        # Loss is at least as large as the maxima of each individual loss (total desynchrony + total synchrony)
        loss_bound = 1 + .5 * num_groups * (1. /
                                            np.arange(1, num_groups + 1) ** 2)[:int(num_groups / 2.)].sum()

        # Consider only the phases with active amplitude
        active_phases = phases * torch.where(amp != 0, torch.ones_like(amp), torch.zeros_like(amp))

        # Calculate global order within each group

        masked_phases = active_phases.unsqueeze(1) * masks2.repeat(1, 1, 8, 1, 1)

        xx = torch.where(masks2.bool(), torch.cos(masked_phases), torch.zeros_like(masked_phases))
        yy = torch.where(masks2.bool(), torch.sin(masked_phases), torch.zeros_like(masked_phases))
        go = torch.sqrt((xx.sum((3, 4))) ** 2 + (yy.sum((3, 4))) ** 2) / group_size
        synch = 1 - go.mean(-1).sum(-1) / num_groups

        # Average angle within a group
        mean_angles = torch.atan2(yy.sum((3, 4)).mean(-1), xx.sum((3, 4)).mean(-1))

        # Calculate desynchrony between average group phases
        desynch = 0
        for m in np.arange(1, int(np.floor(num_groups / 2.)) + 1):
            desynch += (1.0 / (2 * num_groups * m ** 2)) * (
                    torch.cos(m * mean_angles).sum(-1) ** 2 + torch.sin(m * mean_angles).sum(-1) ** 2)

        # Total loss is average of invidual losses, averaged over time
        loss = (synch + desynch) / loss_bound

        return loss.mean(dim=-1), synch.mean(dim=-1), desynch.mean(dim=-1)

    def training_step(self, train_batch, batch_idx):
        setattr(self, 'test_mode', False)

        x, y, m = train_batch

        if x.shape[-1] != 32:
            x = torch.functional.F.interpolate(x, size=(32, 32))
            m = torch.functional.F.interpolate(m.squeeze(), size=(32, 32), mode='nearest-exact')
            m = m.unsqueeze(2)
        phases = None
        loss = 0

        for t in range(args.timesteps):
            logits, phases, amp = self.forward(x.to(self.device), t, phases, m)
            loss += self.bce_loss(logits, y, 'bce')
            loss_last = self.bce_loss(logits, y, 'bce')

        loss_synch, synch, desynch = self.synch_loss(phases, m, amp)

        loss_ce = loss / args.timesteps
        if args.kuramoto_mode == 'endtoend':
            loss = loss_ce + args.loss_coef * loss_synch
        else:
            loss = loss_last

        acc = utils.accuracy_multi(y, torch.sigmoid(logits), display=False, n_obj=args.num_objects)
        binary_acc = self.acc(torch.sigmoid(logits), y)
        f1 = self.f1(torch.sigmoid(logits), y)

        if args.kuramoto_mode == 'endtoend':
            self.log(f"train/loss", loss_ce, on_step=True, on_epoch=True)
        self.log(f"train/loss_all", loss, on_step=True, on_epoch=True)
        self.log(f"train/loss_synch", loss_synch, on_step=True, on_epoch=True)
        self.log(f"train/synch", synch, on_step=True, on_epoch=True)
        self.log(f"train/desynch", desynch, on_step=True, on_epoch=True)
        self.log(f"train/acc", acc, on_step=True, on_epoch=True)
        self.log(f"train/binary_acc", binary_acc, on_step=True, on_epoch=True)
        self.log(f"train/f1", f1, on_step=True, on_epoch=True)

        return loss

    def k_hot_encode(self, array, max_val):
        """k-hot encodes sparse vector of targets
        Array should be N x k for N samples, k targets for each sample"""
        if len(array.shape) == 1:
            array = array.long()
            array = one_hot(array, num_classes=args.num_classes)
            return array.float()
        else:
            b = torch.zeros((array.shape[0], max_val + 1))

            for col in range(array.shape[1]):
                b[torch.arange(array.shape[0]), array[:, col]] = 1
            return b

    def validation_step(self, batch, batch_idx):

        x, y, m = batch

        if x.shape[-1] != 32:
            x = torch.functional.F.interpolate(x, size=(32, 32))
            m = torch.functional.F.interpolate(m.squeeze(), size=(32, 32), mode='nearest-exact')
            m = m.unsqueeze(2)

        phases = None
        loss = 0
        for t in range(args.timesteps):
            logits, phases, amp = self.forward(x.to(self.device), t, phases, m)
            loss += self.bce_loss(logits, y, 'bce')
            loss_last = self.bce_loss(logits, y, 'bce')

        loss_synch, synch, desynch = self.synch_loss(phases, m, amp)

        loss_ce = loss / args.timesteps
        if args.kuramoto_mode == 'endtoend':
            loss = loss_ce + args.loss_coef * loss_synch
        else:
            loss = loss_last

        acc = utils.accuracy_multi(y, torch.sigmoid(logits), display=False, n_obj=args.num_objects)
        binary_acc = self.acc(torch.sigmoid(logits), y)
        f1 = self.f1(torch.sigmoid(logits), y)

        if args.kuramoto_mode == 'endtoend':
            self.log(f"Val_loss", loss_ce, on_step=True, on_epoch=True)
        self.log(f"Val_loss_all", loss, on_step=True, on_epoch=True)
        self.log(f"Val_loss_synch", loss_synch, on_step=True, on_epoch=True)
        self.log(f"Val_synch", synch, on_step=True, on_epoch=True)
        self.log(f"Val_desynch", desynch, on_step=True, on_epoch=True)
        self.log(f"Val_acc", acc, on_step=True, on_epoch=True)
        self.log(f"Val_binary_acc", binary_acc, on_step=True, on_epoch=True)
        self.log(f"Val_f1", f1, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        setattr(self, 'test_mode', True)
        x, y, m = batch

        if x.shape[-1] != 32:
            x = torch.functional.F.interpolate(x, size=(32, 32))
            m = torch.functional.F.interpolate(m.squeeze(), size=(32, 32), mode='nearest-exact')
            m = m.unsqueeze(2)

        if args.add_noise:
            if args.dataset == 'multi_mnist':
                x = torch.tensor(random_noise(x.cpu(), mode='gaussian', mean=0, var=0.1, clip=True), dtype=torch.float)
                # x = torch.tensor(random_noise(x.cpu(), mode = 's&p', amount = 0.05, clip=True), dtype=torch.float)
            else:
                x = torch.tensor(random_noise(x.cpu(), mode='gaussian', mean=0, var=0.05, clip=True), dtype=torch.float)
                # x = torch.tensor(random_noise(x.cpu(), mode = 's&p', amount = 0.05, clip=True), dtype=torch.float)

        phases = None
        loss = 0
        for t in range(args.timesteps):
            logits, z_in, outA, pred1, last, phases = self.forward(x.to(self.device), t, phases, m)

            self.losses[batch_idx, t] = self.bce_loss(logits, y, 'bce')
            self.accs[batch_idx, t] = utils.accuracy_multi(y, torch.sigmoid(logits), display=False,
                                                           n_obj=args.num_objects)
            self.binary_accs[batch_idx, t] = self.acc(torch.sigmoid(logits), y)
            self.f1_score[batch_idx, t] = self.f1(torch.sigmoid(logits), y)
            self.synchs[batch_idx, t], _, _ = self.synch_loss(phases, m, z_in.abs())

            loss += self.bce_loss(logits, y, 'bce')
            loss_last = self.bce_loss(logits, y, 'bce')

        loss_synch, synch, desynch = self.synch_loss(phases, m, z_in.abs())

        loss_ce = loss / args.timesteps
        if args.kuramoto_mode == 'endtoend':
            loss = loss_ce + args.loss_coef * loss_synch
        else:
            loss = loss_last

        acc = utils.accuracy_multi(y, torch.sigmoid(logits), display=False, n_obj=args.num_objects)
        binary_acc = self.acc(torch.sigmoid(logits), y)
        f1 = self.f1(torch.sigmoid(logits), y)

        if args.kuramoto_mode == 'endtoend':
            self.log(f"Test_loss", loss_ce, on_step=True, on_epoch=True)
        self.log(f"Test_loss_all", loss, on_step=True, on_epoch=True)
        self.log(f"Test_acc", acc, on_step=True, on_epoch=True)
        self.log(f"Test_binary_acc", binary_acc, on_step=True, on_epoch=True)
        self.log(f"Test_f1", f1, on_step=True, on_epoch=True)
        self.log(f"Test_loss_synch", loss_synch, on_step=True, on_epoch=True)
        self.log(f"Test_synch", synch, on_step=True, on_epoch=True)
        self.log(f"Test_desynch", desynch, on_step=True, on_epoch=True)

        if batch_idx == 0:
            im1 = self.fig2img(self.plot_phases(x, z_in, outA, pred1, last))
            im3 = self.fig2img(self.plot_phases_l1(z_in))
            self.logger.log_image(key="Phases_per_layer", images=[im1])
            self.logger.log_image(key="Phases_per_channels", images=[im3])
        if batch_idx == 38:
            if self.out_file is not None:
                self.out_file.writerow([self.id, [(t, self.accs.mean(dim=0)[t]) for t in range(args.timesteps)],
                                        [(t, self.f1_score.mean(dim=0)[t]) for t in range(args.timesteps)],
                                        [(t, self.binary_accs.mean(dim=0)[t]) for t in range(args.timesteps)],
                                        [(t, self.synchs.mean(dim=0)[t]) for t in range(args.timesteps)],
                                        ])
            else:
                data = [[x, y] for (x, y) in
                        zip(torch.linspace(0, args.timesteps, args.timesteps + 1), self.accs.mean(dim=0))]
                self.logger.log_table(key="Accuracy per time", data=data, columns=["Timesteps", "Accuracy"])

    def fig2img(self, fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        import io
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

    def plot_phases(self, input, out_l1, out_l2, out_l3, out_l4, out_l4bis=None):
        import seaborn as sns
        sns.set_style('darkgrid')
        img_idx = 2
        cmap = 'hsv'
        NORM = mpl.colors.Normalize(-np.pi, np.pi)
        plt.figure(figsize=(15, 5))
        plt.subplot(151)

        plt.imshow(input[img_idx, 0].cpu().numpy())
        axs = plt.gca()
        axs.title.set_text("Input image")
        axs.set_xticks([])
        axs.set_yticks([])

        idx = 152
        j = 0
        for i in [out_l1, out_l2, out_l3, out_l4]:
            plt.subplot(idx + j, projection='polar')

            to_plot = i
            phase = to_plot.angle()
            magnitude = to_plot.abs()
            color = phase[img_idx]

            plt.scatter(
                phase[img_idx].cpu().numpy(),
                magnitude[img_idx].cpu().numpy(),
                c=color.cpu().numpy(),
                s=100,
                norm=NORM,
                linewidths=0,
                cmap=cmap
            )
            if j == 3 and out_l4bis is not None:
                to_plot = out_l4bis
                phase = to_plot.angle()
                magnitude = to_plot.abs()
                color = phase[img_idx]
                plt.scatter(
                    phase[img_idx].cpu().numpy(),
                    magnitude[img_idx].cpu().numpy(),
                    c=color.cpu().numpy(),
                    s=100,
                    norm=NORM,
                    linewidths=0,
                    cmap=cmap
                )
            axs = plt.gca()
            axs.title.set_text('Layer ' + str(j))
            j += 1

        fig = plt.gcf()
        plt.close()
        return fig

    def plot_phases_l1(self, z_in):
        import seaborn as sns
        sns.set_style('dark')
        NORM = mpl.colors.Normalize(-np.pi, np.pi)
        phases = z_in.angle()[2, :].squeeze().cpu().numpy()
        amp = z_in.abs()[2, :].squeeze().cpu()

        plt.figure(figsize=(20, 20))
        for i in range(8):
            plt.subplot(8, 2, i + 1)
            if torch.max(amp[i]).numpy() == 0:
                alpha = 0
            else:
                alpha = amp[i].numpy() / torch.max(amp[i]).numpy()
            plt.imshow(amp[i])  # , alpha=amp[i].numpy()/torch.max(amp[i]).numpy()
        plt.show()

        # plt.figure(figsize=(20, 20))
        for i in range(8):
            plt.subplot(8, 2, 8 + i + 1)
            if torch.max(amp[i]).numpy() == 0:
                alpha = 0
            else:
                alpha = amp[i].numpy() / torch.max(amp[i]).numpy()
            plt.imshow(phases[i], cmap="hsv", norm=NORM,
                       alpha=alpha)  # , alpha=amp[i].numpy()/torch.max(amp[i]).numpy()
            plt.xticks([])
            plt.yticks([])
        plt.show()

        fig2 = plt.gcf()
        plt.close()
        return fig2
