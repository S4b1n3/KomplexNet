import torch
import torch.nn as nn
from opts import parser
from model import Model
from smaller_complex_model import SmallModel
from utils import Flatten, Unflatten
import torch.nn.functional as F
import complex_functions as complex_functions
from complex_functions import complex_addition
import math
import utils
from torchmetrics.functional import accuracy
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.util import random_noise

import pytorch_lightning as pl

args = parser.parse_args()


class SmallModelWithFeedback(SmallModel):

    def __init__(self, in_channels=16, channels=32, kernel_size=3, stride=2, biases=True, num_classes=10, h=5, w=5,
                 epsilon=0, lr_kuramoto=1, mean_r=0, std_r=0.5, lr=args.lr, test_mode=False, k_l2=None,
                 lr_kuramoto_l2=0.006, lr_kuramoto_l3=0.006, lr_kuramoto_l4=0.006):
        SmallModel.__init__(self, in_channels, channels, kernel_size, stride, biases, num_classes, h, w, epsilon,
                            lr_kuramoto, mean_r, std_r, lr, test_mode)

        self.h_fb2 = k_l2
        self.w_fb2 = k_l2

        self.lr_kuramoto_l2 = nn.Parameter(torch.Tensor([lr_kuramoto_l2]), requires_grad=False)
        g_l2 = torch.zeros([channels, 8, self.h_fb2, self.w_fb2]) + 1e-6
        self.kernel_kuramoto_l2 = nn.Parameter(g_l2, requires_grad=True)
        self.epsilon_l2 = nn.Parameter(torch.Tensor([0]), requires_grad=False)

        self.lr_kuramoto_l3 = nn.Parameter(torch.Tensor([lr_kuramoto_l3]), requires_grad=False)
        g_l3 = torch.zeros([8,50]) + 1e-6
        self.kernel_kuramoto_l3 = nn.Parameter(g_l3, requires_grad=True)
        self.epsilon_l3 = nn.Parameter(torch.Tensor([0]), requires_grad=False)

        g_l4 = torch.zeros([8, num_classes]) + 1e-6
        self.lr_kuramoto_l4 = nn.Parameter(torch.Tensor([lr_kuramoto_l4]), requires_grad=False)
        self.kernel_kuramoto_l4 = nn.Parameter(g_l4, requires_grad=True)
        self.epsilon_l4 = nn.Parameter(torch.Tensor([0]), requires_grad=False)

        self.losses = torch.zeros([39, args.timesteps])
        self.accs = torch.zeros([39, args.timesteps])
        self.b_accs = torch.zeros([39, args.timesteps])
        self.f1s = torch.zeros([39, args.timesteps])
        self.synchs = torch.zeros([39, args.timesteps])

    def forward(self, input, t, phases, outA_last, pred1_last, pred2_last, mask=None):
        input = input.to(torch.float)

        if args.kuramoto_mode == 'channels' or args.kuramoto_mode == 'endtoend':
            amp = torch.relu(self.downsampling(input))
        elif args.kuramoto_mode == 'input':
            amp = input
        else:
            raise NotImplementedError

        if t == 0:
            phases = (torch.rand_like(amp) * 2 * math.pi) - math.pi
            phases = phases + self.update_phases(amp, phases)
            z_in = complex_functions.get_complex_number(amp, phases)

            if args.kuramoto_mode == 'input':
                z_in = self.downsampling(z_in)

            outA = self.convA(z_in)
            pred1 = self.classif1(outA)

            pred, last = self.classif2(pred1)
        else:
            phases = phases

            if not args.random:
                phases = phases + self.update_phases(amp, phases) \
                                + self.update_phases_froml3(pred2_last.abs(), phases, complex_functions.stable_angle(pred2_last), self.kernel_kuramoto_l4, self.lr_kuramoto_l4) \
                                + self.update_phases_froml3(pred1_last.abs(), phases, complex_functions.stable_angle(pred1_last), self.kernel_kuramoto_l3, self.lr_kuramoto_l3) \
                                + self.update_phases_froml2(outA_last.abs(), phases, outA_last.angle())

            z_in = complex_functions.get_complex_number(amp, phases)

            if args.kuramoto_mode == 'input':
                z_in = self.downsampling(z_in)

            outA = self.convA(z_in)
            pred1 = self.classif1(outA)

            pred, last = self.classif2(pred1)

        if self.test_mode:
            return pred, z_in, outA, pred1, last, phases
        else:
            return pred, phases, amp, outA, pred1, last

    def update_phases_froml2(self, amp, phases_l1, phases_l2):

        b = torch.tanh(amp)  # 16x16

        B_cos = torch.cos(phases_l2) * b  # 16x16
        B_sin = torch.sin(phases_l2) * b  # 16x16

        C_cos = torch.nn.functional.conv_transpose2d(B_cos, self.kernel_kuramoto_l2, stride=2,
                                                     padding=self.h_fb2//2, output_padding=1)
        C_sin = torch.nn.functional.conv_transpose2d(B_sin, self.kernel_kuramoto_l2, stride=2,
                                                     padding=self.h_fb2//2, output_padding=1)

        S_cos = torch.sum(B_cos, dim=(1, 2, 3))[:, None, None, None]
        S_sin = torch.sum(B_sin, dim=(1, 2, 3))[:, None, None, None]

        phases_update = torch.cos(phases_l1) * (C_sin - self.epsilon_l2 * S_sin) - torch.sin(phases_l1) * (
                C_cos - self.epsilon_l2 * S_cos)
        final_phases = self.lr_kuramoto_l2 * phases_update

        return final_phases

    def update_phases_froml3(self, amp, phases_l1, phases_l3, kernel, lr_kuramoto):

        b = torch.tanh(amp)

        phases_update = torch.zeros_like(phases_l1)
        for j in range(8):
            phases_update[:,j] = (kernel[j,:,None,None]*torch.sin(phases_l3[:,:,None, None] - phases_l1[:,j].unsqueeze(1))*b[:,:,None, None]).sum(dim=1)

        final_phases = lr_kuramoto * phases_update

        return final_phases

    def training_step(self, train_batch, batch_idx):
        setattr(self, 'test_mode', False)
        x, y, m = train_batch

        if x.shape[-1] != 32:
            x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=True)
            m = F.interpolate(m.squeeze(), size=(32, 32), mode='nearest-exact').unsqueeze(2)

        phases, outA_last, pred1_last, pred2_last = None, None, None, None
        loss = 0
        for t in range(args.timesteps):
            logits, phases, amp, outA_last, pred1_last, pred2_last = self.forward(x.to(self.device), t, phases,
                                                                            outA_last, pred1_last, pred2_last, m)
            loss += self.bce_loss(logits, y, 'bce')
        loss_synch, synch, desynch = self.synch_loss(phases, m, amp)

        loss_ce = loss / args.timesteps
        loss = loss_ce + args.loss_coef * loss_synch

        b_acc = self.acc(torch.sigmoid(logits), y)
        acc = utils.accuracy_multi(y, torch.sigmoid(logits), display=False, n_obj=args.num_objects)
        f1 = self.f1(torch.sigmoid(logits), y)

        self.log(f"train/loss_ce", loss_ce, on_step=True, on_epoch=True)
        self.log(f"train/loss", loss, on_step=True, on_epoch=True)
        self.log(f"train/acc", acc, on_step=True, on_epoch=True)
        self.log(f"train/b_acc", b_acc, on_step=True, on_epoch=True)
        self.log(f"train/f1", f1, on_step=True, on_epoch=True)
        self.log(f"train/loss_synch", loss_synch, on_step=True, on_epoch=True)
        self.log(f"train/synch", synch, on_step=True, on_epoch=True)
        self.log(f"train/desynch", desynch, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y, m = batch

        if x.shape[-1] != 32:
            x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=True)
            m = F.interpolate(m.squeeze(), size=(32, 32), mode='nearest-exact').unsqueeze(2)

        phases, outA_last, pred1_last, pred2_last = None, None, None, None
        loss = 0
        for t in range(args.timesteps):
            logits, phases, amp, outA_last, pred1_last, pred2_last = self.forward(x.to(self.device), t, phases,
                                                                                  outA_last, pred1_last, pred2_last, m)
            loss += self.bce_loss(logits, y, 'bce')
        loss_synch, synch, desynch = self.synch_loss(phases, m, amp)

        loss_ce = loss / args.timesteps
        loss = loss_ce + args.loss_coef * loss_synch

        b_acc = self.acc(torch.sigmoid(logits), y)
        acc = utils.accuracy_multi(y, torch.sigmoid(logits), display=False, n_obj=args.num_objects)
        f1 = self.f1(torch.sigmoid(logits), y)

        self.log(f"Val_loss_ce", loss_ce, on_step=True, on_epoch=True)
        self.log(f"Val_loss", loss, on_step=True, on_epoch=True)
        self.log(f"Val_acc", acc, on_step=True, on_epoch=True)
        self.log(f"Val_b_acc", b_acc, on_step=True, on_epoch=True)
        self.log(f"Val_f1", f1, on_step=True, on_epoch=True)
        self.log(f"Val_loss_synch", loss_synch, on_step=True, on_epoch=True)
        self.log(f"Val_synch", synch, on_step=True, on_epoch=True)
        self.log(f"Val_desynch", desynch, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        setattr(self, 'test_mode', True)
        x, y, m = batch

        if x.shape[-1]!=32:
            x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=True)
            m = F.interpolate(m.squeeze(), size=(32, 32), mode='nearest-exact').unsqueeze(2)

        if args.add_noise:
            if args.dataset == 'multi_mnist':
                x = torch.tensor(random_noise(x.cpu(), mode='gaussian', mean=0, var=0.1, clip=True), dtype=torch.float)
                # x = torch.tensor(random_noise(x.cpu(), mode = 's&p', amount = 0.05, clip=True), dtype=torch.float)
            else:
                x = torch.tensor(random_noise(x.cpu(), mode='gaussian', mean=0, var=0.01, clip=True), dtype=torch.float)
                # x = torch.tensor(random_noise(x.cpu(), mode = 's&p', amount = 0.05, clip=True), dtype=torch.float)


        phases, outA, pred1, last = None, None, None, None
        loss = 0
        for t in range(args.timesteps):
            logits, z_in, outA, pred1, last, phases = self.forward(x.to(self.device), t, phases, outA, pred1, last, m)
            self.losses[batch_idx, t] = self.bce_loss(logits, y, 'bce')
            loss += self.bce_loss(logits, y, 'bce')
            self.b_accs[batch_idx, t] = self.acc(torch.sigmoid(logits), y)
            self.accs[batch_idx, t] = utils.accuracy_multi(y, torch.sigmoid(logits), display=False, n_obj=args.num_objects)
            self.f1s[batch_idx, t] = self.f1(torch.sigmoid(logits), y)
            self.synchs[batch_idx, t], _, _ = self.synch_loss(phases, m, z_in.abs())

        loss_synch, synch, desynch = self.synch_loss(phases, m, z_in.abs())

        loss_ce = loss / args.timesteps
        loss = loss_ce + args.loss_coef * loss_synch
        b_acc = self.acc(torch.sigmoid(logits), y)
        acc = utils.accuracy_multi(y, torch.sigmoid(logits), display=False, n_obj=args.num_objects)
        f1 = self.f1(torch.sigmoid(logits), y)

        self.log(f"Test_loss_ce", loss_ce, on_step=True, on_epoch=True)
        self.log(f"Test_loss", loss, on_step=True, on_epoch=True)
        self.log(f"Test_acc", acc, on_step=True, on_epoch=True)
        self.log(f"Test_b_acc", b_acc, on_step=True, on_epoch=True)
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
                                        [(t, self.f1s.mean(dim=0)[t]) for t in range(args.timesteps)],
                                        [(t, self.b_accs.mean(dim=0)[t]) for t in range(args.timesteps)],
                                        [(t, self.synchs.mean(dim=0)[t]) for t in range(args.timesteps)]
                                        ])
            else:
                data = [[x, y] for (x, y) in zip(torch.linspace(0,args.timesteps,args.timesteps+1), self.accs.mean(dim=0))]
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
        img_idx = 2
        cmap = 'hsv'
        NORM = mpl.colors.Normalize(-np.pi, np.pi)
        plt.figure(figsize=(10, 5))
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
            plt.imshow(amp[i], alpha=alpha)
        plt.show()

        for i in range(8):
            plt.subplot(8, 2, 8 + i + 1)
            if torch.max(amp[i]).numpy() == 0:
                alpha = 0
            else:
                alpha = amp[i].numpy() / torch.max(amp[i]).numpy()
            plt.imshow(phases[i], cmap="hsv", norm=NORM,
                       alpha=alpha)
        plt.show()

        fig2 = plt.gcf()
        plt.close()
        return fig2
