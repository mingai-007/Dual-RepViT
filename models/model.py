import torch.nn as nn
from torchvision import models
from thop import profile  # For calculating FLOPs and parameters
import time

from pytorch_wavelets import DWTForward  # Discrete Wavelet Transform

from timm.models.layers import SqueezeExcite
from timm.models.layers.create_act import create_act_layer
import torch

class Conv2d_BN(torch.nn.Sequential):
    """
    Conv2d followed by BatchNorm2d, with optional weight initialization and fuse ability.
    """
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        """
        Fuse Conv2d and BatchNorm2d for efficient inference.
        """
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Residual(torch.nn.Module):
    """
    Standard residual connection module, with optional stochastic depth.
    """
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        # Apply stochastic depth in training, skip during inference
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        """
        Fuse if possible, else return self.
        """
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self

class sRepViTBlock(nn.Module):
    """
    The core block of sRepViT, consisting of a token mixer and a channel mixer.
    """
    def __init__(self, inp, oup, kernel_size, stride, use_se):
        super(sRepViTBlock, self).__init__()
        assert stride in [1, 2]
        # Token mixer: depthwise conv + optional Squeeze-and-Excitation + pointwise conv
        self.token_mixer = nn.Sequential(
            Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
            SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
        )
        # Channel mixer: residual connection with GELU activation
        self.channel_mixer = Residual(nn.Sequential(
            Conv2d_BN(oup, oup, 1, 1, 0),
            nn.GELU()
        ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

from timm.models.vision_transformer import trunc_normal_

class BN_Linear(torch.nn.Sequential):
    """
    BatchNorm1d followed by Linear layer, with truncated normal initialization and fuse ability.
    """
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        """
        Fuse BatchNorm1d and Linear for efficient inference.
        """
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Classfier(nn.Module):
    """
    Classification head, can be BN_Linear or Identity.
    """
    def __init__(self, dim, num_classes):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        x = self.classifier(x)
        return x

    @torch.no_grad()
    def fuse(self):
        classifier = self.classifier.fuse()
        return classifier

class Down_wt(nn.Module):
    """
    Downsampling module using Discrete Wavelet Transform (DWT).
    """
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')  # Single-level Haar DWT
        self.conv_bn_relu = nn.Sequential(
                                    Conv2d_BN(in_ch*4, out_ch, ks=1, stride=1),
                                    nn.ReLU(inplace=True),
                                    # nn.Conv2d(in_ch * 2, out_ch, kernel_size=1, stride=1),
                                    # nn.BatchNorm2d(out_ch),
                                    )
    def forward(self, x):
        # Apply DWT: returns low-frequency and high-frequency components
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        # Concatenate low-frequency and all high-frequency subbands
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

class Dual_RepViT(nn.Module):
    """
    The main Dual-RepViT architecture for efficient tuberculosis detection.
    Integrates wavelet-based downsampling and sRepViT blocks.
    """
    def __init__(self, num_classes=1):
        super(Dual_RepViT, self).__init__()

        # First and second wavelet downsampling
        self.dw1 = Down_wt(1,20)
        self.dw2 = Down_wt(20, 40)

        # First and second sRepViT main branches
        self.rep1 = sRepViTBlock(1, 20, 3, 2, 1)
        self.rep2 = sRepViTBlock(20, 40, 3, 2, 1)

        # Feature fusion convolutions
        self.conv1 = nn.Sequential(Conv2d_BN(40, 20, 1), nn.ReLU())
        self.conv2 = nn.Sequential(Conv2d_BN(80, 40, 1), nn.ReLU())

        # Classification head
        self.classifier = Classfier(40, num_classes)

    def forward(self, x):
        # First downsampling branch
        y1 = self.dw1(x)
        # First main branch
        z1 = self.rep1(x)
        # Concatenate and fuse
        m1 = torch.cat([y1,z1],dim=1)
        m1 = self.conv1(m1)

        # Second downsampling and main branches
        y2 = self.dw2(m1)
        z2 = self.rep2(m1)
        m2 = torch.cat([y2, z2], dim=1)
        m2 = self.conv2(m2)

        # Global average pooling and classification
        out = torch.nn.functional.adaptive_avg_pool2d(m2, 1).flatten(1)
        out = self.classifier(out)
        return out

if __name__ == '__main__':
    # Device selection for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate random input data
    x = torch.randn(1, 1, 256, 256).to(device)
    model = Dual_RepViT(num_classes=1).to(device)
    model.eval()

    # ------------------ Warm-up ------------------
    # Warm up to avoid cold start effect in timing
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)

    # ------------------ GPU Timing ------------------
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    timings = []
    repetitions = 100

    # Measure average inference time
    with torch.no_grad():
        for _ in range(repetitions):
            starter.record()
            _ = model(x)
            ender.record()

            torch.cuda.synchronize()
            elapsed_time_ms = starter.elapsed_time(ender)
            timings.append(elapsed_time_ms)

    avg_time = sum(timings) / repetitions
    print(f"\n✅ Average inference time ({repetitions} runs): {avg_time:.2f} ms")
    # ------------------ FLOPs & Parameters ------------------
    flops, params = profile(model, inputs=(x,))
    print(f"Output shape : {_.shape}")
    print(f"Total params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"FLOPs        : {flops:,}")
    print(f"Parameters   : {params:,}")
