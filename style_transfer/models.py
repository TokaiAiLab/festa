import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .preprocessing import gram_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

content_layers_default = ["conv_4"]
style_layers_default = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]


class ContentLoss(nn.Module):
    def __init__(self, target: torch.Tensor) -> None:
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.loss = F.mse_loss(inputs, self.target)
        return inputs


class StyleLoss(nn.Module):
    def __init__(self, target_features: torch.Tensor) -> None:
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_features).detach()
        self.loss = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        G = gram_matrix(inputs)
        self.loss = F.mse_loss(G, self.target)
        return inputs


class Normalization(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std


def get_style_model_and_losses(
        vgg: nn.Module,
        norm_mean,
        norm_std,
        style_img,
        content_img,
        content_layer=None,
        style_layers=None):

    if style_layers is None:
        style_layers = style_layers_default
    if content_layer is None:
        content_layer = content_layers_default

    vgg = copy.deepcopy(vgg)
    normalization = Normalization(norm_mean, norm_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
            layer = nn.ReLU(inplace=True)
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError("想定外のレイヤーです。 {}".format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layer:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_los_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_features = model(style_img).detach()
            style_loss = StyleLoss(target_features)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i+1)]

    return model, style_losses, content_losses
