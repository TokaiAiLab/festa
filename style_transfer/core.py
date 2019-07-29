import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from . import models
from . import preprocessing


def get_input_optimizer(input_img: torch.Tensor) -> torch.optim.Optimizer:
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(
        cnn,
        normalization_mean,
        normalization_std,
        content_img,
        style_img,
        input_img,
        num_steps=300,
        style_weight=1e6,
        content_weight=1.0):

    model, style_losses, content_losses = \
        models.get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img)

    optimizer = get_input_optimizer(input_img)

    run = [0]
    while run[0] <= num_steps:
        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img


def run(content_img, style_img):
    output = run_style_transfer(
        models.cnn,
        models.cnn_normalization,
        models.cnn_normalization_std,
        content_img,
        style_img,
        content_img.clone()
    )

    plt.figure(figsize=(15, 10), dpi=200)
    preprocessing.img_show(output)
    plt.show()
