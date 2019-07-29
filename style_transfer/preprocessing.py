import os
import torch
import requests
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from .models import device

img_size = 512 if torch.cuda.is_available() else 128


loader = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])

unloader = transforms.ToPILImage()


def load_img(img_name: str) -> torch.Tensor:
    img = Image.open(img_name)
    img = loader(img)
    return img.to(device, torch.float)


def img_show(tensor: torch.Tensor, title: str = None) -> None:
    img = tensor.cpu().clone()
    img = img.squeeze(0)
    img = unloader(img)
    plt.imshow(img)

    if title is not None:
        plt.title(title)


def fetch_img(url: str) -> None:
    img = requests.get(url)
    file_name = os.path.basename(url)

    with open("images/{}".format(file_name), "wb") as f:
        f.write(img.content)


def gram_matrix(inputs: torch.Tensor) -> torch.Tensor:
    a, b, c, d = inputs.shape
    features = inputs.view(a*b, c*d)
    G = torch.mm(features, features.t())

    return torch.div(G, a*b*c*d)

