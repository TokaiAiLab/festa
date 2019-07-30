# Festa

## Run in Colab
You can run `style_transfer`.
Please open [Jupyter Notebook](Neural_Style_Transfer.ipynb) and click `Open in Colab badge`.

Note, You need Google Account, if you want to run in Colab.

## Installation
Install command is here.
I recommend you using virtual environment such as Anaconda, Pipenv, etc.

```shell script
git clone https://github.com/KawashimaHirotaka/festa.git
cd festa/
pip install .
```

## Docker Image
Docker Image is available via building from Dockerfile in this repository.

```shell script
git clone https://github.com/KawashimaHirotaka/festa.git
cd festa/
docker build -t festa .
```

## Reference
Most of codes in `style_transfer`.
I referred to 
[Neural Transfer Using PyTorch in PyTorch Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html).

Those authors are Alexis Jacq, and Winston Herring.
