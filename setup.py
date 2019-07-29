from setuptools import setup, find_packages


setup(
    name='festa',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "Pillow",
        "matplotlib",
        "torch",
        "torchvision",
        "requests",
    ],
    url='',
    license='None',
    author='Kawashima Hirotaka',
    author_email='',
    description=''
)
