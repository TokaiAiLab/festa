FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

WORKDIR /tmp
RUN git clone https://github.com/KawashimaHirotaka/festa.git && cd festa && pip install .

WORKDIR /workspace
