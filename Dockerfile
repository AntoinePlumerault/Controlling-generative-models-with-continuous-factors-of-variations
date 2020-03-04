# Base image
FROM nvidia/cuda:10.0-devel-ubuntu18.04

# System packages 
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y apt-utils \
 && apt-get install -y curl \
 && apt-get install -y dvipng \
 && /bin/echo -e "8\n37\n" | apt-get install -y git-all \
 && /bin/echo -e "8\n37\n" | apt-get install -y texlive-latex-extra

# Miniconda installation
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b \
 && rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda \
 && conda init bash \
 && echo "cd code" >> ~\.bashrc \
 && echo "conda activate base" >> ~/.bashrc

# Install matplotlib and Tensorflow
RUN conda install matplotlib \
 && conda install tensorflow-gpu \
 && conda install numba \
 && conda install pytorch torchvision cudatoolkit=10.0 -c pytorch \
 && conda install opencv \
 && pip install tensorflow_hub

EXPOSE 3000

CMD /bin/bash