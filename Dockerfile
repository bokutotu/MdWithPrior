FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

RUN pip install hydra-core mlflow

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
