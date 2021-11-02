FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

RUN pip install hydra-core==1.1 mlflow==1.21.0 scipy==1.7.1 pytorch_lightning==1.4.9

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
