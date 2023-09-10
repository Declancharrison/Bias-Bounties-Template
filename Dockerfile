FROM ubuntu:22.04

RUN mkdir -p home/tmp

RUN apt update
RUN apt install -y python3
RUN apt install -y pip

COPY requirements.txt home/container_tmp/requirements.txt
COPY data/training_data.npy home/container_tmp/training_data.npy
COPY security.py home/container_tmp/security.py
COPY bad_argvals.txt home/container_tmp/bad_argvals.txt

RUN pip install -r home/container_tmp/requirements.txt

