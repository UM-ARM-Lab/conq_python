#FROM tensorflow/tensorflow:2.1.1-gpu
#FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
FROM ubuntu:22.04

WORKDIR /conq_python

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=America/Detroit
ENV BOSDYN_CLIENT_USERNAME=user
ENV BOSDYN_CLIENT_PASSWORD=4f9y0eftzh76

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# we need python 3.10
RUN apt-get update && \
    apt-get install -y git python3 python3-pip vim tree htop libglib2.0-dev libsm6 libxext6 libxrender-dev libgl1-mesa-glx

# Change the working directory to the project directory
WORKDIR /app/conq_python

# Copy in the code
COPY pyproject.toml .
COPY src src
COPY scripts scripts

# upgrade pip first
RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install .
