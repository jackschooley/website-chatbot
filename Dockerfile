# syntax=docker/dockerfile:1
FROM continuumio/miniconda3
ENV PYTHONUNBUFFERED=1
WORKDIR /code/
COPY requirements.txt /code/
RUN pip install -r requirements.txt
RUN conda install pytorch cpuonly -c pytorch-lts
RUN conda install -c conda-forge transformers=4.10.0
COPY . /code/