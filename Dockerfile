FROM python:latest

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir .
