FROM python:latest

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir .

ENV PYART_QUIET=1


