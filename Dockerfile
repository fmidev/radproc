FROM python:3

WORKDIR /usr/src/app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt upgrade -y && apt install -y libnetcdf-dev && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install -U pip && pip install --no-cache-dir '.[ml,mch]'

ENV PYART_QUIET=1

ENTRYPOINT ["/usr/local/bin/sulatiirain"]
