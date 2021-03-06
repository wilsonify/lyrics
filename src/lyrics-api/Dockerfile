FROM python:3.9.0-slim-buster as base
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN apt-get update && \
apt-get -y install \
libhdf5-dev \
libc-ares-dev \
libeigen3-dev \
gcc \
gfortran \
g++ \
libgfortran5 \
libatlas3-base \
libatlas-base-dev \
libopenblas-dev \
libopenblas-base \
libblas-dev \
liblapack-dev \
libatlas-base-dev \
openmpi-bin \
libopenmpi-dev \
python3-dev \
netcat

COPY . /usr/src/app
RUN pip install --upgrade pip
RUN python -m pip install /usr/src/app/external/tensorflow-2.4.0-cp39-cp39-linux_x86_64.whl
RUN pip install -r requirements.txt
RUN python setup.py bdist_wheel

FROM python:3.9.0-slim-buster
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN apt-get update && \
apt-get -y install \
libhdf5 \
libc-ares \
libeigen3 \
libgfortran5 \
libatlas3-base \
libatlas-base \
libopenblas \
libopenblas-base \
libblas \
liblapack \
libatlas-base \
openmpi-bin \
libopenmpi-dev

COPY --from=base /usr/src/app/dist/lyrics_api-0.1.0-py3-none-any.whl /usr/src/app
COPY --from=base /usr/src/app/external/python_consumer-0.1.0-py3-none-any.whl /usr/src/app
COPY --from=base /usr/src/app/external/tensorflow_consumer-0.1.0-py3-none-any.whl /usr/src/app
COPY --from=base /usr/src/app/external/tensorflow-2.4.0-cp39-cp39-linux_x86_64.whl /usr/src/app
COPY --from=base /usr/src/app/requirements.txt /usr/src/app

RUN python -m pip install --upgrade pip && \
python -m pip install /usr/src/app/tensorflow-2.4.0-cp39-cp39-linux_x86_64.whl && \
python -m pip install /usr/src/app/python_consumer-0.1.0-py3-none-any.whl && \
python -m pip install /usr/src/app/tensorflow_consumer-0.1.0-py3-none-any.whl && \
python -m pip install -r /usr/src/app/requirements.txt && \
python -m pip install /usr/src/app/lyrics_api-0.1.0-py3-none-any.whl

EXPOSE 8000

ENTRYPOINT ["python"]
CMD ["-m","lyrics_api"]
