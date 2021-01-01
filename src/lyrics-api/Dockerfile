# pull official base image
FROM python:3.9.0-slim-buster as base

# set working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update && \
apt-get -y install \
netcat \
gcc && \
apt-get clean

# install python dependencies
COPY . /usr/src/app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# add app
COPY lyrics_api /usr/src/app
RUN python setup.py bdist_wheel

FROM python:3.9.0-slim-buster
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY --from=base /usr/src/app/dist/first_step-0.1.0-py3-none-any.whl /usr/src/app
COPY --from=base /usr/src/app/requirements.txt /usr/src/app
RUN python -m pip install --upgrade pip && \
python -m pip install -r /usr/src/app/requirements.txt && \
python -m pip install /usr/src/app/first_step-0.1.0-py3-none-any.whl

EXPOSE 8000
RUN python -m pip list
ENTRYPOINT ["python"]
CMD ["-m","first_step"]
