FROM tensorflow/tensorflow
ENV AMQP_HOST "localhost"
ENV AMQP_PORT "5672"
ADD . /src/tensorflow-consumer
WORKDIR /src/tensorflow-consumer
RUN pip install -r requirements.txt
RUN python setup.py install
ENTRYPOINT ["python"]
CMD ["-m", "python_consumer"]