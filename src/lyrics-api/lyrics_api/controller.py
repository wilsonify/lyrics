"""
REST api using FastAPI.

Create an app instance.
Run the development server

With FastAPI, by using short, intuitive and standard Python type declarations, you get:

    Editor support: error checks, autocompletion, etc.
    Data "parsing"
    Data validation
    API annotation and automatic documentation

And you only have to declare them once.

That's probably the main visible advantage of FastAPI compared to alternative frameworks
(apart from the raw performance).
"""
import json
import logging
import uuid

import pika
from fastapi import FastAPI, HTTPException
from lyrics_api import __version__, AMQP_HOST, AMQP_PORT, AMQP_PASSWORD, AMQP_USERNAME
from lyrics_api.model import Phoneme, Grapheme, SpanishGrapheme, EnglishGrapheme


class RemoteProcedure():
    """
    sending and receiving results of a remote procedure call
    """
    credentials = pika.PlainCredentials(
        username=AMQP_USERNAME,
        password=AMQP_PASSWORD
    )
    connection_parameters = pika.ConnectionParameters(
        host=AMQP_HOST,
        port=AMQP_PORT,  # default is 5672
        credentials=credentials,
        heartbeat=10,
        blocked_connection_timeout=100,
    )

    def __init__(self, routing_key):
        self.connection = pika.BlockingConnection(parameters=self.connection_parameters)
        self.channel = self.connection.channel()
        self.channel.basic_qos(prefetch_count=1)
        self.routing_key = routing_key
        self.corr_id = str(uuid.uuid4())
        self.response = None

        new_queue_method_frame = self.channel.queue_declare(queue='', exclusive=True)
        self.reply_to_queue = new_queue_method_frame.method.queue
        self.properties = pika.BasicProperties(
            reply_to=self.reply_to_queue,
            correlation_id=self.corr_id,
        )
        self.declare()

    def declare(self):
        """
        make sure exchange and queue exist
        """
        route = "python"
        self.channel.exchange_declare(exchange=f"try_{route}", exchange_type="topic")
        self.channel.queue_declare(f"try_{route}", durable=True, exclusive=False, auto_delete=False)
        self.channel.queue_bind(queue=f"try_{route}", exchange=f"try_{route}", routing_key=route)
        route = "tensorflow"
        self.channel.exchange_declare(exchange=f"try_{route}", exchange_type="topic")
        self.channel.queue_declare(f"try_{route}", durable=True, exclusive=False, auto_delete=False)
        self.channel.queue_bind(queue=f"try_{route}", exchange=f"try_{route}", routing_key=route)

    def on_response(self, channel, method, props, body):
        """
        what to do when you get a repsponse
        """
        logging.debug("%r", f"ch={channel}")
        logging.debug("%r", f"ch={method}")

        if self.corr_id == props.correlation_id:
            self.response = json.loads(body.decode("utf-8"))

    def call(self, body_new):
        """
        send message wait for response
        """
        logging.debug("%r", f"body_new = {body_new}")

        self.channel.basic_publish(
            exchange=f'try_{self.routing_key}',
            routing_key=self.routing_key,
            body=json.dumps(body_new).encode("utf-8")
        )

        self.channel.basic_consume(
            queue=self.reply_to_queue,
            on_message_callback=self.on_response,
            auto_ack=True
        )

        while self.response is None:
            self.connection.process_data_events()
        return self.response


app = FastAPI(
    debug=False,
    title="lyrics",
    description="access functionality of various consumers",
    version=__version__,
    openapi_url="/openapi.json",
    openapi_tags=None,  # : Optional[List[Dict[str, Any]]]
    servers=None,  # : Optional[List[Dict[str, Union[str, Any]]]]
    dependencies=None,  # : Optional[Sequence[Depends]]
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.post(
    path="/grapheme2phoneme",
    response_model=Phoneme,
    summary="Summary: convert graphemes to phonemes",
    description="""Description:
    convert graphemes (smallest functional unit of a writing system) to phonemes (perceptually distinct units of sound)
    """,
)
async def grapheme2phoneme(input_grapheme: Grapheme):
    """
    convert a grapheme to a phoneme
    """


@app.post(
    path="/phoneme2grapheme",
    response_model=Grapheme,
    summary="Summary: convert phonemes to graphemes",
    description="""Description:
    convert phonemes (perceptually distinct units of sound) to graphemes (smallest functional unit of a writing system)
    """
)
async def phoneme2grapheme(input_phoneme: Phoneme):
    """
    convert a phoneme to grapheme
    """
    request_body = {
        'strategy': "phoneme2grapheme",
        'text': input_phoneme.text
    }
    rpc = RemoteProcedure(routing_key='python')
    response_body = rpc.call(request_body)
    status_code = response_body['status_code']
    if response_body['status_code'] != 200:
        status_detail = response_body['detail']
        raise HTTPException(status_code=status_code, detail=status_detail)
    return EnglishGrapheme(
        text=response_body['output']
    )


@app.post(
    path="/preprocess-sentence",
    response_model=Grapheme,
    summary="Summary: preprocess a sentence",
    description="""Description:
    create a space between a word and the punctuation following it eg: "he is a boy." => "he is a boy ."    
    replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    add a start and an end token to the sentence so that a model know when to start and stop predicting.
    """
)
async def preprocess_sentence(input_grapheme: Grapheme):
    """
    preprocess a sentence
    """

    request_body = {
        'strategy': "preprocess_sentence",
        'text': input_grapheme.text
    }
    rpc = RemoteProcedure(routing_key='python')
    response_body = rpc.call(request_body)
    status_code = response_body['status_code']
    if response_body['status_code'] != 200:
        status_detail = response_body['detail']
        raise HTTPException(status_code=status_code, detail=status_detail)
    return EnglishGrapheme(
        text=response_body['output']
    )


@app.post(
    path="/translate-spanish-to-english",
    response_model=EnglishGrapheme,
    summary="Summary: translate a spanish sentence to english",
    description="""Description:
    Uses Neural machine translation with attention to translate spanish to english    
    """
)
async def translate_spanish_to_english(input_grapheme: SpanishGrapheme):
    """
    translate spanish to english
    """

    request_body = {
        'strategy': "spanish_to_english",
        'text': input_grapheme.text
    }
    rpc = RemoteProcedure(routing_key='python')
    response_body = rpc.call(request_body)
    status_code = response_body['status_code']
    if response_body['status_code'] != 200:
        status_detail = response_body['detail']
        raise HTTPException(status_code=status_code, detail=status_detail)
    return EnglishGrapheme(
        text=response_body['output']
    )


@app.post(
    path="/translate-english-to-spanish",
    response_model=SpanishGrapheme,
    summary="Summary: translate an english sentence to spanish",
    description="""Description:
    Uses Neural machine translation with attention to translate english to spanish     
    """
)
async def translate_english_to_spanish(input_grapheme: EnglishGrapheme):
    """
    translate english to spanish
    """

    request_body = {
        'strategy': "english_to_spanish",
        'text': input_grapheme.text
    }
    rpc = RemoteProcedure(routing_key='python')
    response_body = rpc.call(request_body)
    status_code = response_body['status_code']
    if response_body['status_code'] != 200:
        status_detail = response_body['detail']
        raise HTTPException(status_code=status_code, detail=status_detail)
    return EnglishGrapheme(
        text=response_body['output']
    )


if __name__ == "__main__":
    print("I'm the main")
