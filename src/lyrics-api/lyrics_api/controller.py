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

from fastapi import FastAPI, HTTPException
from lyrics_api import __version__
from lyrics_api.model import Phoneme, Grapheme, SpanishGrapheme, EnglishGrapheme
from tensorflow_consumer.translation import nmt
from tensorflow_consumer.translation.eng2spa import eng2spa_translate
from tensorflow_consumer.translation.spa2eng import spa2eng_translate
from tensorflow_consumer.translation.phoneme2grapheme import phoneme2grapheme_translate
import pika
import uuid


class RemoteProcedure():
    def __init__(self, routing_key):
        self.host = 'localhost'
        self.port = 32777  # 5672
        self.credentials = pika.PlainCredentials(username='guest', password='guest')
        self.connection_parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            credentials=self.credentials,
            heartbeat=10,
            blocked_connection_timeout=100,
        )
        self.connection = pika.BlockingConnection(parameters=self.connection_parameters)
        self.channel = self.connection.channel()
        self.channel.basic_qos(prefetch_count=1)
        self.corr_id = str(uuid.uuid4())
        self.response = None
        self.routing_key = routing_key

        new_queue_method_frame = self.channel.queue_declare(queue='', exclusive=True)
        self.properties = pika.BasicProperties(
            reply_to=new_queue_method_frame.method.queue,
            correlation_id=self.corr_id,
        )

        self.channel.basic_consume(
            queue=new_queue_method_frame.method.queue,
            on_message_callback=self.on_response,
            auto_ack=True
        )

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = json.loads(body.decode("utf-8"))

    def call(self, body_new):

        self.channel.basic_publish(
            exchange=f'try_{self.routing_key}',
            routing_key=self.routing_key,
            properties=self.properties,
            body=json.dumps(body_new).encode("utf-8")
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
    request_body = {
        'strategy': "grapheme2phoneme",
        'text': input_grapheme.text
    }
    rpc = RemoteProcedure(routing_key='python')
    response_body = rpc.call(request_body)
    status_code = response_body['status_code']
    if response_body['status_code'] != 200:
        status_detail = response_body['detail']
        raise HTTPException(status_code=status_code, detail=status_detail)
    return Phoneme(
        name=input_grapheme.name,
        text=response_body['output']
    )


@app.post(
    path="/phoneme2grapheme",
    response_model=Grapheme,
    summary="Summary: convert phonemes to graphemes",
    description="""Description: 
    convert phonemes (perceptually distinct units of sound) to graphemes (smallest functional unit of a writing system)
    """
)
async def phoneme2grapheme(input_phoneme: Phoneme):
    output = phoneme2grapheme_translate.main(input_phoneme.text)
    return Grapheme(
        name=input_phoneme.name,
        text=output
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
    output = nmt.preprocess_sentence(input_grapheme.text)
    return Grapheme(
        name=input_grapheme.name,
        text=output
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
    output = spa2eng_translate.main(input_grapheme.text)
    return EnglishGrapheme(text=output)


@app.post(
    path="/translate-english-to-spanish",
    response_model=SpanishGrapheme,
    summary="Summary: translate an english sentence to spanish",
    description="""Description:
    Uses Neural machine translation with attention to translate english to spanish     
    """
)
async def translate_english_to_spanish(input_grapheme: EnglishGrapheme):
    output = eng2spa_translate.main(input_grapheme.text)
    return EnglishGrapheme(text=output)
