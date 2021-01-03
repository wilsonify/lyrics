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
import os
import uvicorn
from fastapi import FastAPI
from lyrics_api import __version__
from lyrics_api.model import Phoneme, Grapheme
from python_consumer import consumer
from tensorflow_consumer import nmt

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
    description="Description: convert graphemes (smallest functional unit of a writing system) to phonemes (perceptually distinct units of sound)"
)
async def grapheme2phoneme(input_grapheme: Grapheme):
    output = consumer.graphemes2phonemes(input_grapheme.text)
    output = consumer.reduce_to_string(output)
    return Phoneme(
        name=input_grapheme.name,
        text=output
    )


@app.post(
    path="/phoneme2grapheme",
    response_model=Grapheme,
    summary="Summary: convert phonemes to graphemes",
    description="Description: convert phonemes (perceptually distinct units of sound) to graphemes (smallest functional unit of a writing system)"
)
async def phoneme2grapheme(item: Phoneme):
    return item


@app.post(
    path="/preprocess_sentence",
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
    path="/translate_english_to_spanish",
    response_model=Grapheme,
    summary="Summary: translate an english sentence to spanish",
    description="""Description:
    Uses Neural machine translation with attention    
    """
)
async def translate_sentence(input_grapheme: Grapheme):
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, "training_checkpoints")
    checkpoint = tf.train.Checkpoint()
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    output = nmt.translate(
        input_grapheme.text,
        max_length_targ=max_length_targ,
        max_length_inp=max_length_inp,
        inp_lang=inp_lang,
        encoder=encoder,
        targ_lang=targ_lang,
        decoder=decoder,
    )

    return Grapheme(name=input_grapheme.name, text=output)


if __name__ == "__main__":
    uvicorn.run(
        app="__main__:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload_dirs=True
    )
