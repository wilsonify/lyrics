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
from fastapi import FastAPI
from lyrics_api import __version__
from lyrics_api.model import Phoneme, Grapheme, SpanishGrapheme, EnglishGrapheme
from python_consumer import consumer
from tensorflow_consumer.translation import nmt
from tensorflow_consumer.translation.eng2spa import eng2spa_translate
from tensorflow_consumer.translation.spa2eng import spa2eng_translate
from tensorflow_consumer.translation.phoneme2grapheme import phoneme2grapheme_translate

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
    """
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
