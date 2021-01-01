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

from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import Field


class Phoneme(BaseModel):
    name: str = Field(example="Phoneme")
    description: Optional[str] = Field(
        example="""any of the perceptually distinct units of sound in a specified language that distinguish one word from another,
         for example p, b, d, and t in the English words pad, pat, bad, and bat."""
    )
    text: str = Field(example="Y EH1 S T ER0 D EY2 AO1 L M AY1 T R AH1 B AH0 L Z S IY1 M D S OW1 F AA1 R AH0 W EY1")


class Grapheme(BaseModel):
    name: str = Field(example="Grapheme")
    description: Optional[str] = Field(example="grapheme is the smallest functional unit of a writing system")
    text: str = Field(example="yesterday all my troubles seemed so far away")


app = FastAPI()


@app.post("/grapheme2phoneme", response_model=Phoneme)
async def create_item(item: Grapheme):
    return item


@app.post("/phoneme2grapheme", response_model=Grapheme)
async def create_item(item: Phoneme):
    return item


if __name__ == "__main__":
    uvicorn.run(
        app="__main__:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload_dirs=True
    )
