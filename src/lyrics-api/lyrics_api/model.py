# generated by datamodel-codegen:
#   filename:  openapi.yaml
#   timestamp: 2021-01-01T21:04:31+00:00

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Grapheme(BaseModel):
    name: str = Field(..., example='Yesterday', title='Name')
    text: str = Field(
        ..., example='yesterday all my troubles seemed so far away', title='Text'
    )


class Phoneme(BaseModel):
    name: str = Field(..., example='Yesterday', title='Name')
    text: str = Field(
        ...,
        example='Y EH1 S T ER0 D EY2 AO1 L M AY1 T R AH1 B AH0 L Z S IY1 M D S OW1 F AA1 R AH0 W EY1',
        title='Text',
    )


class ValidationError(BaseModel):
    loc: List[str] = Field(..., title='Location')
    msg: str = Field(..., title='Message')
    type: str = Field(..., title='Error Type')


class HTTPValidationError(BaseModel):
    detail: Optional[List[ValidationError]] = Field(None, title='Detail')
