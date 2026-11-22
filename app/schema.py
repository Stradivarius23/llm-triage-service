"""GIVEN. The tests and the mock's JSON schema both derive from this file, so
changing it changes the goalposts."""

from __future__ import annotations

from typing import AsyncIterator, Literal, Protocol, get_args

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["trivial", "low", "medium", "high", "critical"]

# A free string let a real model answer "authentication" where the golden set
# says "auth": right answer, wrong word. As a Literal the list reaches the model
# as a JSON-schema enum, so it picks rather than invents, and a word outside it
# fails validation -- a permanent error, since a retry would get the same word.
# COMPONENTS is read off the type so the two can never drift apart.
Component = Literal[
    "auth", "payments", "database", "api", "ui",
    "search", "notifications", "storage", "unknown",
]
COMPONENTS: tuple[str, ...] = get_args(Component)


class TriageResult(BaseModel):
    """extra="forbid" is deliberate: a field the model invents is a bug, not a
    bonus, and silent field drift is how a contract rots."""

    model_config = ConfigDict(extra="forbid")

    severity: Severity
    component: Component
    duplicate_of: str | None = None
    steps: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class StreamEvent(BaseModel):
    """One event from a streamed extraction.

    `type` mirrors the coarse lifecycle a participant needs to reason about:
    `delta` carries incremental text, `done` carries the final parsed
    result, `error` carries a terminal failure (e.g. a truncated stream).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["start", "delta", "done", "error"]
    text: str | None = None
    result: TriageResult | None = None
    error: str | None = None
    # On an `error` event: would the same request plausibly succeed later? A
    # truncated stream, a 429 or a stall is transient; a refusal, max_tokens or
    # output that will not validate is permanent. The breaker counts only the
    # transient ones -- a bad input says nothing about upstream health. None is
    # read as transient, so an adapter that forgets to say keeps the old,
    # cautious behaviour rather than silently disabling the breaker.
    transient: bool | None = None
    # Token usage, on the terminal event of a stream that ran to completion --
    # `done`, or an `error` for output that came back but would not validate,
    # since that was billed too. The ledger settles from these; their absence
    # means the cost is unknown and the reservation is released instead. They
    # ride on the event because only the adapter sees the vendor's usage, and
    # exclude=True keeps them off the SSE wire: billing is not the client's API.
    input_tokens: int | None = Field(default=None, exclude=True)
    output_tokens: int | None = Field(default=None, exclude=True)


class TriageRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)


# Stripped before the schema goes on the wire. Measured 2026-09-01 against
# api.anthropic.com: `minimum`/`maximum` on a number are rejected outright --
# "For 'number' type, properties maximum, minimum are not supported" -- and
# llama.cpp accepts them and does not enforce them. Neither decoder gives you
# the bound, so it stays where it is actually checked: on TriageResult above.
_WIRE_UNSUPPORTED = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")


def _strip_unsupported(node: object) -> None:
    """Walk the whole document: bounds can appear nested, not just at the top."""
    if isinstance(node, dict):
        for key in _WIRE_UNSUPPORTED:
            node.pop(key, None)
        for value in node.values():
            _strip_unsupported(value)
    elif isinstance(node, list):
        for item in node:
            _strip_unsupported(item)


def triage_result_json_schema() -> dict:
    """A function, not a constant: SDKs mutate the dict you hand them."""
    schema = TriageResult.model_json_schema()
    _strip_unsupported(schema)
    return schema