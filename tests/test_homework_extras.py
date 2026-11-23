"""What the homework solution does beyond tests/homework/.

Those tests define the tasks; these cover the choices made on the way that
nothing else would notice breaking: the fallback's own settings and the
startup checks on them, billing a stream whose output came back unusable, and
what the stream route answers before it commits to a 200.
"""

from __future__ import annotations

import httpx
import pytest

import app.config as config_module
from app.main import app as fastapi_app
from app.schema import StreamEvent
from app.service import TriageService

TEXT = "Checkout fails with a 500 for every card payment since the 4.2 deploy."


# --- the fallback's settings ------------------------------------------------


def test_no_fallback_by_default(env):
    assert env().for_fallback() is None


def test_the_fallback_inherits_model_and_url_from_the_primary(env, mock_server):
    fallback = env(FALLBACK_PROVIDER="openai").for_fallback()
    assert fallback.provider == "openai"
    assert fallback.model == "claude-haiku-4-5"
    assert fallback.provider_base_url == mock_server.base_url
    assert fallback.fallback_provider == "", "a fallback must not have a fallback of its own"


def test_the_fallback_model_and_url_override_the_primary(env):
    fallback = env(
        FALLBACK_PROVIDER="openai",
        FALLBACK_MODEL="gpt-4o-mini",
        FALLBACK_BASE_URL="http://127.0.0.1:9/v1",
    ).for_fallback()
    assert fallback.model == "gpt-4o-mini"
    assert fallback.provider_base_url == "http://127.0.0.1:9/v1"


def test_a_typo_in_the_fallback_provider_fails_at_startup(env):
    with pytest.raises(ValueError, match="FALLBACK_PROVIDER"):
        env(FALLBACK_PROVIDER="opnai")


def test_a_fallback_to_the_real_api_needs_the_opt_in(env):
    """Set-but-empty is the SDK default, which is the real API."""
    with pytest.raises(ValueError, match="ALLOW_REAL_API"):
        env(FALLBACK_PROVIDER="openai", FALLBACK_MODEL="gpt-4o-mini", FALLBACK_BASE_URL="")


def test_a_real_fallback_without_its_own_model_fails_at_startup(env):
    """Not on the first outage, by sending a Claude model name to OpenAI."""
    real_openai = {
        "ALLOW_REAL_API": "true",
        "FALLBACK_PROVIDER": "openai",
        "FALLBACK_BASE_URL": "https://api.openai.com/v1",
    }
    with pytest.raises(ValueError, match="FALLBACK_MODEL"):
        env(**real_openai)
    assert env(**real_openai, FALLBACK_MODEL="gpt-4o-mini").for_fallback().model == "gpt-4o-mini"


def test_a_real_fallback_does_not_inherit_the_other_vendors_host(env):
    with pytest.raises(ValueError, match="FALLBACK_BASE_URL"):
        env(
            ALLOW_REAL_API="true",
            PROVIDER_BASE_URL="https://api.anthropic.com",
            FALLBACK_PROVIDER="openai",
            FALLBACK_MODEL="gpt-4o-mini",
        )


def test_the_mock_needs_neither_model_nor_url(env):
    """It answers to any model name and speaks both dialects on one port."""
    assert env(FALLBACK_PROVIDER="openai").for_fallback() is not None


# --- billing a stream -------------------------------------------------------


def test_token_usage_stays_off_the_wire():
    event = StreamEvent(type="done", input_tokens=12, output_tokens=34)
    assert "tokens" not in event.model_dump_json(exclude_none=True)


async def _assert_charged_for(service, mock_server) -> None:
    events = [event async for event in service.triage_stream(TEXT)]

    assert events[-1].type == "error" and events[-1].transient is False, [e.type for e in events]
    ledger = service.limits.ledger
    billed = mock_server.stats()["estimated_cost_usd"]
    assert billed > 0
    assert ledger.spent_usd == pytest.approx(billed, abs=1e-6)
    assert ledger.committed_usd == pytest.approx(ledger.spent_usd, abs=1e-9)


async def test_an_anthropic_stream_that_ends_in_a_refusal_is_charged(make_service, mock_server, env):
    """The tokens were generated and billed; only the answer was unusable.

    Refusal, not bad_content: the mock's bad_content is schema-valid and wrong,
    so it ends in `done`. A refusal ends in a permanent `error` after a
    complete, billed response, which is the path under test.
    """
    env(FAULT_RATE="1.0", FAULT_MODE="refusal", TEMPERATURE="0.7")
    service = make_service()
    mock_server.reset_stats()
    await _assert_charged_for(service, mock_server)


async def test_an_openai_stream_that_will_not_validate_is_charged(
    make_service, mock_server, env, monkeypatch
):
    """The mock models refusals on the Anthropic side only, so the unusable
    answer is made in the adapter instead; the stream and its billing are real."""
    import app.providers.openai_p as openai_p
    from app.providers.base import PermanentProviderError

    def will_not_validate(text: str):
        raise PermanentProviderError("provider output did not match schema")

    monkeypatch.setattr(openai_p, "_parse", will_not_validate)
    env(PROVIDER="openai", FAULT_RATE="0.0", TEMPERATURE="0.7")
    service = make_service()
    mock_server.reset_stats()
    await _assert_charged_for(service, mock_server)


# --- the stream route, before the first byte --------------------------------


def _client(service: TriageService) -> httpx.AsyncClient:
    fastapi_app.state.settings = config_module.get_settings()
    fastapi_app.state.service = service
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=fastapi_app), base_url="http://test")


async def _open(breaker) -> None:
    for _ in range(breaker.failure_threshold):
        await breaker.record_failure()


async def test_a_stream_over_the_ceiling_answers_402(env, mock_server):
    env(FAULT_RATE="0.0", SPEND_CEILING_USD="0.001")
    service = TriageService(settings=config_module.get_settings())
    try:
        async with _client(service) as client:
            response = await client.post("/triage/stream", json={"text": TEXT})
    finally:
        await service.aclose()

    assert response.status_code == 402, response.text
    assert response.json()["error"] == "spend_ceiling_exceeded"
    assert mock_server.stats()["calls"] == 0
    assert service.limits.ledger.committed_usd == 0


async def test_an_open_breaker_without_a_fallback_answers_503(env, mock_server):
    env(FAULT_RATE="0.0")
    service = TriageService(settings=config_module.get_settings())
    await _open(service.breaker)
    try:
        async with _client(service) as client:
            response = await client.post("/triage/stream", json={"text": TEXT})
    finally:
        await service.aclose()

    assert response.status_code == 503
    assert response.headers["retry-after"]
    assert mock_server.stats()["calls"] == 0
    assert service.limits.ledger.committed_usd == 0


async def test_an_open_breaker_streams_from_the_fallback(env, mock_server):
    env(FAULT_RATE="0.0", FALLBACK_PROVIDER="openai")
    service = TriageService(settings=config_module.get_settings())
    await _open(service.breaker)
    try:
        async with _client(service) as client:
            response = await client.post("/triage/stream", json={"text": TEXT})
            limits = (await client.get("/limits")).json()
    finally:
        await service.aclose()

    assert response.status_code == 200
    assert "event: done" in response.text
    assert service.limits.ledger.spent_usd == pytest.approx(
        mock_server.stats()["estimated_cost_usd"], abs=1e-6
    )
    assert (limits["breaker_state"], limits["fallback_breaker_state"]) == ("open", "closed")
