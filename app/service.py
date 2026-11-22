"""The composition root: the order the reliability layers wrap the provider.

The order is the lesson. From outside in:

    idempotency  ->  spend ceiling  ->  breaker + retry  ->  semaphore  ->  SDK

  * Idempotency outermost. A cached result must cost nothing: no permit, no
    budget, no upstream call. Put it inside the semaphore and a burst of
    duplicate retries queues up to be told something we already knew.

  * Spend ceiling next. It reserves before the attempt, so the refusal
    happens before the money is spent. Note it wraps the *retry loop*: every
    attempt is charged, because every attempt costs real tokens. Charging only
    the successful one is how a retry storm goes unnoticed.

  * Breaker and retry together. Retry asks the breaker before each attempt,
    so an outage stops the loop instead of feeding it.

  * Semaphore innermost, held around the upstream call alone. It bounds what
    reaches the provider, which is the scarce thing.

  * Rate limit inside the semaphore, immediately before the call. Every
    attempt takes a slot, retries included, because the provider counts
    every request it receives.

  * A fallback provider, if configured, sits beside breaker + retry rather
    than around the whole stack: when the primary's retries run out or its
    breaker is open, the same spend ceiling, semaphore and rate limit carry
    the request to the fallback, which has a breaker of its own. Idempotency
    stays outermost, so a replayed key costs nothing whoever answered it.
    Only outages switch vendors. A refusal or an answer that will not
    validate is the request's problem, and the fallback would fail it too.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from app.config import Settings, get_settings
from app.providers.base import Provider
from app.reliability.breaker import CircuitBreaker, CircuitOpenError
from app.reliability.idempotency import IdempotencyStore, build_redis, make_key
from app.reliability.limits import Limits, SpendCeilingExceeded
from app.reliability.retry import RetryExhaustedError, RetryPolicy, with_retry
from app.schema import StreamEvent, TriageResult


def build_provider(settings: Settings) -> Provider:
    """The one place a dialect name becomes a concrete adapter.

    Imports are local so a process imports only the SDK of the dialect it is
    configured for. Until stage 7 is done, the OpenAI stub never loads for a
    service running the Anthropic dialect.

    Note what is NOT here: no reliability logic, no branching anywhere above
    this function. Everything from the semaphore upward is written against the
    Provider Protocol, so adding a dialect costs one adapter and one line here.
    """
    if settings.provider == "openai":
        from app.providers.openai_p import OpenAIProvider

        return OpenAIProvider(settings)
    from app.providers.anthropic_p import AnthropicProvider

    return AnthropicProvider(settings)


class TriageService:
    """Holds the provider and the reliability layers for the process."""

    def __init__(
        self,
        *,
        provider: Provider | None = None,
        settings: Settings | None = None,
        store: IdempotencyStore | None = None,
        limits: Limits | None = None,
        breaker: CircuitBreaker | None = None,
        fallback: Provider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or build_provider(self.settings)
        self.store = store or IdempotencyStore(build_redis(self.settings))
        self.limits = limits or Limits.from_settings(self.settings)
        self.breaker = breaker or CircuitBreaker(name="anthropic")
        fallback_settings = self.settings.for_fallback()
        if fallback is None and fallback_settings is not None:
            fallback = build_provider(fallback_settings)
        self.fallback = fallback
        # Its own breaker: the primary's says nothing about the fallback's
        # health, and sharing one would let an Anthropic outage open the
        # circuit on OpenAI too -- the moment it is the only vendor answering.
        self.fallback_breaker = (
            CircuitBreaker(name=f"fallback:{self.settings.fallback_provider or 'custom'}")
            if fallback is not None else None
        )
        self.policy = RetryPolicy(max_retries=self.settings.max_retries)
        # Salts the cache key with the settings that change the answer, so a
        # run at a different temperature or model does not serve the previous
        # run's cached result.
        self._key_salt = f"{self.settings.model}:{self.settings.temperature}"

    # --- unary ------------------------------------------------------------

    async def triage(self, text: str, *, idem_key: str | None = None) -> tuple[TriageResult, bool]:
        """Triage `text`. Returns `(result, served_from_cache)`.

        With an idempotency key, the second call returns the stored object and
        makes no upstream call at all.
        """
        key = make_key(text, idem_key=idem_key, salt=self._key_salt)
        return await self.store.get_or_call(
            key, lambda: self._attempt_with_limits(text, idem_key)
        )

    async def _attempt_with_limits(self, text: str, idem_key: str | None) -> TriageResult:
        """One logical request: budget, breaker+retry, semaphore, upstream --
        and if the primary is down, the same again against the fallback."""
        try:
            return await self._through(self.provider, self.breaker, text, idem_key)
        except (RetryExhaustedError, CircuitOpenError):
            if self.fallback is None:
                raise
            # Raised from here, a failure carries the primary's as __context__,
            # so a 503 with both vendors down still shows why each one failed.
            return await self._through(self.fallback, self.fallback_breaker, text, idem_key)

    async def _through(
        self, provider: Provider, breaker: CircuitBreaker | None, text: str, idem_key: str | None
    ) -> TriageResult:
        async def one_attempt() -> TriageResult:
            # Reserve worst-case budget BEFORE the call. Raises rather than
            # reporting an overrun after the fact.
            reserved = await self.limits.ledger.reserve()
            settled = False
            try:
                async with self.limits.limiter():
                    await self._rate_slot()
                    result, in_tok, out_tok = await provider.extract_with_usage(
                        text, idem_key=idem_key
                    )
                await self.limits.ledger.settle(reserved, in_tok, out_tok)
                settled = True
                return result
            finally:
                if not settled:
                    await self.limits.ledger.release(reserved)

        return await with_retry(one_attempt, policy=self.policy, breaker=breaker)

    # --- streaming --------------------------------------------------------

    async def triage_stream(self, text: str) -> AsyncIterator[StreamEvent]:
        """Stream a triage result.

        No retry and no cache on this path: a stream has already started
        delivering bytes to the caller, so a transparent retry would replay a
        different answer mid-response. A truncated stream surfaces as an
        `error` event, which is what `test_stream.py` asserts.

        Everything that can refuse the request happens before the first event:
        the breaker (the fallback's, if the primary's is open), then the spend
        reservation. Both raise out of the first `anext()`, which is why the
        route pulls one event before it commits to a 200 -- once a
        StreamingResponse has started there is no status code left to change.
        For the same reason, a stream fails over only while the primary's
        breaker is open: after the first byte, switching vendors would splice
        two answers into one response.

        The reservation is the unary path's worst case. It settles from the
        usage the adapter puts on the terminal event, and is released when
        there is none -- a severed stream reports no final usage.
        """
        provider, breaker = await self._stream_target()
        reserved = await self.limits.ledger.reserve()
        settled = False
        upstream_failed = False
        try:
            async with self.limits.limiter():
                await self._rate_slot()
                async for event in provider.stream(text):
                    if event.type == "error" and event.transient is not False:
                        upstream_failed = True
                    if event.input_tokens is not None and event.output_tokens is not None:
                        await self.limits.ledger.settle(
                            reserved, event.input_tokens, event.output_tokens
                        )
                        settled = True
                    yield event
        finally:
            if not settled:
                await self.limits.ledger.release(reserved)
        # A permanent error still counts as success here: upstream answered,
        # and it was the request that could not be served. Only a transient
        # error is evidence that upstream is sick.
        if upstream_failed:
            await breaker.record_failure()
        else:
            await breaker.record_success()

    async def _stream_target(self) -> tuple[Provider, CircuitBreaker]:
        """The provider to stream from, or CircuitOpenError if none will do."""
        try:
            await self.breaker.before_call()
            return self.provider, self.breaker
        except CircuitOpenError:
            if self.fallback is None or self.fallback_breaker is None:
                raise
        await self.fallback_breaker.before_call()
        return self.fallback, self.fallback_breaker

    async def _rate_slot(self) -> None:
        if self.limits.rate is not None:
            await self.limits.rate.acquire()

    async def aclose(self) -> None:
        try:
            await self.provider.aclose()
        finally:
            if self.fallback is not None:
                await self.fallback.aclose()
