"""Serialized transport with visible progress and response-bound completion."""
import asyncio
import time
from provider_adapters import adapter_for

class BrowserBridge:
    def __init__(self, config):
        from browser import BrowserBridge as LegacyBridge
        self.config = config
        self.legacy = LegacyBridge(config)
        self._adapter = None
        self._baseline = None
        self._locked = False
        self._saw_stop = False
        self.debug = bool(config.get('debug_transactions', False))
        self.phase = 'Ready'

    def _debug(self, message):
        if self.debug:
            print('[browser] ' + message, flush=True)

    async def start(self):
        self.phase = 'Opening browser'
        page = await self.legacy.start()
        self._adapter = adapter_for(self.legacy.provider, page, self.legacy.selectors)
        self.phase = 'Ready'
        return page

    async def close(self):
        await self.legacy.close()

    async def ensure_logged_in(self, timeout_seconds=120):
        self.phase = 'Waiting for browser login'
        await self.legacy.ensure_logged_in(timeout_seconds)
        await self.legacy.page.locator(self.legacy.selectors['input']).first.wait_for(state='visible')
        self.phase = 'Ready'

    async def select_model(self, name):
        return await self.legacy.select_model(name)

    async def new_chat(self):
        if self._locked:
            raise RuntimeError('Cannot open a new chat while a submission is unresolved')
        selector = self.legacy.selectors.get('new_chat')
        if not selector:
            raise RuntimeError('Provider has no new-chat adapter')
        self.phase = 'Opening fresh conversation'
        await self.legacy.page.locator(selector).first.click()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if (await self._adapter.snapshot()).count == 0:
                self.phase = 'Ready'
                return
            await asyncio.sleep(.05)
        raise RuntimeError('Could not verify a fresh conversation; nothing submitted')

    async def send_message(self, text):
        if self._locked:
            raise RuntimeError('Previous submission is unresolved; inspect the browser before retrying')
        limit = int(self.config.get('max_browser_message_chars', 25000))
        if len(text) > limit:
            raise ValueError(f'Message exceeds browser limit ({limit}); request narrower context')
        self._baseline = await self._adapter.snapshot()
        if self._baseline.generating:
            raise RuntimeError('Provider is still generating; no prompt sent')
        self._locked = True
        self._saw_stop = False
        self.phase = 'Submitting prompt'
        self._debug('SUBMITTING')
        try:
            await self._adapter.prepare()
            # Preserve the existing provider-specific typing/paste/submit path.
            self.legacy.debug = self.debug
            await self.legacy.send_message(text)
            self.phase = 'Waiting for model response'
        except asyncio.CancelledError:
            self.phase = 'Submission cancelled; browser outcome unknown'
            raise
        except Exception as exc:
            self.phase = 'Submission outcome unknown'
            raise RuntimeError('SUBMISSION_UNKNOWN: inspect browser; no automatic resend') from exc

    async def wait_for_response(self, timeout_seconds=300):
        if not self._locked:
            raise RuntimeError('No active transaction')
        deadline = time.monotonic() + timeout_seconds
        last, changed, last_debug = '', time.monotonic(), 0.0
        snap = None
        while time.monotonic() < deadline:
            rejection = await self._adapter.rejection()
            if rejection:
                self._locked = False
                self.phase = 'Provider rejected request'
                raise RuntimeError('RATE_LIMITED: ' + rejection.split(':', 1)[-1])
            snap = await self._adapter.snapshot()
            # A disappeared old message must never become a fresh response.
            fresh = snap.count > 0 and (snap.count > self._baseline.count or bool(snap.identity and snap.identity != self._baseline.identity) or bool(snap.text and snap.text != self._baseline.text))
            if snap.generating:
                self._saw_stop = True
            if snap.text != last:
                changed, last = time.monotonic(), snap.text
            settled = time.monotonic() - changed >= float(self.config.get('response_settle_seconds', 1.0))
            self.phase = 'Model is responding' if fresh and snap.generating else 'Checking response completion' if fresh else 'Waiting for model response'
            # The answer-level finished toolbar handles fast replies where no
            # Stop control was observed. Code-copy buttons do not qualify.
            completed = self._saw_stop or snap.ready
            if fresh and snap.text and settled and not snap.generating and completed:
                self._locked = False
                self.phase = 'Response received'
                self._debug('RESPONSE_CAPTURED')
                return snap.text
            if self.debug and time.monotonic() - last_debug >= 1:
                last_debug = time.monotonic()
                self._debug(f'count={snap.count} chars={len(snap.text)} fresh={bool(fresh)} generating={snap.generating} ready={snap.ready} saw_stop={self._saw_stop}')
            await asyncio.sleep(.05)
        self.phase = 'Response completion could not be verified'
        detail = f' nodes={snap.count}, chars={len(snap.text)}, generating={snap.generating}, ready={snap.ready}' if snap else ''
        raise TimeoutError('SUBMISSION_UNKNOWN: inspect browser; no automatic resend.' + detail)
