"""Serialized browser transport; uncertain submissions are never retried."""
import asyncio
import time
from provider_adapters import adapter_for

class BrowserBridge:
    def __init__(self, config):
        # Retain existing browser/login/model-selection code, not its response parser.
        from browser import BrowserBridge as LegacyBridge
        self.config = config
        self.legacy = LegacyBridge(config)
        self._adapter = None
        self._baseline = None
        self._locked = False
        self._saw_stop = False
        self.debug = bool(config.get('debug_transactions', False))

    def _debug(self, state):
        if self.debug:
            print(f'[browser] {state}', flush=True)

    async def start(self):
        page = await self.legacy.start()
        self._adapter = adapter_for(self.legacy.provider, page, self.legacy.selectors)
        return page

    async def close(self):
        await self.legacy.close()

    async def ensure_logged_in(self, timeout_seconds=120):
        await self.legacy.ensure_logged_in(timeout_seconds)
        await self.legacy.page.locator(self.legacy.selectors['input']).first.wait_for(state='visible')

    async def select_model(self, name):
        return await self.legacy.select_model(name)

    async def new_chat(self):
        if self._locked:
            raise RuntimeError('Cannot roll over an active transaction')
        selector = self.legacy.selectors.get('new_chat')
        if not selector:
            raise RuntimeError('Provider has no new-chat adapter')
        await self.legacy.page.locator(selector).first.click()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if (await self._adapter.snapshot()).count == 0:
                return
            await asyncio.sleep(.1)
        raise RuntimeError('Could not verify a fresh conversation; nothing submitted')

    async def send_message(self, text):
        if self._locked:
            raise RuntimeError('Previous submission has not been reconciled')
        limit = int(self.config.get('max_browser_message_chars', 25000))
        if len(text) > limit:
            raise ValueError(f'Message exceeds browser limit ({limit}); request narrower context')
        self._baseline = await self._adapter.snapshot()
        if self._baseline.generating:
            raise RuntimeError('Provider is still generating; no prompt sent')
        self._locked = True
        self._debug('SUBMITTING')
        self._saw_stop = False
        try:
            await self._adapter.prepare()
            box = self.legacy.page.locator(self.legacy.selectors['input']).first
            await box.fill(text)
            selector = self.legacy.selectors.get('submit')
            if selector:
                await self.legacy.page.locator(selector).first.click()
            else:
                await box.press('Enter')
        except BaseException as exc:
            # Leave locked: submission may have happened before the failure.
            raise RuntimeError('SUBMISSION_UNKNOWN: inspect browser; do not automatically resend') from exc

    async def wait_for_response(self, timeout_seconds=300):
        if not self._locked:
            raise RuntimeError('No active transaction')
        deadline = time.monotonic() + timeout_seconds
        last, changed = '', time.monotonic()
        try:
            while time.monotonic() < deadline:
                rejection = await self._adapter.rejection()
                if rejection:
                    self._locked = False
                    raise RuntimeError('RATE_LIMITED: ' + rejection.split(':', 1)[-1])
                snap = await self._adapter.snapshot()
                fresh = snap.count > self._baseline.count or (snap.identity and snap.identity != self._baseline.identity) or snap.text != self._baseline.text
                if snap.generating:
                    self._saw_stop = True
                if snap.text != last:
                    changed, last = time.monotonic(), snap.text
                settled = time.monotonic() - changed >= float(self.config.get('response_settle_seconds', 1.0))
                allowed_fallback = self.config.get('allow_copy_completion_fallback', False) and snap.ready
                if fresh and snap.text and settled and not snap.generating and (self._saw_stop or allowed_fallback):
                    self._locked = False
                    self._debug('RESPONSE_CAPTURED')
                    return snap.text
                await asyncio.sleep(.1)
            raise TimeoutError('SUBMISSION_UNKNOWN: no verified complete response; inspect browser')
        except asyncio.CancelledError:
            # Keep locked until explicit user reconciliation or a fresh process.
            raise
