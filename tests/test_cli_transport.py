"""Offline regression tests for fast completion, toolbar scoping and Rich UI."""
import asyncio
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from provider_adapters import ProviderAdapter, Snapshot
from reliable_browser import BrowserBridge

class TransportTests(unittest.IsolatedAsyncioTestCase):
    def bridge(self, snapshots):
        class Adapter:
            def __init__(self): self.index = 0
            async def rejection(self): return ''
            async def snapshot(self):
                snap = snapshots[min(self.index, len(snapshots)-1)]
                self.index += 1
                return snap
        bridge = object.__new__(BrowserBridge)
        bridge.config = {'response_settle_seconds': 0}
        bridge._adapter = Adapter()
        bridge._baseline = Snapshot(1, 'old', 'previous', False, True)
        bridge._locked = True
        bridge._saw_stop = False
        bridge.debug = False
        bridge.phase = 'Waiting'
        return bridge

    async def test_fast_reply_without_observing_stop(self):
        bridge = self.bridge([Snapshot(2, 'new', 'answer', False, True)])
        self.assertEqual(await bridge.wait_for_response(.3), 'answer')
        self.assertFalse(bridge._locked)

    async def test_old_toolbar_cannot_complete_old_response(self):
        bridge = self.bridge([Snapshot(1, 'old', 'previous', False, True)])
        with self.assertRaises(TimeoutError):
            await bridge.wait_for_response(.1)
        self.assertTrue(bridge._locked)

    async def test_pause_without_completion_evidence_is_not_complete(self):
        bridge = self.bridge([Snapshot(2, 'new', 'partial', False, False)])
        with self.assertRaises(TimeoutError):
            await bridge.wait_for_response(.1)

    async def test_streaming_toolbar_does_not_complete(self):
        bridge = self.bridge([Snapshot(2, 'new', 'partial', True, True), Snapshot(2, 'new', 'complete', False, True)])
        self.assertEqual(await bridge.wait_for_response(.3), 'complete')

    async def test_send_uses_existing_provider_submission_path(self):
        bridge = self.bridge([Snapshot(0, '', '', False, False)])
        bridge._locked = False
        bridge._adapter.prepare = AsyncMock()
        bridge.legacy = SimpleNamespace(send_message=AsyncMock(), debug=False)
        await bridge.send_message('hello')
        bridge.legacy.send_message.assert_awaited_once_with('hello')
        self.assertEqual(bridge.phase, 'Waiting for model response')

    async def test_submission_cancellation_is_not_swallowed(self):
        bridge = self.bridge([Snapshot(0, '', '', False, False)])
        bridge._locked = False
        bridge._adapter.prepare = AsyncMock()
        bridge.legacy = SimpleNamespace(send_message=AsyncMock(side_effect=asyncio.CancelledError()), debug=False)
        with self.assertRaises(asyncio.CancelledError):
            await bridge.send_message('hello')
        self.assertTrue(bridge._locked)

class UITests(unittest.IsolatedAsyncioTestCase):
    async def test_greeting_is_pretty_and_not_duplicated(self):
        import cli
        from rich.console import Console
        class Agent:
            def gather_context(self): return None
            async def run_turn(self, text, context):
                yield SimpleNamespace(kind='status', text='Sending task', data={})
                await asyncio.sleep(.15)
                yield SimpleNamespace(kind='response', text='Hello there!', data={})
                yield SimpleNamespace(kind='done', text='Hello there!', data={'complete':True})
        shell = object.__new__(cli.AgentShell)
        shell.tools = SimpleNamespace(change_log=[])
        shell.bridge = SimpleNamespace(phase='Waiting for model response')
        shell.agent = Agent()
        shell.status = None
        buffer = io.StringIO()
        with patch.object(cli, 'console', Console(file=buffer, force_terminal=False, width=90)):
            await shell.process('hi')
        output = buffer.getvalue()
        self.assertIn('Assistant', output)
        self.assertEqual(output.count('Hello there!'), 1)
        self.assertNotIn('[status]', output)
        self.assertIsNone(shell.status)

class ToolbarDOMTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        try:
            from playwright.async_api import async_playwright
            self.pw = await async_playwright().start()
            self.browser = await self.pw.chromium.launch()
            self.page = await self.browser.new_page()
        except Exception as exc:
            if hasattr(self, 'pw'): await self.pw.stop()
            if os.environ.get('REQUIRE_BROWSER_TESTS') == '1': raise
            self.skipTest(str(exc).splitlines()[0])

    async def asyncTearDown(self):
        await self.browser.close()
        await self.pw.stop()

    async def snapshot(self, html):
        await self.page.set_content(html)
        return await ProviderAdapter(self.page, {'response':'.answer', 'stop_button':'.stop'}).snapshot()

    async def test_sibling_answer_toolbar_is_found(self):
        snap = await self.snapshot('<article><div class="answer">hello</div><button aria-label="Copy response">Copy</button></article>')
        self.assertTrue(snap.ready)

    async def test_code_copy_is_not_completion_evidence(self):
        snap = await self.snapshot('<article><div class="answer"><pre><code>partial</code><button aria-label="Copy code">Copy</button></pre></div></article>')
        self.assertFalse(snap.ready)

    async def test_old_answer_toolbar_does_not_apply_to_new_answer(self):
        snap = await self.snapshot('<article><div class="answer">old</div><button aria-label="Copy response">Copy</button></article><article><div class="answer">new</div></article>')
        self.assertEqual(snap.text, 'new')
        self.assertFalse(snap.ready)

    async def test_hidden_toolbar_is_not_completion_evidence(self):
        snap = await self.snapshot('<article><div class="answer">partial</div><button hidden aria-label="Copy">Copy</button></article>')
        self.assertFalse(snap.ready)
