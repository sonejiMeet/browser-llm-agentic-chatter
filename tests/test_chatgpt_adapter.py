"""Offline ChatGPT DOM variants; no account, API or live chat required."""
import os
import unittest
from provider_adapters import adapter_for
from chatgpt_adapter import ChatGPTAdapter

class ChatGPTDOMTests(unittest.IsolatedAsyncioTestCase):
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
        # Deliberately pass the obsolete selectors to verify factory dispatch.
        adapter = adapter_for('chatgpt', self.page, {'response':'div[data-message-author-role="assistant"], div.agent-turn'})
        self.assertIsInstance(adapter, ChatGPTAdapter)
        return await adapter.snapshot()

    async def test_section_turn_with_sibling_toolbar(self):
        snap = await self.snapshot('<section data-turn="assistant" data-turn-id="a"><div class="markdown">hello</div><button data-testid="copy-turn-action-button">Copy</button></section>')
        self.assertEqual(snap.count, 1)
        self.assertEqual(snap.text, 'hello')
        self.assertEqual(snap.identity, 'a')
        self.assertTrue(snap.ready)

    async def test_new_markdown_and_explicit_complete_state(self):
        snap = await self.snapshot('<section data-turn-key="a" data-talvt-turn-state="complete"><h2>ChatGPT said:</h2><div data-markdown-text-style="assistant-message"><pre><code>{"kind":"final","message":"hello"}</code></pre></div></section>')
        self.assertEqual(snap.count, 1)
        self.assertEqual(snap.identity, 'a')
        self.assertEqual(snap.text, '{"kind":"final","message":"hello"}')
        self.assertTrue(snap.ready)
        self.assertFalse(snap.generating)

    async def test_non_div_role_and_nested_candidates_deduplicate(self):
        snap = await self.snapshot('<article data-turn="assistant"><section data-message-author-role="assistant"><div data-markdown-text-style="assistant-message">hello</div></section></article>')
        self.assertEqual(snap.count, 1)
        self.assertEqual(snap.text, 'hello')

    async def test_user_is_not_a_response(self):
        snap = await self.snapshot('<section data-turn="user"><div class="markdown">my prompt</div><button>Copy</button></section>')
        self.assertEqual(snap.count, 0)
        self.assertFalse(snap.ready)

    async def test_old_copy_button_cannot_complete_latest_turn(self):
        snap = await self.snapshot('<section data-turn="assistant" data-turn-id="a"><div class="markdown">old</div><button>Copy</button></section><section data-turn="assistant" data-turn-id="b"><div class="markdown">new</div></section>')
        self.assertEqual(snap.count, 2)
        self.assertEqual(snap.text, 'new')
        self.assertFalse(snap.ready)

    async def test_code_copy_is_not_response_completion(self):
        snap = await self.snapshot('<section data-turn="assistant"><div class="markdown"><pre><code>partial</code><button>Copy code</button></pre></div></section>')
        self.assertEqual(snap.text, 'partial')
        self.assertFalse(snap.ready)

    async def test_streaming_state_overrides_toolbar(self):
        snap = await self.snapshot('<section data-turn="assistant" data-talvt-turn-state="streaming"><div class="markdown">partial</div><button>Copy</button></section>')
        self.assertTrue(snap.generating)
        self.assertFalse(snap.ready)

    async def test_hidden_and_user_embedded_candidates_ignored(self):
        snap = await self.snapshot('<section hidden data-turn="assistant">hidden</section><section data-turn="user"><div data-markdown-text-style="assistant-message">quoted</div></section>')
        self.assertEqual(snap.count, 0)

    async def test_other_providers_keep_original_adapter(self):
        from provider_adapters import ProviderAdapter, DeepSeekAdapter
        self.assertIs(type(adapter_for('claude', self.page, {})), ProviderAdapter)
        self.assertIs(type(adapter_for('deepseek', self.page, {})), DeepSeekAdapter)
