"""Offline DOM tests; require an installed Chromium, not a provider login."""
import os
from pathlib import Path
import unittest
from provider_adapters import ProviderAdapter, DeepSeekAdapter
from protocol import parse_response

class DOMTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        try:
            from playwright.async_api import async_playwright
            self.pw = await async_playwright().start()
            self.browser = await self.pw.chromium.launch(headless=True)
            self.page = await self.browser.new_page()
        except Exception as exc:
            if hasattr(self, 'pw'):
                await self.pw.stop()
            if os.environ.get('REQUIRE_BROWSER_TESTS') == '1':
                raise
            self.skipTest('Chromium unavailable: ' + str(exc).splitlines()[0])

    async def asyncTearDown(self):
        await self.browser.close()
        await self.pw.stop()

    async def fixture(self, name):
        await self.page.set_content((Path(__file__).parent/'fixtures'/name).read_text(encoding='utf-8'))

    async def test_nested_roots_and_raw_json(self):
        await self.fixture('chat.html')
        adapter = ProviderAdapter(self.page, {'response':'[data-message-author-role="assistant"], .markdown'})
        snap = await adapter.snapshot()
        self.assertEqual(snap.count,1)
        obj = parse_response(snap.text,'t')
        self.assertEqual(obj['message'],'python\n  exact code')
        self.assertNotIn('Copy',snap.text)

    async def test_deepseek_stale_and_fresh_errors(self):
        await self.fixture('deepseek.html')
        adapter = DeepSeekAdapter(self.page,{'response':'.ds-markdown'})
        await adapter.prepare()
        self.assertEqual(await adapter.rejection(),'')
        await self.page.locator('span._1ce76f5').evaluate("n => n.textContent = 'Too many messages.'")
        self.assertIn('Too many messages',await adapter.rejection())
        snap = await adapter.snapshot()
        self.assertIn('python\n  exact code',snap.text)
