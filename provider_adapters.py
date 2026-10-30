"""Provider DOM boundary. No tool parsing or payload cleanup lives here."""
import hashlib
import re
from dataclasses import dataclass

@dataclass
class Snapshot:
    count: int
    identity: str
    text: str
    generating: bool
    ready: bool

EXTRACT_JS = r"""node => {
    const clone = node.cloneNode(true);
    clone.querySelectorAll('button,[role="button"],svg,[aria-label*="copy" i],[aria-label*="download" i]').forEach(x => x.remove());
    clone.querySelectorAll('pre').forEach(pre => {
        const code = pre.querySelector('code');
        pre.replaceWith(document.createTextNode('\n' + (code ? code.textContent : pre.textContent) + '\n'));
    });
    clone.querySelectorAll('p,li,br').forEach(x => x.appendChild(document.createTextNode('\n')));
    return clone.textContent || '';
}"""

class ProviderAdapter:
    def __init__(self, page, selectors):
        self.page, self.selectors = page, selectors
        self.baseline_errors = set()

    async def snapshot(self):
        nodes = self.page.locator(self.selectors['response'])
        visible = [nodes.nth(i) for i in range(await nodes.count()) if await nodes.nth(i).is_visible()]
        # Select outermost matching message roots, not nested markdown fragments.
        roots = []
        for node in visible:
            nested = await node.evaluate('(n, selector) => !!n.parentElement?.closest(selector)', self.selectors['response'])
            if not nested:
                roots.append(node)
        text, identity, ready = '', '', False
        if roots:
            latest = roots[-1]
            text = (await latest.evaluate(EXTRACT_JS)).strip()
            identity = await latest.evaluate("n => n.getAttribute('data-message-id') || n.id || ''")
            copy = latest.locator('button[aria-label*="copy" i],button[title*="copy" i]')
            ready = await copy.count() > 0
        stop = self.selectors.get('stop_button')
        generating = False
        if stop:
            buttons = self.page.locator(stop)
            generating = any([await buttons.nth(i).is_visible() for i in range(await buttons.count())])
        return Snapshot(len(roots), identity, text, generating, ready)

    async def errors(self):
        return set()

    async def prepare(self):
        self.baseline_errors = await self.errors()

    async def rejection(self):
        errors = await self.errors()
        fresh = errors - self.baseline_errors
        return next(iter(fresh), '')

class DeepSeekAdapter(ProviderAdapter):
    async def errors(self):
        nodes = self.page.locator("[role='alert'],span._1ce76f5")
        found = set()
        for i in range(await nodes.count()):
            node = nodes.nth(i)
            if not await node.is_visible():
                continue
            # Exclude assistant prose and quoted examples of provider errors.
            if await node.evaluate('(n, s) => !!n.closest(s)', self.selectors['response']):
                continue
            text = ' '.join((await node.inner_text()).split())
            if re.fullmatch(r'(?:Messages? too frequent\.? Try again later\.?|Too many messages\.?|Rate limit exceeded\.?)', text, re.I):
                found.add(hashlib.sha256((await node.evaluate('n => n.outerHTML')).encode()).hexdigest() + ':' + text)
        return found

ADAPTERS = {'deepseek': DeepSeekAdapter}

def adapter_for(provider, page, selectors):
    return ADAPTERS.get(provider, ProviderAdapter)(page, selectors)
