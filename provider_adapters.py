"""Provider DOM extraction and response-level completion evidence."""
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

READY_JS = r"""(node, selector) => {
    const visible = n => {
        const s = getComputedStyle(n), r = n.getBoundingClientRect();
        return s.display !== 'none' && s.visibility !== 'hidden' && r.width > 0 && r.height > 0;
    };
    // A code-block Copy button is not evidence that the response finished.
    let scope = node;
    for (let depth = 0; scope && depth < 4; depth++, scope = scope.parentElement) {
        if (scope.matches('body,main') || scope.querySelector('[data-message-author-role="user"]')) break;
        const roots = [...scope.querySelectorAll(selector)].filter(n => !n.parentElement?.closest(selector) || n === node);
        if (roots.some(n => n !== node && !node.contains(n))) break;
        if (scope.matches('[aria-busy="true"],[data-is-streaming="true"]') || scope.querySelector('[aria-busy="true"],[data-is-streaming="true"]')) return false;
        const buttons = [...scope.querySelectorAll('button,[role="button"]')];
        if (buttons.some(button => {
            if (!visible(button) || button.closest('pre,code') || button.getAttribute('aria-disabled') === 'true' || button.disabled) return false;
            const label = [button.getAttribute('aria-label'), button.getAttribute('title'), button.getAttribute('data-testid'), button.textContent].filter(Boolean).join(' ');
            return /\bcopy\b|copy-turn-action/i.test(label);
        })) return true;
    }
    return false;
}"""

class ProviderAdapter:
    def __init__(self, page, selectors):
        self.page, self.selectors = page, selectors
        self.baseline_errors = set()

    async def snapshot(self):
        selector = self.selectors['response']
        nodes = self.page.locator(selector)
        roots = []
        for i in range(await nodes.count()):
            node = nodes.nth(i)
            if not await node.is_visible():
                continue
            if await node.evaluate("n => !!n.closest('[data-message-author-role=\"user\"]')"):
                continue
            if not await node.evaluate('(n, s) => !!n.parentElement?.closest(s)', selector):
                roots.append(node)
        text, identity, ready = '', '', False
        busy = False
        if roots:
            latest = roots[-1]
            text = (await latest.evaluate(EXTRACT_JS)).strip()
            identity = await latest.evaluate("n => n.getAttribute('data-message-id') || n.id || ''")
            ready = await latest.evaluate(READY_JS, selector)
            busy = await latest.evaluate("n => !!n.closest('[aria-busy=\"true\"],[data-is-streaming=\"true\"]') || !!n.querySelector('[aria-busy=\"true\"],[data-is-streaming=\"true\"]')")
        stop = self.selectors.get('stop_button')
        if stop:
            buttons = self.page.locator(stop)
            for i in range(await buttons.count()):
                if await buttons.nth(i).is_visible():
                    busy = True
                    break
        return Snapshot(len(roots), identity, text, busy, ready)

    async def errors(self):
        return set()

    async def prepare(self):
        self.baseline_errors = await self.errors()

    async def rejection(self):
        return next(iter(await self.errors() - self.baseline_errors), '')

class DeepSeekAdapter(ProviderAdapter):
    async def errors(self):
        nodes = self.page.locator("[role='alert'],span._1ce76f5")
        found = set()
        for i in range(await nodes.count()):
            node = nodes.nth(i)
            if not await node.is_visible():
                continue
            if await node.evaluate('(n, s) => !!n.closest(s)', self.selectors['response']):
                continue
            text = ' '.join((await node.inner_text()).split())
            if re.fullmatch(r'(?:Messages? too frequent\.? Try again later\.?|Too many messages\.?|Rate limit exceeded\.?)', text, re.I):
                found.add(hashlib.sha256((await node.evaluate('n => n.outerHTML')).encode()).hexdigest() + ':' + text)
        return found

ADAPTERS = {'deepseek': DeepSeekAdapter}

def adapter_for(provider, page, selectors):
    return ADAPTERS.get(provider, ProviderAdapter)(page, selectors)
