"""ChatGPT assistant discovery across role, turn and markdown DOM layouts."""
from provider_adapters import ProviderAdapter, Snapshot, EXTRACT_JS

ASSISTANT = '[data-message-author-role="assistant"], [data-turn="assistant"], [data-markdown-text-style="assistant-message"], .agent-turn'
USER = '[data-message-author-role="user"], [data-turn="user"], [data-markdown-text-style="user-message"]'
TURN = '[data-turn-key], [data-turn-id], [data-testid^="conversation-turn-"], [data-turn]'
CONTENT = '[data-markdown-text-style="assistant-message"], [data-message-author-role="assistant"], .markdown'
STOP = 'button[data-testid="stop-button"], button[aria-label*="stop" i], button[title*="stop" i], [role="button"][aria-label*="stop" i]'

# Execute the extraction once per snapshot so a rerender cannot invalidate
# separate locator calls or accidentally mix two different response turns.
SNAPSHOT_JS = r"""(options) => {
    const visible = node => {
        const style = getComputedStyle(node), rect = node.getBoundingClientRect();
        return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
    };
    const turns = new Map();
    for (const candidate of document.querySelectorAll(options.assistant)) {
        if (!visible(candidate) || candidate.closest(options.user)) continue;
        const turn = candidate.closest(options.turn) || candidate;
        if (turn.matches(options.user)) continue;
        if (!turns.has(turn)) turns.set(turn, candidate);
    }
    const entries = [...turns.entries()].sort(([a], [b]) => a === b ? 0 : a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1);
    let text = '', identity = '', ready = false, busy = false;
    if (entries.length) {
        const [turn, candidate] = entries[entries.length - 1];
        let contents = [...turn.querySelectorAll(options.content)].filter(node => visible(node) && !node.closest(options.user));
        contents = contents.filter(node => !contents.some(other => other !== node && other.contains(node)));
        if (!contents.length) contents = [candidate];
        const extract = EXTRACT_FUNCTION;
        text = contents.map(extract).join('\n').trim();
        identity = turn.getAttribute('data-turn-key') || turn.getAttribute('data-turn-id') || candidate.getAttribute('data-message-id') || turn.getAttribute('data-testid') || turn.id || '';
        const stateNodes = [turn, ...turn.querySelectorAll('[data-talvt-turn-state],[data-turn-state],[data-is-streaming],[aria-busy]')];
        let explicitComplete = false;
        for (const node of stateNodes) {
            const state = node.getAttribute('data-talvt-turn-state') || node.getAttribute('data-turn-state');
            explicitComplete ||= state === 'complete';
            busy ||= ['streaming', 'generating', 'thinking', 'pending', 'in_progress'].includes(state) || node.getAttribute('data-is-streaming') === 'true' || node.getAttribute('aria-busy') === 'true';
        }
        // Only controls inside this assistant turn qualify, never older
        // answers, user turns, hidden controls or code-block Copy buttons.
        const toolbarReady = [...turn.querySelectorAll('button,[role="button"]')].some(button => {
            if (!visible(button) || button.closest('pre,code') || button.disabled || button.getAttribute('aria-disabled') === 'true') return false;
            const label = [button.getAttribute('aria-label'), button.getAttribute('title'), button.getAttribute('data-testid'), button.textContent].filter(Boolean).join(' ');
            return /\bcopy\b|copy-turn-action/i.test(label);
        });
        ready = !busy && (explicitComplete || toolbarReady);
    }
    busy ||= [...document.querySelectorAll(options.stop)].some(visible);
    return {count: entries.length, identity, text, generating: busy, ready: ready && !busy};
}""".replace('EXTRACT_FUNCTION', '(' + EXTRACT_JS + ')')

class ChatGPTAdapter(ProviderAdapter):
    async def snapshot(self):
        result = await self.page.evaluate(SNAPSHOT_JS, {'assistant': ASSISTANT, 'user': USER, 'turn': TURN, 'content': CONTENT, 'stop': STOP})
        return Snapshot(**result)
