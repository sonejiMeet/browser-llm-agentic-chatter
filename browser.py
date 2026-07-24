"""
browser.py - Playwright wrapper for driving web LLM chats.
Handles typing messages, reading responses, session persistence, and
model selection for multi-model providers like Perplexity.
"""

import asyncio
import platform
import time
from pathlib import Path
from playwright.async_api import async_playwright, Page, Browser, BrowserContext


SELECTORS = {
    "chatgpt": {
        "input": 'div[contenteditable="true"].ProseMirror, #prompt-textarea, div[contenteditable="true"][id="prompt-textarea"]',
        "submit": 'button[data-testid="send-button"], button[aria-label="Send prompt"], button[aria-label="Send"]',
        "response": "div[data-message-author-role='assistant'], div.agent-turn",
        "stop_button": 'button[data-testid="stop-button"], button[aria-label="Stop generating"]',
        "new_chat": 'a[href="/"], button:has-text("New chat"), a:has-text("New chat")',
        "copy_btn": 'button[aria-label="Copy"]',
    },
    "claude": {
        "input": 'div[contenteditable="true"].ProseMirror, div[contenteditable="true"]',
        "submit": 'button[aria-label="Send Message"], button[aria-label="Send message"]',
        "response": "div.font-claude-message, div[data-is-streaming], div.assistant-message",
        "stop_button": 'button[aria-label="Stop"], button[aria-label="Stop response"]',
        "new_chat": 'button:has-text("New chat"), a:has-text("Start new chat")',
        "copy_btn": 'button[aria-label="Copy response"], button[aria-label="Copy"]',
    },
    "gemini": {
        "input": 'div[contenteditable="true"], rich-textarea div[contenteditable="true"]',
        "submit": 'button[aria-label="Send message"], button[aria-label="Send"]',
        "response": "model-response, .model-response-text, message-content",
        "stop_button": 'button[aria-label="Stop"]',
        "new_chat": 'button:has-text("New chat")',
        "copy_btn": 'button[aria-label="Copy"], button[data-action="copy"]',
    },
    "perplexity": {
        "input": 'textarea[placeholder*="Ask"], textarea[placeholder*="ask"], div[contenteditable="true"]',
        "submit": 'button[aria-label="Submit"], button[type="submit"]',
        "response": "div.prose, div[data-testid='answer'], div.answer-content",
        "stop_button": 'button[aria-label="Stop generating"]',
        "new_chat": 'button:has-text("New Thread"), a:has-text("Home")',
        "model_btn": 'button:has-text("Sonar"), button:has-text("GPT"), button:has-text("Claude"), [data-testid="model-selector"]',
        "model_option": 'div[role="option"], div[role="menuitem"], button',
        "model_dropdown": 'div[role="listbox"], div[role="menu"]',
        "copy_btn": 'button[aria-label="Copy"]',
    },
    "deepseek": {
        "input": 'textarea[placeholder*="message"], textarea[placeholder*="Message"], textarea[placeholder*="Send"], div[contenteditable="true"], #chat-input',
        "submit": 'button[aria-label="Send"], button[type="submit"], div[role="button"]:has(svg)',
        "response": "div.ds-markdown, div[class*='markdown'], div[class*='message'], div[class*='assistant'], div[class*='response']",
        "stop_button": 'button[aria-label="Stop"], button:has(svg), div[role="button"]:has(svg)',
        "new_chat": 'button:has-text("New Chat"), a:has-text("New Chat")',
        "copy_btn": 'button[aria-label="Copy"], div[role="button"]:has-text("Copy"), button[class*="copy"]',
    },
}

PERPLEXITY_MODELS = [
    "Sonar", "Sonar Pro", "Sonar Reasoning",
    "GPT-4o", "GPT-4o Mini",
    "Claude 3.5 Sonnet", "Claude 3 Opus",
    "Gemini 2.0 Flash",
    "Grok-2",
]

# Perplexity Pro-search toggle — must disable so LLM doesn't see built-in tools
PERPLEXITY_PRO_SELECTOR = (
    'button:has-text("Pro"), '
    '[data-testid="pro-toggle"], '
    'label:has-text("Pro"), '
    'div[role="switch"][aria-label*="Pro"]'
)


class BrowserBridge:
    def __init__(self, config: dict):
        self.provider = config.get("provider", "chatgpt")
        self.url = config["urls"][self.provider]
        self.user_data_dir = Path(config.get("user_data_dir", "./browser_profile")).resolve()
        self.headless = config.get("headless", False)
        self.browser_type = config.get("browser", "chromium")
        self.selectors = SELECTORS.get(self.provider, SELECTORS["chatgpt"])
        self.model = config.get("model", None)
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._is_mac = platform.system() == "Darwin"
        self._mod = "Meta" if self._is_mac else "Control"

    async def start(self) -> Page:
        """Launch browser with persistent profile."""
        self._playwright = await async_playwright().start()
        launcher = getattr(self._playwright, self.browser_type)
        self._context = await launcher.launch_persistent_context(
            user_data_dir=str(self.user_data_dir),
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled"],
            viewport={"width": 1280, "height": 900},
        )
        await self._context.grant_permissions(["clipboard-read", "clipboard-write"])
        # Reuse existing tab if present (faster, less profile thrash)
        if self._context.pages:
            self._page = self._context.pages[0]
        else:
            self._page = await self._context.new_page()
        await self._page.goto(self.url, wait_until="domcontentloaded")
        return self._page

    async def ensure_logged_in(self, timeout_seconds: int = 120) -> bool:
        try:
            await self._page.wait_for_selector(
                self.selectors["input"], timeout=timeout_seconds * 1000
            )
            return True
        except Exception:
            print(f"\n[!] Chat input not found. Log in manually at {self.url}")
            print("    Press Enter here when ready...")
            await asyncio.get_event_loop().run_in_executor(None, input)
            return True

    async def select_model(self, model_name: str) -> bool:
        """For providers with model selectors (Perplexity)."""
        if self.provider != "perplexity":
            return False

        model_btn_sel = self.selectors.get("model_btn")
        if not model_btn_sel:
            return False

        try:
            btn = await self._page.query_selector(model_btn_sel)
            if not btn:
                return False

            current = await btn.inner_text()
            if model_name.lower() in current.lower():
                return True

            await btn.click()
            await asyncio.sleep(0.5)

            dropdown = await self._page.query_selector(
                self.selectors.get("model_dropdown", 'div[role="listbox"]')
            )
            if not dropdown:
                option = await self._page.query_selector(
                    f'div[role="option"]:has-text("{model_name}"), '
                    f'div[role="menuitem"]:has-text("{model_name}"), '
                    f'button:has-text("{model_name}")'
                )
                if option:
                    await option.click()
                    await asyncio.sleep(0.3)
                    print(f"  [model] Selected: {model_name}")
                    return True
            else:
                options = await dropdown.query_selector_all(
                    'div[role="option"], div[role="menuitem"]'
                )
                for opt in options:
                    text = await opt.inner_text()
                    if model_name.lower() in text.lower():
                        await opt.click()
                        await asyncio.sleep(0.3)
                        print(f"  [model] Selected: {model_name}")
                        return True

            return False
        except Exception as e:
            print(f"  [model] Selection failed: {e}")
            return False

    async def _response_text_stable(self, timeout: float = 2.0) -> bool:
        """Return True if the last response element's text has grown beyond
        its initial state AND stopped changing for *timeout* seconds.
        Old/stale text that never changed is NOT considered stable."""
        resp_sel = self.selectors["response"]
        deadline = time.time() + timeout
        last = ""
        stable_since = time.time()
        initial_len = -1  # -1 = not yet captured
        while time.time() < deadline:
            try:
                msgs = await self._page.query_selector_all(resp_sel)
                if not msgs:
                    await asyncio.sleep(0.2)
                    continue
                text = await msgs[-1].evaluate("el => el.textContent")
                if initial_len < 0:
                    initial_len = len(text)
                if text == last:
                    if len(text) >= initial_len and time.time() - stable_since >= timeout:
                        return True
                else:
                    last = text
                    stable_since = time.time()
            except Exception:
                pass
            await asyncio.sleep(0.15)
        return len(last) >= initial_len  # grew or stayed same? accept. shrunk? reject.

    async def disable_pro_search(self) -> bool:
        """Turn off Perplexity Pro search toggle if it's on.
        Pro search enables built-in tools that confuse our marker protocol."""
        if self.provider != "perplexity":
            return False
        try:
            # Pro toggle is a button/switch — try common selectors
            for sel in (
                'button[aria-label*="Pro"]',
                'button:has-text("Pro")',
                '[data-testid="pro-toggle"]',
                'label:has-text("Pro")',
            ):
                btn = await self._page.query_selector(sel)
                if btn:
                    # Check if it's already off (aria-checked="false")
                    checked = await btn.get_attribute("aria-checked")
                    if checked == "false":
                        return True  # already off
                    await btn.click()
                    await asyncio.sleep(0.3)
                    print("  [perplexity] Pro search disabled")
                    return True
            return False
        except Exception as e:
            print(f"  [perplexity] Pro toggle failed: {e}")
            return False

    async def new_chat(self):
        try:
            sel = self.selectors.get("new_chat")
            if sel:
                await self._page.click(sel, timeout=5000)
                await asyncio.sleep(1.0)
        except Exception:
            pass

    async def _focus_input(self) -> None:
        input_sel = self.selectors["input"]
        await self._page.wait_for_selector(input_sel, timeout=15000)
        await self._page.click(input_sel, timeout=5000)

    async def _clear_input(self) -> None:
        await self._page.keyboard.press(f"{self._mod}+a")
        await self._page.keyboard.press("Backspace")
        await asyncio.sleep(0.05)

    async def _paste_text(self, text: str) -> bool:
        """Insert full text instantly (clipboard / insertText). Far faster than typing."""
        try:
            # Method 1 (best for ProseMirror/React): insertText into focused editor.
            # This is instantaneous and does not depend on OS clipboard sync.
            ok = await self._page.evaluate(
                """(text) => {
                    const el = document.activeElement;
                    if (!el) return false;
                    el.focus();

                    // Select-all then insert replaces existing content cleanly
                    try { document.execCommand('selectAll', false, null); } catch (e) {}

                    try {
                        if (document.execCommand('insertText', false, text)) {
                            const t = (el.innerText || el.value || '');
                            // Accept if a substantial portion landed (editors may trim)
                            return t.length >= Math.min(text.length, 20) * 0.5;
                        }
                    } catch (e) {}

                    // Method 2: synthetic paste event with DataTransfer
                    try {
                        const dt = new DataTransfer();
                        dt.setData('text/plain', text);
                        const evt = new ClipboardEvent('paste', {
                            clipboardData: dt,
                            bubbles: true,
                            cancelable: true,
                        });
                        el.dispatchEvent(evt);
                        const t = (el.innerText || el.value || '');
                        if (t.length >= Math.min(text.length, 20) * 0.5) return true;
                    } catch (e) {}

                    // Method 3: direct value for plain <textarea>
                    if ('value' in el && el.tagName === 'TEXTAREA') {
                        el.value = text;
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        return true;
                    }
                    return false;
                }""",
                text,
            )
            if ok:
                return True

            # Method 4: OS clipboard + Ctrl/Meta+V (helps some providers)
            try:
                await self._context.grant_permissions(
                    ["clipboard-read", "clipboard-write"]
                )
            except Exception:
                pass

            # Write via a temporary textarea + execCommand('copy') so the
            # system clipboard is actually populated (navigator.clipboard
            # alone is flaky under automation).
            copied = await self._page.evaluate(
                """(text) => {
                    const ta = document.createElement('textarea');
                    ta.value = text;
                    ta.style.position = 'fixed';
                    ta.style.left = '-9999px';
                    document.body.appendChild(ta);
                    ta.focus();
                    ta.select();
                    let ok = false;
                    try { ok = document.execCommand('copy'); } catch (e) {}
                    document.body.removeChild(ta);
                    return ok;
                }""",
                text,
            )
            if copied:
                await self._focus_input()
                await self._clear_input()
                await self._page.keyboard.press(f"{self._mod}+v")
                await asyncio.sleep(0.1)
                has_content = await self._page.evaluate(
                    """() => {
                        const el = document.activeElement;
                        if (!el) return false;
                        return ((el.innerText || el.value || '').trim().length > 0);
                    }"""
                )
                if has_content:
                    return True

            return False
        except Exception as e:
            print(f"  [browser] paste failed: {e}")
            return False

    async def _type_chunked(self, text: str, chunk_size: int = 80) -> None:
        """Faster fallback than char-by-char: type large chunks with minimal delay."""
        i = 0
        n = len(text)
        while i < n:
            # Find a clean chunk boundary (prefer newline)
            end = min(i + chunk_size, n)
            if end < n:
                nl = text.rfind("\n", i, end)
                if nl > i:
                    end = nl + 1
            chunk = text[i:end]
            # keyboard.type handles most chars; shift+enter for newlines in contenteditable
            if "\n" in chunk:
                parts = chunk.split("\n")
                for j, part in enumerate(parts):
                    if part:
                        await self._page.keyboard.type(part, delay=0)
                    if j < len(parts) - 1:
                        await self._page.keyboard.press("Shift+Enter")
            else:
                await self._page.keyboard.type(chunk, delay=0)
            i = end

    async def send_message(self, text: str) -> None:
        """Send a message to the chat. Uses clipboard paste for speed."""
        await self._focus_input()
        await self._clear_input()

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        pasted = await self._paste_text(text)
        if not pasted:
            # Fast chunked typing fallback (still much faster than delay=5 char typing)
            await self._type_chunked(text)

        await asyncio.sleep(0.12)

        # Submit
        submit_sel = self.selectors["submit"]
        try:
            await self._page.click(submit_sel, timeout=4000)
        except Exception:
            # Enter as fallback
            await self._page.keyboard.press("Enter")

    async def wait_for_response(self, timeout_seconds: int = 300) -> str:
        """Wait for generation to finish. After the stop button disappears,
        confirms the Copy button is visible — thinking models pause/restart
        generation (stop button flickers), but Copy only appears when the
        full response including all thinking blocks is fully rendered."""
        stop_sel = self.selectors.get("stop_button")
        copy_sel = self.selectors.get("copy_btn")
        deadline = time.time() + timeout_seconds

        if stop_sel and copy_sel:
            # Wait for initial generation to start
            saw_stop = False
            gave_up_stop_at = None  # type: float | None
            try:
                await self._page.wait_for_selector(stop_sel, state="visible", timeout=8000)
                saw_stop = True
            except Exception:
                gave_up_stop_at = time.time()  # selector didn't match — track when we gave up

            # Loop: stop-disappear → check Copy → if stop reappears, loop again
            while time.time() < deadline:
                # Wait for stop button to disappear
                try:
                    remaining = max(1, deadline - time.time())
                    await self._page.wait_for_selector(
                        stop_sel, state="hidden", timeout=remaining * 1000
                    )
                except Exception:
                    pass

                # Copy button visible? → full response rendered.
                try:
                    btn = await self._page.query_selector(copy_sel)
                    if btn and await btn.is_visible():
                        await asyncio.sleep(0.2)
                        if await self._response_text_stable(timeout=2.0):
                            return await self._read_last_response()
                except Exception:
                    pass

                # Not visible — brief settle, then check if stop reappeared
                await asyncio.sleep(0.3)
                try:
                    btn = await self._page.query_selector(stop_sel)
                    if btn and await btn.is_visible():
                        saw_stop = True
                        continue
                except Exception:
                    pass

                # Neither stop nor copy visible.
                # If we never saw stop AND gave up >15s ago, fall through
                # to text-stability (selector probably doesn't match this provider).
                if not saw_stop and gave_up_stop_at and (time.time() - gave_up_stop_at) > 15:
                    break
                if not saw_stop:
                    await asyncio.sleep(0.5)
                    continue
                await asyncio.sleep(0.2)
                return await self._read_last_response()

            return await self._read_last_response()

        # Fallback: text-stability polling (no stop/copy selectors available)
        if stop_sel:
            try:
                await self._page.wait_for_selector(stop_sel, state="visible", timeout=8000)
            except Exception:
                pass
            try:
                remaining = max(1, deadline - time.time())
                await self._page.wait_for_selector(
                    stop_sel, state="hidden", timeout=remaining * 1000
                )
                await asyncio.sleep(0.2)
                return await self._read_last_response()
            except Exception:
                pass

        try:
            before = await self._page.inner_text("body")
        except Exception:
            before = ""

        await asyncio.sleep(0.3)
        last_len = len(before)
        stable = 0
        while time.time() < deadline:
            try:
                current = await self._page.inner_text("body")
            except Exception:
                await asyncio.sleep(0.1)
                continue
            cur_len = len(current)
            if cur_len > len(before) + 10 and cur_len == last_len:
                stable += 1
                if stable >= 2:
                    break
            elif cur_len != last_len:
                stable = 0
                last_len = cur_len
            await asyncio.sleep(0.1)

        return await self._read_last_response()

    async def _read_last_response(self) -> str:
        """Read the last assistant response. Tries Copy button first for
        markdown formatting, but falls back to innerText if the clipboard
        lost our [[[ markers (ChatGPT sometimes strips them on copy)."""
        copy_sel = self.selectors.get("copy_btn")
        resp_sel = self.selectors["response"]

        # Helper: read innerText from last response element
        async def _read_innertext():
            try:
                msgs = await self._page.query_selector_all(resp_sel)
                if msgs:
                    t = await msgs[-1].evaluate("el => el.innerText || el.textContent")
                    return (t or "").strip()
            except Exception:
                pass
            return ""

        # Method 1: click copy button → read clipboard
        if copy_sel:
            try:
                text = await self._page.evaluate("""
                    async (sel) => {
                        const btns = [...document.querySelectorAll(sel)];
                        if (!btns.length) return null;
                        btns[btns.length - 1].click();
                        await new Promise(r => setTimeout(r, 800));
                        try {
                            let t = await navigator.clipboard.readText();
                            if (t && t.trim()) return t;
                            await new Promise(r => setTimeout(r, 600));
                            t = await navigator.clipboard.readText();
                            if (t && t.trim()) return t;
                        } catch (e) { return null; }
                        return null;
                    }
                """, copy_sel)
                if text and text.strip():
                    # Verify [[[ markers survived the copy — ChatGPT may strip them
                    if "[[[" in text:
                        return text.strip()
                    # Markers lost — fall through to innerText
            except Exception:
                pass

        # Method 2: innerText (preserves [[[ brackets ChatGPT copy drops)
        text = await _read_innertext()
        if text:
            return text

        # Method 3: textContent (absolute fallback)
        try:
            messages = await self._page.query_selector_all(resp_sel)
            if messages:
                t = await messages[-1].evaluate("el => el.textContent")
                return (t or "").strip()
        except Exception as e:
            return f"[ERROR reading response: {e}]"
        return "[No response found]"

    async def get_full_conversation(self) -> str:
        try:
            all_msgs = await self._page.query_selector_all(
                "article[data-testid^='conversation-turn-'], "
                "div.agent-turn, div[data-message-author-role], div.prose"
            )
            parts = []
            for m in all_msgs:
                try:
                    txt = await m.inner_text()
                    parts.append(txt)
                except Exception:
                    pass
            return "\n\n---\n\n".join(parts)
        except Exception:
            return "[Could not read conversation]"

    async def close(self):
        if self._context:
            await self._context.close()
        if self._playwright:
            await self._playwright.stop()
