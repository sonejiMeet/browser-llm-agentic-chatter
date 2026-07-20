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
    },
    "claude": {
        "input": 'div[contenteditable="true"].ProseMirror, div[contenteditable="true"]',
        "submit": 'button[aria-label="Send Message"], button[aria-label="Send message"]',
        "response": "div.font-claude-message, div[data-is-streaming], div.assistant-message",
        "stop_button": 'button[aria-label="Stop"], button[aria-label="Stop response"]',
        "new_chat": 'button:has-text("New chat"), a:has-text("Start new chat")',
    },
    "gemini": {
        "input": 'div[contenteditable="true"], rich-textarea div[contenteditable="true"]',
        "submit": 'button[aria-label="Send message"], button[aria-label="Send"]',
        "response": "model-response, .model-response-text, message-content",
        "stop_button": 'button[aria-label="Stop"]',
        "new_chat": 'button:has-text("New chat")',
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
    },
}

PERPLEXITY_MODELS = [
    "Sonar", "Sonar Pro", "Sonar Reasoning",
    "GPT-4o", "GPT-4o Mini",
    "Claude 3.5 Sonnet", "Claude 3 Opus",
    "Gemini 2.0 Flash",
    "Grok-2",
]


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
        """Wait for generation to finish. Uses Playwright's native wait_for_selector
        to watch for the stop button to appear-then-disappear — zero polling overhead.
        Falls back to fast text-stability check if no stop button exists."""
        stop_sel = self.selectors.get("stop_button")
        deadline = time.time() + timeout_seconds

        if stop_sel:
            # Wait for stop button to appear (generation started)
            try:
                await self._page.wait_for_selector(stop_sel, state="visible", timeout=8000)
            except Exception:
                pass  # may not appear, continue

            # Wait for stop button to DISAPPEAR (generation done) — native, no polling
            try:
                remaining = max(1, deadline - time.time())
                await self._page.wait_for_selector(
                    stop_sel, state="hidden", timeout=remaining * 1000
                )
                await asyncio.sleep(0.2)  # settle
                return await self._read_last_response()
            except Exception:
                pass  # button didn't hide in time, fall through

        # Fallback: fast text-stability polling (for providers without stop button)
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
        """Read last response preserving whitespace (indentation matters for code)."""
        try:
            messages = await self._page.query_selector_all(self.selectors["response"])
            if messages:
                last = messages[-1]
                # textContent preserves leading spaces/tabs (inner_text strips them)
                text = await last.evaluate("el => el.textContent")
                return text.strip()
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
