"""
browser.py — Playwright wrapper for browser-hosted LLM chats.

The bridge treats each prompt as a transaction:

    send_message()
        -> capture the prior assistant-response fingerprint
        -> submit
        -> continuously observe the latest response
        -> complete immediately when a NEW response is stable

Important DeepSeek detail:
DeepSeek may temporarily remove/recreate response DOM nodes while it starts
generating. Therefore response-node counts are not transaction identity.
The previous last-response text is the baseline; a transaction starts only
when the current last-response text differs from that baseline.
"""

from __future__ import annotations

import asyncio
import platform
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
)


# ── timings ────────────────────────────────────────────────────────

# Poll the DOM frequently while a transaction is active.
POLL_INTERVAL_SECONDS = 0.05

# A new response is accepted after it has not changed for this duration.
TEXT_STABLE_SECONDS = 0.35

# Maximum wait for a full LLM transaction.
DEFAULT_RESPONSE_TIMEOUT_SECONDS = 300.0

# Used only by the clipboard fallback.
CLIPBOARD_SETTLE_SECONDS = 0.12


# ── provider selectors ─────────────────────────────────────────────

SELECTORS = {
    "chatgpt": {
        "input": (
            'div[contenteditable="true"].ProseMirror, '
            '#prompt-textarea, '
            'div[contenteditable="true"][id="prompt-textarea"]'
        ),
        "submit": (
            'button[data-testid="send-button"], '
            'button[aria-label="Send prompt"], '
            'button[aria-label="Send"]'
        ),
        "response": (
            "div[data-message-author-role='assistant'], "
            "div.agent-turn"
        ),
        "stop_button": (
            'button[data-testid="stop-button"], '
            'button[aria-label="Stop generating"]'
        ),
        "new_chat": (
            'a[href="/"], '
            'button:has-text("New chat"), '
            'a:has-text("New chat")'
        ),
        "copy_btn": 'button[aria-label="Copy"]',
    },
    "claude": {
        "input": (
            'div[contenteditable="true"].ProseMirror, '
            'div[contenteditable="true"]'
        ),
        "submit": (
            'button[aria-label="Send Message"], '
            'button[aria-label="Send message"]'
        ),
        "response": (
            "div.font-claude-message, "
            "div[data-is-streaming], "
            "div.assistant-message"
        ),
        "stop_button": (
            'button[aria-label="Stop"], '
            'button[aria-label="Stop response"]'
        ),
        "new_chat": (
            'button:has-text("New chat"), '
            'a:has-text("Start new chat")'
        ),
        "copy_btn": (
            'button[aria-label="Copy response"], '
            'button[aria-label="Copy"]'
        ),
    },
    "gemini": {
        "input": (
            'div[contenteditable="true"], '
            'rich-textarea div[contenteditable="true"]'
        ),
        "submit": (
            'button[aria-label="Send message"], '
            'button[aria-label="Send"]'
        ),
        "response": (
            "model-response, "
            ".model-response-text, "
            "message-content"
        ),
        "stop_button": 'button[aria-label="Stop"]',
        "new_chat": 'button:has-text("New chat")',
        "copy_btn": (
            'button[aria-label="Copy"], '
            'button[data-action="copy"]'
        ),
    },
    "perplexity": {
        "input": "#ask-input",
        "submit": 'button[aria-label="Use voice mode"]',
        "response": (
            "div.prose, "
            "div[data-testid='answer'], "
            "div.answer-content"
        ),
        "stop_button": 'button[aria-label="Stop generating"]',
        "new_chat": 'a[aria-label="New"]',
        "copy_btn": 'button[aria-label="Copy"]',
        "model_btn": (
            'div[data-ask-input-container="true"] '
            'button[aria-haspopup="menu"]'
        ),
        "model_dropdown": 'div[role="menu"]',
        "model_option": (
            'div[role="menu"] '
            'div[role="menuitemradio"]'
        ),
    },
    "deepseek": {
        "input": 'textarea[placeholder="Message DeepSeek"]',
        "submit": None,
        "response": (
            "div.ds-markdown, "
            "div[class*='markdown']"
        ),
        "stop_button": (
            'button[aria-label="Stop"], '
            'div[role="button"] svg[data-icon="stop"]'
        ),
        "new_chat": 'div._5a8ac7a:has-text("New chat")',
        "copy_btn": (
            'button[aria-label="Copy"], '
            'div[role="button"]:has-text("Copy")'
        ),
        "model_selector": 'div[role="radiogroup"]',
        "model_option": (
            'div[data-model-type="expert"], '
            'div[data-model-type="instant"], '
            'div[data-model-type="vision"]'
        ),
        "model_selected": (
            'div[data-model-type][aria-checked="true"]'
        ),
    },
}


PERPLEXITY_MODELS = [
    "Sonar",
    "Sonar Pro",
    "Sonar Reasoning",
    "GPT-4o",
    "GPT-4o Mini",
    "Claude 3.5 Sonnet",
    "Claude 3 Opus",
    "Gemini 2.0 Flash",
    "Grok-2",
]


# ── transaction state ──────────────────────────────────────────────

class TransactionState(Enum):
    IDLE = auto()
    SENT = auto()
    STREAMING = auto()
    COMPLETE = auto()
    ERROR = auto()


@dataclass
class ChatTransaction:
    state: TransactionState = TransactionState.IDLE
    text: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    saw_stop_button: bool = False
    error: str = ""
    snapshots: list[tuple[float, int]] = field(
        default_factory=list
    )

    @property
    def duration_ms(self) -> int:
        if not self.started_at:
            return 0

        end = self.completed_at or time.monotonic()
        return int((end - self.started_at) * 1000)

    @property
    def done(self) -> bool:
        return self.state in {
            TransactionState.COMPLETE,
            TransactionState.ERROR,
        }


# ── browser bridge ─────────────────────────────────────────────────

class BrowserBridge:
    def __init__(self, config: dict):
        self.provider = config.get("provider", "chatgpt")
        self.url = config["urls"][self.provider]

        self.user_data_dir = Path(
            config.get("user_data_dir", "./browser_profile")
        ).resolve()

        self.headless = config.get("headless", False)
        self.browser_type = config.get("browser", "chromium")
        self.selectors = SELECTORS.get(
            self.provider,
            SELECTORS["chatgpt"],
        )
        self.model = config.get("model")

        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

        self._mod = (
            "Meta"
            if platform.system() == "Darwin"
            else "Control"
        )

        self._transaction: Optional[ChatTransaction] = None

        # The prior final assistant response is the transaction baseline.
        # This is intentionally text-based, not count-based.
        self._pre_send_response_text = ""
        self._pre_send_response_count = 0

        self.debug = bool(config.get("debug_transactions", False))
        self._transaction_id = 0

    # ── diagnostics ───────────────────────────────────────────────

    def _debug(self, message: str) -> None:
        if not self.debug:
            return

        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"[tx:{self._transaction_id:03d}] "
            f"{message}",
            flush=True,
        )

    # ── lifecycle ─────────────────────────────────────────────────

    async def start(self) -> Page:
        self._playwright = await async_playwright().start()

        launcher = getattr(
            self._playwright,
            self.browser_type,
        )

        self._context = await launcher.launch_persistent_context(
            user_data_dir=str(self.user_data_dir),
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled"],
            viewport={"width": 1280, "height": 900},
        )

        await self._context.grant_permissions(
            ["clipboard-read", "clipboard-write"]
        )

        if self._context.pages:
            self._page = self._context.pages[0]
        else:
            self._page = await self._context.new_page()

        await self._page.goto(
            self.url,
            wait_until="domcontentloaded",
        )

        return self._page

    async def close(self) -> None:
        if self._context:
            await self._context.close()
            self._context = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def ensure_logged_in(
        self,
        timeout_seconds: int = 120,
    ) -> bool:
        try:
            await self._page.wait_for_selector(
                self.selectors["input"],
                timeout=timeout_seconds * 1000,
            )
            return True
        except Exception:
            print(
                f"\n[!] Chat input not found at {self.url}. "
                "Log in manually, then press Enter."
            )
            await asyncio.get_event_loop().run_in_executor(
                None,
                input,
            )
            return True

    async def new_chat(self) -> None:
        selector = self.selectors.get("new_chat")

        if not selector:
            return

        try:
            await self._page.click(selector, timeout=5000)
            await asyncio.sleep(0.25)
        except Exception:
            pass

    # ── model selection ───────────────────────────────────────────

    async def select_model(self, model_name: str) -> bool:
        if self.provider == "perplexity":
            return await self._select_model_perplexity(model_name)

        if self.provider == "deepseek":
            return await self._select_model_deepseek(model_name)

        return False

    async def _select_model_perplexity(
        self,
        model_name: str,
    ) -> bool:
        try:
            button = await self._page.query_selector(
                self.selectors["model_btn"]
            )

            if not button:
                return False

            current = await button.get_attribute("aria-label")

            if model_name.lower() in (current or "").lower():
                return True

            await button.click()

            dropdown = await self._page.wait_for_selector(
                self.selectors["model_dropdown"],
                timeout=3000,
            )

            options = await dropdown.query_selector_all(
                self.selectors["model_option"]
            )

            for option in options:
                label = await option.inner_text()

                if model_name.lower() in label.lower():
                    await option.click()
                    print(f"  [model] Perplexity: {model_name}")
                    return True

            return False

        except Exception as exc:
            print(f"  [model] Perplexity selection failed: {exc}")
            return False

    async def _select_model_deepseek(
        self,
        model_name: str,
    ) -> bool:
        try:
            group = await self._page.query_selector(
                self.selectors["model_selector"]
            )

            if not group:
                return False

            selected = await self._page.query_selector(
                self.selectors["model_selected"]
            )

            if selected:
                current_type = await selected.get_attribute(
                    "data-model-type"
                )

                if (
                    current_type
                    and model_name.lower()
                    in current_type.lower()
                ):
                    return True

            options = await group.query_selector_all(
                self.selectors["model_option"]
            )

            for option in options:
                option_type = (
                    await option.get_attribute("data-model-type")
                    or ""
                )
                label = await option.inner_text()

                if (
                    model_name.lower() in option_type.lower()
                    or model_name.lower() in label.lower()
                ):
                    await option.click()
                    print(f"  [model] DeepSeek: {model_name}")
                    return True

            return False

        except Exception as exc:
            print(f"  [model] DeepSeek selection failed: {exc}")
            return False

    # ── DOM snapshots ─────────────────────────────────────────────

    async def _response_snapshot(
        self,
    ) -> tuple[int, str]:
        """
        Return response-node count and text from the final response node.

        Count is debug information only. It is not trusted for DeepSeek
        transaction identity because DeepSeek changes this count while
        hydrating and replacing markdown nodes.
        """
        try:
            nodes = await self._page.query_selector_all(
                self.selectors["response"]
            )

            if not nodes:
                return 0, ""

            text = await nodes[-1].evaluate(
                "element => element.innerText || "
                "element.textContent || ''"
            )

            return len(nodes), (text or "").strip()

        except Exception:
            return 0, ""

    async def _stop_button_visible(self) -> bool:
        selector = self.selectors.get("stop_button")

        if not selector:
            return False

        try:
            button = await self._page.query_selector(selector)
            return bool(button and await button.is_visible())
        except Exception:
            return False

    async def _copy_button_visible(self) -> bool:
        selector = self.selectors.get("copy_btn")

        if not selector:
            return False

        try:
            buttons = await self._page.query_selector_all(
                selector
            )

            return bool(
                buttons
                and await buttons[-1].is_visible()
            )
        except Exception:
            return False

    async def _capture_response_baseline(self) -> None:
        """
        Capture final assistant text immediately before a send.

        We do not decide that a response started merely because the number
        of matching response nodes changed. DeepSeek can go 2 -> 1 -> 2
        while still displaying the old response.
        """
        count, text = await self._response_snapshot()

        self._pre_send_response_count = count
        self._pre_send_response_text = text

    # ── input / sending ───────────────────────────────────────────

    async def _focus_input(self) -> None:
        selector = self.selectors["input"]

        await self._page.wait_for_selector(
            selector,
            timeout=15000,
        )

        await self._page.focus(selector)

    async def _clear_input(self) -> None:
        await self._page.keyboard.press(f"{self._mod}+a")
        await self._page.keyboard.press("Backspace")

    async def _insert_text(self, text: str) -> bool:
        """
        Insert text instantly through the active contenteditable / textarea.

        Falls back to keyboard input if the page rejects execCommand.
        """
        try:
            inserted = await self._page.evaluate(
                """
                (text) => {
                    const element = document.activeElement;

                    if (!element) {
                        return false;
                    }

                    element.focus();

                    try {
                        document.execCommand("selectAll", false, null);

                        const success = document.execCommand(
                            "insertText",
                            false,
                            text
                        );

                        const value = (
                            element.innerText ||
                            element.value ||
                            ""
                        );

                        return success && value.length > 0;
                    } catch (_) {
                        return false;
                    }
                }
                """,
                text,
            )

            return bool(inserted)

        except Exception:
            return False

    async def _type_text_fallback(self, text: str) -> None:
        lines = text.split("\n")

        for index, line in enumerate(lines):
            if line:
                await self._page.keyboard.type(line, delay=0)

            if index < len(lines) - 1:
                await self._page.keyboard.press("Shift+Enter")

    async def send_message(self, text: str) -> None:
        """
        Submit a browser-chat prompt.

        This only performs the SEND half of a transaction. The caller must
        invoke wait_for_response() to run the observation/completion phase.
        """
        if not self._page:
            raise RuntimeError(
                "BrowserBridge.start() was not called"
            )

        self._transaction_id += 1
        self._debug(f"SEND begin chars={len(text)}")

        await self._capture_response_baseline()

        self._debug(
            "baseline captured "
            f"nodes={self._pre_send_response_count} "
            f"chars={len(self._pre_send_response_text)}"
        )

        await self._focus_input()
        await self._clear_input()

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        inserted = await self._insert_text(text)

        if inserted:
            self._debug("input inserted via execCommand")
        else:
            self._debug("input fallback: keyboard typing")
            await self._type_text_fallback(text)

        # Scheduler yield only: this is not an intentional time delay.
        await asyncio.sleep(0)

        submit_selector = self.selectors.get("submit")

        if submit_selector:
            try:
                await self._page.click(
                    submit_selector,
                    timeout=3000,
                )
                self._debug("submit click completed")
            except Exception as exc:
                self._debug(
                    f"submit click failed ({exc}); using Enter"
                )
                await self._page.keyboard.press("Enter")
                self._debug("Enter submit completed")
        else:
            await self._page.keyboard.press("Enter")
            self._debug("Enter submit completed")

        self._transaction = ChatTransaction(
            state=TransactionState.SENT,
            started_at=time.monotonic(),
        )

        self._debug("SEND complete state=SENT")

    # ── completion ────────────────────────────────────────────────

    async def wait_for_response(
        self,
        timeout_seconds: int = 300,
    ) -> str:
        """
        Wait for one NEW assistant response.

        A response starts only when latest_response_text differs from the
        response text captured before send_message(). Once started, a stable
        response completes immediately after TEXT_STABLE_SECONDS.

        For normal providers, a stop-button disappearance is an extra strong
        signal. For DeepSeek, it is optional: stable new output is sufficient.
        """
        if not self._page:
            raise RuntimeError(
                "BrowserBridge.start() was not called"
            )

        transaction = self._transaction

        if transaction is None:
            transaction = ChatTransaction(
                state=TransactionState.SENT,
                started_at=time.monotonic(),
            )
            self._transaction = transaction

        timeout_seconds = float(timeout_seconds)
        deadline = time.monotonic() + timeout_seconds

        baseline_text = self._pre_send_response_text

        last_text = ""
        stable_since: Optional[float] = None
        response_started = False
        stop_disappeared = False
        last_debug_at = 0.0

        self._debug(
            f"WAIT begin timeout={timeout_seconds:.0f}s "
            f"baseline_chars={len(baseline_text)}"
        )

        while time.monotonic() < deadline:
            now = time.monotonic()

            count, current_text = await self._response_snapshot()
            stop_visible = await self._stop_button_visible()

            # This is the key fix:
            #
            # Never use count > old_count as the primary "new response"
            # condition. DeepSeek's count can become 1 while it still shows
            # the old text. A response starts only after content differs.
            current_is_new = bool(
                current_text
                and current_text != baseline_text
            )

            previous_state = transaction.state

            if stop_visible:
                if not transaction.saw_stop_button:
                    self._debug("stop button appeared")

                transaction.saw_stop_button = True

                if transaction.state == TransactionState.SENT:
                    transaction.state = TransactionState.STREAMING

            elif transaction.saw_stop_button:
                if not stop_disappeared:
                    self._debug("stop button disappeared")

                stop_disappeared = True

            if current_is_new and not response_started:
                response_started = True
                transaction.state = TransactionState.STREAMING
                last_text = current_text
                stable_since = now

                self._debug(
                    "NEW RESPONSE detected "
                    f"nodes={count} chars={len(current_text)}"
                )

            elif response_started:
                if current_text != last_text:
                    self._debug(
                        "response changed "
                        f"chars={len(last_text)}->{len(current_text)}"
                    )

                    last_text = current_text
                    stable_since = now
                    transaction.snapshots.append(
                        (now, len(current_text))
                    )

            stable_for = (
                now - stable_since
                if stable_since is not None
                else 0.0
            )

            text_is_stable = (
                response_started
                and bool(last_text)
                and stable_for >= TEXT_STABLE_SECONDS
            )

            if transaction.state != previous_state:
                self._debug(
                    f"state transition "
                    f"{previous_state.name}->"
                    f"{transaction.state.name}"
                )

            # One concise diagnostic heartbeat per second.
            if now - last_debug_at >= 1.0:
                self._debug(
                    f"WAIT state={transaction.state.name} "
                    f"nodes={count} "
                    f"chars={len(current_text)} "
                    f"new={current_is_new} "
                    f"stop={stop_visible} "
                    f"stable={stable_for:.2f}s"
                )

                last_debug_at = now

            # Strong completion signal for providers that expose Stop.
            if (
                response_started
                and transaction.saw_stop_button
                and stop_disappeared
                and text_is_stable
            ):
                self._debug(
                    "COMPLETE stop disappeared + stable response"
                )
                transaction.state = TransactionState.COMPLETE
                break

            # Universal completion signal:
            #
            # Once a new answer exists and its DOM text has been unchanged
            # for 350ms, return it. This is particularly important for
            # DeepSeek, whose Stop selector is unreliable or absent.
            if text_is_stable:
                if self.provider == "deepseek":
                    self._debug(
                        "COMPLETE DeepSeek new response stable"
                    )
                    transaction.state = TransactionState.COMPLETE
                    break

                # For the other providers, Copy visibility is confirmation.
                # If unavailable, stable new text still wins after 1 second.
                copy_visible = await self._copy_button_visible()

                if copy_visible or stable_for >= 1.0:
                    reason = (
                        "copy visible"
                        if copy_visible
                        else "stable for 1.0s"
                    )

                    self._debug(
                        f"COMPLETE new response stable ({reason})"
                    )

                    transaction.state = TransactionState.COMPLETE
                    break

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

        if transaction.state != TransactionState.COMPLETE:
            self._debug(
                f"TIMEOUT after {timeout_seconds:.0f}s; "
                f"started={response_started} "
                f"chars={len(last_text)}"
            )

            transaction.state = TransactionState.COMPLETE

        transaction.completed_at = time.monotonic()

        self._debug("reading final response")

        # Prefer the tracked text because it is guaranteed to be the NEW
        # response, rather than reading a DOM node that DeepSeek may replace.
        if last_text:
            transaction.text = self._strip_thinking(last_text)
        else:
            transaction.text = await self._read_last_response()

        self._debug(
            f"WAIT complete "
            f"elapsed={transaction.duration_ms}ms "
            f"final_chars={len(transaction.text)}"
        )

        return transaction.text

    # ── response reading ──────────────────────────────────────────

    async def _read_last_response(self) -> str:
        """
        Read the visible final response.

        The transaction loop normally returns its tracked stable text. This
        method is primarily a timeout and external-call fallback.
        """
        _, text = await self._response_snapshot()

        if text:
            return self._strip_thinking(text)

        copied = await self._copy_last_response()

        if copied:
            return self._strip_thinking(copied)

        return "[No response found]"

    async def _copy_last_response(self) -> str:
        selector = self.selectors.get("copy_btn")

        if not selector:
            return ""

        try:
            text = await self._page.evaluate(
                """
                async ({selector, delayMs}) => {
                    const buttons = [
                        ...document.querySelectorAll(selector)
                    ];

                    const button = buttons.at(-1);

                    if (!button) {
                        return "";
                    }

                    button.click();

                    await new Promise(resolve => {
                        setTimeout(resolve, delayMs);
                    });

                    try {
                        return await navigator.clipboard.readText();
                    } catch (_) {
                        return "";
                    }
                }
                """,
                {
                    "selector": selector,
                    "delayMs": int(
                        CLIPBOARD_SETTLE_SECONDS * 1000
                    ),
                },
            )

            return (text or "").strip()

        except Exception:
            return ""

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """
        Strip DeepSeek reasoning if Copy/DOM contains a reasoning divider.

        Does not alter [[[FILE]]], [[[SHELL]]], [[[READ]]], or [[[END]]].
        """
        if not text:
            return ""

        parts = re.split(r"\n---+\n", text, maxsplit=1)

        if len(parts) == 2 and len(parts[1].strip()) > 20:
            return parts[1].strip()

        return text.strip()

    # ── convenience API ───────────────────────────────────────────

    async def transact(
        self,
        text: str,
        timeout_seconds: int = 300,
    ) -> ChatTransaction:
        """Perform one complete send-and-wait browser transaction."""
        await self.send_message(text)

        result = await self.wait_for_response(
            timeout_seconds=timeout_seconds
        )

        if self._transaction:
            self._transaction.text = result
            return self._transaction

        return ChatTransaction(
            state=TransactionState.COMPLETE,
            text=result,
            completed_at=time.monotonic(),
        )

    async def get_full_conversation(self) -> str:
        try:
            messages = await self._page.query_selector_all(
                "article[data-testid^='conversation-turn-'], "
                "div.agent-turn, "
                "div[data-message-author-role], "
                "div.prose"
            )

            parts: list[str] = []

            for message in messages:
                try:
                    text = (await message.inner_text()).strip()

                    if text:
                        parts.append(text)

                except Exception:
                    continue

            return "\n\n---\n\n".join(parts)

        except Exception:
            return "[Could not read conversation]"  