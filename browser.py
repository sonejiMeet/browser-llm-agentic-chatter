"""
browser.py - Playwright wrapper for browser-hosted LLM chats.

Each prompt is a transaction:

    send_message()
        -> capture the previous assistant-response fingerprint
        -> submit the prompt
        -> observe the newest response
        -> return only after a new response is complete

DeepSeek-specific behavior:
- Extract raw source from <pre><code> blocks without rendered language,
  Copy, or Download controls.
- Detect rejected requests such as:
      Messages too frequent. Try again later.
  including the message rendered directly below the user's sent prompt.
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
    Browser,
    BrowserContext,
    Page,
    async_playwright,
)


POLL_INTERVAL_SECONDS = 0.05
TEXT_STABLE_SECONDS = 0.35
DEFAULT_RESPONSE_TIMEOUT_SECONDS = 300.0
CLIPBOARD_SETTLE_SECONDS = 0.12


DEEPSEEK_REJECTION_PATTERN = re.compile(
    r"(?i)"
    r"("
    r"messages?\s+too\s+frequent\.?\s*"
    r"try\s+again\s+later\.?"
    r"|"
    r"too\s+many\s+messages\.?"
    r"|"
    r"requests?\s+too\s+frequent\.?"
    r"|"
    r"request\s+too\s+frequent\.?"
    r"|"
    r"rate\s+limit(?:\s+exceeded)?\.?"
    r"|"
    r"you\s+are\s+sending\s+messages\s+too\s+quickly\.?"
    r")"
)


SELECTORS = {
    "chatgpt": {
        "input": (
            'div[contenteditable="true"].ProseMirror, '
            "#prompt-textarea, "
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
        "submit": None,
        "response": (
            "div[data-testid='answer'], "
            "div.answer-content, "
            "div.prose"
        ),
        "stop_button": (
            'button[aria-label*="stop" i], '
            'button[title*="stop" i], '
            'button[data-testid*="stop" i], '
            '[role="button"][aria-label*="stop" i], '
            '[role="button"][title*="stop" i]'
        ),
        "new_chat": 'a[aria-label="New"]',
        "copy_btn": (
            'button[aria-label*="copy" i], '
            'button[title*="copy" i], '
            'button[data-testid*="copy" i]'
        ),
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
        "rejection": (
            "span._1ce76f5, "
            "[role='alert'], "
            "[role='status'], "
            "[aria-live='assertive'], "
            "[aria-live='polite']"
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


class TransactionState(Enum):
    IDLE = auto()
    SENT = auto()
    STREAMING = auto()
    COMPLETE = auto()
    ERROR = auto()


class ProviderRateLimitError(RuntimeError):
    """Raised when a provider rejects a submitted prompt."""


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


class BrowserBridge:
    """Browser session and completion-aware LLM transaction bridge."""

    def __init__(self, config: dict):
        self.config = config
        self.provider = config.get("provider", "chatgpt")
        self.url = config["urls"][self.provider]

        self.user_data_dir = Path(
            config.get("user_data_dir", "./browser_profile")
        ).resolve()

        self.headless = bool(config.get("headless", False))
        self.browser_type = config.get("browser", "chromium")

        self.window_width = int(
            config.get("window_width", 1440)
        )
        self.window_height = int(
            config.get("window_height", 1000)
        )

        default_message_limit = (
            12_000
            if self.provider == "perplexity"
            else 25_000
        )

        self.max_browser_message_chars = int(
            config.get(
                "max_browser_message_chars",
                default_message_limit,
            )
        )

        self.perplexity_settle_seconds = float(
            config.get("perplexity_settle_seconds", 2.0)
        )

        self.perplexity_no_stop_settle_seconds = float(
            config.get("perplexity_no_stop_settle_seconds", 4.0)
        )

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

        self._pre_send_response_text = ""
        self._pre_send_response_count = 0
        self._pre_send_copy_count = 0

        self.debug = bool(
            config.get("debug_transactions", False)
        )
        self._transaction_id = 0

    @property
    def page(self) -> Page:
        """Return the active page or fail before startup."""
        if self._page is None:
            raise RuntimeError(
                "BrowserBridge.start() was not called"
            )

        return self._page

    def _debug(self, message: str) -> None:
        """Print transaction diagnostics when enabled."""
        if not self.debug:
            return

        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"[tx:{self._transaction_id:03d}] "
            f"{message}",
            flush=True,
        )

    async def start(self) -> Page:
        """Launch the persistent browser profile and visit the provider."""
        self._playwright = await async_playwright().start()

        launcher = getattr(
            self._playwright,
            self.browser_type,
        )

        launch_args = [
            "--disable-blink-features=AutomationControlled",
        ]

        launch_options: dict = {
            "user_data_dir": str(self.user_data_dir),
            "headless": self.headless,
            "args": launch_args,
        }

        if self.headless:
            launch_options["viewport"] = {
                "width": self.window_width,
                "height": self.window_height,
            }
        else:
            launch_args.extend(
                [
                    (
                        "--window-size="
                        f"{self.window_width},{self.window_height}"
                    ),
                    "--start-maximized",
                ]
            )
            launch_options["no_viewport"] = True

        self._context = await launcher.launch_persistent_context(
            **launch_options
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

        self._debug(
            "browser started "
            f"provider={self.provider} "
            f"headless={self.headless} "
            f"window={self.window_width}x{self.window_height}"
        )

        return self._page

    async def close(self) -> None:
        """Close the browser context and Playwright runtime."""
        if self._context:
            await self._context.close()
            self._context = None
            self._page = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def ensure_logged_in(
        self,
        timeout_seconds: int = 120,
    ) -> bool:
        """Wait for the chat input, allowing manual login."""
        try:
            await self.page.wait_for_selector(
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
        """Open a fresh provider conversation when available."""
        selector = self.selectors.get("new_chat")

        if not selector:
            return

        try:
            await self.page.click(selector, timeout=5000)
            await asyncio.sleep(0.25)
        except Exception:
            pass

    async def select_model(self, model_name: str) -> bool:
        """Select a supported provider model."""
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
            button = await self.page.query_selector(
                self.selectors["model_btn"]
            )

            if not button:
                return False

            current = await button.get_attribute("aria-label")

            if model_name.lower() in (current or "").lower():
                return True

            await button.click()

            dropdown = await self.page.wait_for_selector(
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
            group = await self.page.query_selector(
                self.selectors["model_selector"]
            )

            if not group:
                return False

            selected = await self.page.query_selector(
                self.selectors["model_selected"]
            )

            if selected:
                current_type = await selected.get_attribute(
                    "data-model-type"
                )

                if (
                    current_type
                    and model_name.lower() in current_type.lower()
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

    @staticmethod
    def _deepseek_rejection_message(text: str) -> str:
        """Extract a known DeepSeek request-rejection message."""
        if not text:
            return ""

        normalized = " ".join(text.split()).strip()

        match = DEEPSEEK_REJECTION_PATTERN.search(
            normalized
        )

        if not match:
            return ""

        return match.group(0).strip()
    
    
    async def _deepseek_visible_rejection(self) -> str:
        """
        Return a visible DeepSeek request-rejection message.

        DeepSeek displays this below the sent user message, commonly as:

            <span class="_1ce76f5">
                Messages too frequent. Try again later.
            </span>

        Do not depend on the hashed CSS class: use Playwright text locators,
        which match rendered text anywhere in the active DeepSeek page.
        """
        if self.provider != "deepseek":
            return ""

        messages = (
            "Messages too frequent. Try again later.",
            "Messages too frequent",
            "Too many messages",
            "Requests too frequent",
            "Request too frequent",
            "Rate limit exceeded",
            "Rate limit",
            "Try again later",
        )

        for message in messages:
            try:
                locator = self.page.get_by_text(
                    message,
                    exact=False,
                )

                count = await locator.count()

                for index in range(count):
                    candidate = locator.nth(index)

                    if await candidate.is_visible():
                        text = await candidate.inner_text()
                        detected = self._deepseek_rejection_message(
                            text
                        )

                        if detected:
                            return detected

            except Exception:
                continue

        try:
            page_text = await self.page.locator("body").inner_text()

            return self._deepseek_rejection_message(
                page_text or ""
            )

        except Exception as exc:
            self._debug(
                f"DeepSeek rejection text scan failed: {exc}"
            )
            return ""
        

    async def _raise_if_deepseek_rejected(
        self,
        response_text: str = "",
    ) -> None:
        """
        End the transaction when DeepSeek visibly rejects the submitted prompt.

        This checks both the assistant response text and the error displayed
        directly underneath the outgoing user message.
        """
        if self.provider != "deepseek":
            return

        message = self._deepseek_rejection_message(response_text)

        if not message:
            message = await self._deepseek_visible_rejection()

        if not message:
            return

        error = (
            "DeepSeek rejected this request: "
            f"{message}. The agent stopped; wait and try again."
        )

        if self._transaction:
            self._transaction.state = TransactionState.ERROR
            self._transaction.error = error
            self._transaction.completed_at = time.monotonic()

        self._debug(f"PROVIDER REJECTION: {message}")

        raise ProviderRateLimitError(error)

    async def _deepseek_response_text(
        self,
        response_node,
    ) -> str:
        """Extract DeepSeek response text without code-block UI chrome."""
        try:
            text = await response_node.evaluate(
                """
                (response) => {
                    const clone = response.cloneNode(true);

                    const controls = clone.querySelectorAll(
                        [
                            "button",
                            "[role='button']",
                            "svg",
                            "path",
                            "[aria-label*='copy' i]",
                            "[aria-label*='download' i]",
                            "[title*='copy' i]",
                            "[title*='download' i]",
                        ].join(", ")
                    );

                    for (const element of controls) {
                        element.remove();
                    }

                    const codeBlocks = clone.querySelectorAll("pre");

                    for (const pre of codeBlocks) {
                        const code = pre.querySelector("code");
                        const rawCode = (
                            code?.textContent ||
                            pre.textContent ||
                            ""
                        );

                        const replacement = document.createElement("div");

                        replacement.setAttribute(
                            "data-browser-agent-raw-code",
                            "true"
                        );

                        replacement.textContent = (
                            "\\n" + rawCode + "\\n"
                        );

                        pre.replaceWith(replacement);
                    }

                    return clone.innerText || clone.textContent || "";
                }
                """
            )

            return self._clean_deepseek_file_payloads(text or "")

        except Exception:
            return ""

    @staticmethod
    def _clean_deepseek_file_payloads(text: str) -> str:
        """Remove DeepSeek code chrome inside FILE payloads only."""
        if not text:
            return ""

        language_labels = {
            "assembly",
            "bash",
            "c",
            "cpp",
            "csharp",
            "css",
            "dart",
            "dockerfile",
            "go",
            "html",
            "java",
            "javascript",
            "js",
            "json",
            "jsx",
            "kotlin",
            "lua",
            "markdown",
            "md",
            "odin",
            "perl",
            "php",
            "powershell",
            "ps1",
            "py",
            "python",
            "r",
            "ruby",
            "rs",
            "rust",
            "scala",
            "sh",
            "shell",
            "sql",
            "swift",
            "toml",
            "ts",
            "typescript",
            "tsx",
            "xml",
            "yaml",
            "yml",
        }

        chrome_labels = {
            "copy",
            "copy code",
            "copy to clipboard",
            "download",
            "download code",
        }

        pattern = re.compile(
            r"(?P<start>\[\[\[FILE\b[^\]]*\]\]\])"
            r"(?P<body>.*?)"
            r"(?P<end>\[\[\[END\]\]\])",
            flags=re.DOTALL | re.IGNORECASE,
        )

        def clean_match(match: re.Match) -> str:
            lines = match.group("body").splitlines(
                keepends=True
            )
            index = 0

            while index < len(lines) and not lines[index].strip():
                index += 1

            while index < len(lines):
                label = lines[index].strip().lower()

                if label in language_labels or label in chrome_labels:
                    index += 1
                    continue

                break

            body = "".join(lines[index:])

            if body and not body.startswith("\n"):
                body = "\n" + body

            if body and not body.endswith("\n"):
                body += "\n"

            return (
                match.group("start")
                + body
                + match.group("end")
            )

        return pattern.sub(clean_match, text).strip()

    async def _response_snapshot(self) -> tuple[int, str]:
        """Return visible response count and newest response text."""
        try:
            nodes = await self.page.query_selector_all(
                self.selectors["response"]
            )

            visible_nodes = []

            for node in nodes:
                try:
                    if await node.is_visible():
                        visible_nodes.append(node)
                except Exception:
                    continue

            if not visible_nodes:
                return 0, ""

            newest_node = visible_nodes[-1]

            if self.provider == "deepseek":
                text = await self._deepseek_response_text(
                    newest_node
                )
            else:
                text = await newest_node.evaluate(
                    "element => element.innerText || "
                    "element.textContent || ''"
                )

            return len(visible_nodes), (text or "").strip()

        except Exception:
            return 0, ""

    async def _copy_button_count(self) -> int:
        """Return the count of visible Copy controls."""
        selector = self.selectors.get("copy_btn")

        if not selector:
            return 0

        try:
            buttons = await self.page.query_selector_all(selector)
            count = 0

            for button in buttons:
                if await button.is_visible():
                    count += 1

            return count

        except Exception:
            return 0

    async def _stop_button_visible(self) -> bool:
        """Return whether the standard provider Stop control is visible."""
        selector = self.selectors.get("stop_button")

        if not selector:
            return False

        try:
            button = await self.page.query_selector(selector)
            return bool(button and await button.is_visible())
        except Exception:
            return False

    async def _perplexity_generation_active(self) -> bool:
        """Detect Perplexity generation across changing interface variants."""
        try:
            return bool(
                await self.page.evaluate(
                    """
                    () => {
                        const visible = (element) => {
                            const style = getComputedStyle(element);
                            const rect = element.getBoundingClientRect();

                            return (
                                style.display !== "none" &&
                                style.visibility !== "hidden" &&
                                style.opacity !== "0" &&
                                rect.width > 0 &&
                                rect.height > 0
                            );
                        };

                        const busyElements = [
                            ...document.querySelectorAll(
                                '[aria-busy="true"]'
                            ),
                        ];

                        if (busyElements.some(visible)) {
                            return true;
                        }

                        const controls = [
                            ...document.querySelectorAll(
                                "button, [role='button'], [data-testid]"
                            ),
                        ];

                        return controls.some((element) => {
                            if (!visible(element)) {
                                return false;
                            }

                            const label = [
                                element.getAttribute("aria-label"),
                                element.getAttribute("title"),
                                element.getAttribute("data-testid"),
                                element.getAttribute("data-state"),
                                element.textContent,
                            ]
                                .filter(Boolean)
                                .join(" ")
                                .toLowerCase();

                            return (
                                /\\bstop\\b/.test(label) ||
                                /\\bcancel\\b/.test(label) ||
                                /stop.generat/.test(label) ||
                                /generating/.test(label) ||
                                /searching/.test(label)
                            );
                        });
                    }
                    """
                )
            )
        except Exception:
            return False

    async def _generation_active(self) -> bool:
        """Use the special detector only for Perplexity."""
        if self.provider == "perplexity":
            return await self._perplexity_generation_active()

        return await self._stop_button_visible()

    async def _copy_button_visible(self) -> bool:
        """Return whether any Copy control is visible."""
        return await self._copy_button_count() > 0

    async def _capture_response_baseline(self) -> None:
        """Capture assistant-response and Copy-control state before submit."""
        count, text = await self._response_snapshot()

        self._pre_send_response_count = count
        self._pre_send_response_text = text
        self._pre_send_copy_count = await self._copy_button_count()

    async def _focus_input(self) -> None:
        """Focus the provider input."""
        await self.page.wait_for_selector(
            self.selectors["input"],
            timeout=15000,
        )

        await self.page.focus(self.selectors["input"])

    async def _clear_input(self) -> None:
        """Clear text from the focused provider input."""
        await self.page.keyboard.press(f"{self._mod}+a")
        await self.page.keyboard.press("Backspace")

    async def _insert_text(self, text: str) -> bool:
        """Insert text directly into the active editable control."""
        try:
            inserted = await self.page.evaluate(
                """
                (text) => {
                    const element = document.activeElement;

                    if (!element) {
                        return false;
                    }

                    element.focus();

                    try {
                        document.execCommand(
                            "selectAll",
                            false,
                            null
                        );

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
        """Type text if direct insertion is rejected by the frontend."""
        lines = text.split("\n")

        for index, line in enumerate(lines):
            if line:
                await self.page.keyboard.type(line, delay=0)

            if index < len(lines) - 1:
                await self.page.keyboard.press("Shift+Enter")

    async def send_message(self, text: str) -> None:
        """Submit a prompt without waiting for its assistant response."""
        if not self._page:
            raise RuntimeError(
                "BrowserBridge.start() was not called"
            )

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        if len(text) > self.max_browser_message_chars:
            raise ValueError(
                "Refusing to send an oversized browser message: "
                f"{len(text):,} characters exceeds the configured "
                f"limit of {self.max_browser_message_chars:,}. "
                "Reduce or truncate tool feedback first."
            )

        self._transaction_id += 1
        self._debug(f"SEND begin chars={len(text)}")

        await self._capture_response_baseline()

        self._debug(
            "baseline captured "
            f"nodes={self._pre_send_response_count} "
            f"chars={len(self._pre_send_response_text)} "
            f"copies={self._pre_send_copy_count}"
        )

        await self._focus_input()
        await self._clear_input()

        inserted = await self._insert_text(text)

        if inserted:
            self._debug("input inserted via execCommand")
        else:
            self._debug("input fallback: keyboard typing")
            await self._type_text_fallback(text)

        await asyncio.sleep(0)

        if self.provider == "perplexity":
            await self.page.keyboard.press("Enter")
            self._debug("Perplexity Enter submit completed")
        else:
            submit_selector = self.selectors.get("submit")

            if submit_selector:
                try:
                    await self.page.click(
                        submit_selector,
                        timeout=3000,
                    )
                    self._debug("submit click completed")

                except Exception as exc:
                    self._debug(
                        f"submit click failed ({exc}); using Enter"
                    )
                    await self.page.keyboard.press("Enter")
                    self._debug("Enter submit completed")
            else:
                await self.page.keyboard.press("Enter")
                self._debug("Enter submit completed")

        self._transaction = ChatTransaction(
            state=TransactionState.SENT,
            started_at=time.monotonic(),
        )

        self._debug("SEND complete state=SENT")

    async def wait_for_response(
        self,
        timeout_seconds: int = DEFAULT_RESPONSE_TIMEOUT_SECONDS,
    ) -> str:
        """Wait for one new complete assistant response."""
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
        generation_seen = False
        generation_finished = False
        last_debug_at = 0.0

        self._debug(
            f"WAIT begin timeout={timeout_seconds:.0f}s "
            f"baseline_chars={len(baseline_text)}"
        )

        while time.monotonic() < deadline:
            now = time.monotonic()

            count, current_text = await self._response_snapshot()

            await self._raise_if_deepseek_rejected(
                current_text
            )

            generation_active = await self._generation_active()
            copy_count = await self._copy_button_count()

            current_is_new = bool(
                current_text
                and current_text != baseline_text
            )

            previous_state = transaction.state

            if generation_active:
                if not generation_seen:
                    self._debug("generation control appeared")

                generation_seen = True
                transaction.saw_stop_button = True

                if transaction.state == TransactionState.SENT:
                    transaction.state = TransactionState.STREAMING

            elif generation_seen:
                if not generation_finished:
                    self._debug("generation control disappeared")

                generation_finished = True

            if current_is_new and not response_started:
                response_started = True
                transaction.state = TransactionState.STREAMING
                last_text = current_text
                stable_since = now

                self._debug(
                    "NEW RESPONSE detected "
                    f"nodes={count} chars={len(current_text)}"
                )

            elif response_started and current_text != last_text:
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
                    "state transition "
                    f"{previous_state.name}->{transaction.state.name}"
                )

            if now - last_debug_at >= 1.0:
                self._debug(
                    f"WAIT state={transaction.state.name} "
                    f"nodes={count} "
                    f"chars={len(current_text)} "
                    f"new={current_is_new} "
                    f"generating={generation_active} "
                    f"generation_seen={generation_seen} "
                    f"copies={copy_count} "
                    f"stable={stable_for:.2f}s"
                )

                last_debug_at = now

            if self.provider == "perplexity":
                new_copy_button = (
                    copy_count > self._pre_send_copy_count
                )

                if (
                    response_started
                    and generation_seen
                    and generation_finished
                    and text_is_stable
                    and stable_for >= self.perplexity_settle_seconds
                ):
                    self._debug(
                        "COMPLETE Perplexity generation ended + "
                        f"stable for {stable_for:.2f}s"
                    )

                    transaction.state = TransactionState.COMPLETE
                    break

                if (
                    response_started
                    and not generation_seen
                    and new_copy_button
                    and not generation_active
                    and text_is_stable
                    and stable_for >= (
                        self.perplexity_no_stop_settle_seconds
                    )
                ):
                    self._debug(
                        "COMPLETE Perplexity new Copy control + "
                        f"stable for {stable_for:.2f}s"
                    )

                    transaction.state = TransactionState.COMPLETE
                    break

            elif (
                response_started
                and generation_seen
                and generation_finished
                and text_is_stable
            ):
                self._debug(
                    "COMPLETE stop disappeared + stable response"
                )

                transaction.state = TransactionState.COMPLETE
                break

            elif (
                self.provider == "deepseek"
                and text_is_stable
            ):
                self._debug(
                    "COMPLETE DeepSeek new response stable"
                )

                transaction.state = TransactionState.COMPLETE
                break

            elif text_is_stable:
                copy_visible = await self._copy_button_visible()

                if copy_visible or stable_for >= 1.0:
                    reason = (
                        "copy visible"
                        if copy_visible
                        else "stable for 1.0s"
                    )

                    self._debug(
                        "COMPLETE new response stable "
                        f"({reason})"
                    )

                    transaction.state = TransactionState.COMPLETE
                    break

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

        if transaction.state != TransactionState.COMPLETE:
            transaction.completed_at = time.monotonic()

            if self.provider == "perplexity":
                transaction.state = TransactionState.ERROR
                transaction.error = (
                    "Perplexity did not expose a confirmed completion "
                    "signal before the response timeout."
                )

                self._debug(
                    "ERROR Perplexity completion was not confirmed; "
                    "refusing possible partial text"
                )

                raise TimeoutError(transaction.error)

            self._debug(
                f"TIMEOUT after {timeout_seconds:.0f}s; "
                f"started={response_started} "
                f"chars={len(last_text)}"
            )

            transaction.state = TransactionState.COMPLETE

        transaction.completed_at = time.monotonic()
        self._debug("reading final response")

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

    async def _read_last_response(self) -> str:
        """Read the latest visible response, with Copy fallback."""
        _, text = await self._response_snapshot()

        if text:
            return self._strip_thinking(text)

        copied = await self._copy_last_response()

        if copied:
            if self.provider == "deepseek":
                copied = self._clean_deepseek_file_payloads(
                    copied
                )

            return self._strip_thinking(copied)

        return "[No response found]"

    async def _copy_last_response(self) -> str:
        """Copy the latest response using the visible provider control."""
        selector = self.selectors.get("copy_btn")

        if not selector:
            return ""

        try:
            text = await self.page.evaluate(
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

                    await new Promise((resolve) => {
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
        """Strip DeepSeek reasoning separated by a dashed divider."""
        if not text:
            return ""

        parts = re.split(r"\n---+\n", text, maxsplit=1)

        if len(parts) == 2 and len(parts[1].strip()) > 20:
            return parts[1].strip()

        return text.strip()

    async def transact(
        self,
        text: str,
        timeout_seconds: int = DEFAULT_RESPONSE_TIMEOUT_SECONDS,
    ) -> ChatTransaction:
        """Perform one complete send-and-wait transaction."""
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
        """Return visible text from the active provider conversation."""
        try:
            messages = await self.page.query_selector_all(
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