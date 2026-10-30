"""Regression tests for per-turn IDs and fail-closed approval input."""
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from journal import Journal

ROOT = Path(__file__).resolve().parents[1]

def action(path='first.md'):
    return {'id':'a1','tool':'write_file','arguments':{'path':path,'content':'hello','expected_sha256':None}}

class TurnJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.journal = Journal(self.temp.name)
        self.run = self.journal.start('task')
    def tearDown(self):
        self.journal.close()
        self.temp.cleanup()

    def test_reused_id_with_new_arguments_in_new_turn_is_allowed(self):
        first, second = action(), action('second.md')
        self.journal.claim(self.run, first, 'turn1')
        self.journal.finish(self.run, first, {'ok':1}, 'turn1')
        self.assertIsNone(self.journal.lookup(self.run, second, 'turn2'))
        self.assertIsNone(self.journal.claim(self.run, second, 'turn2'))
        self.journal.finish(self.run, second, {'ok':2}, 'turn2')
        self.assertEqual(self.journal.lookup(self.run, first, 'turn1'), {'ok':1})
        self.assertEqual(self.journal.lookup(self.run, second, 'turn2'), {'ok':2})

    def test_same_turn_replays_result(self):
        a = action()
        self.journal.claim(self.run, a, 'turn1')
        self.journal.finish(self.run, a, {'ok':1}, 'turn1')
        self.assertEqual(self.journal.claim(self.run, a, 'turn1'), {'ok':1})

    def test_changed_arguments_in_same_turn_are_rejected(self):
        self.journal.claim(self.run, action(), 'turn1')
        self.journal.finish(self.run, action(), {'ok':1}, 'turn1')
        with self.assertRaises(ValueError):
            self.journal.lookup(self.run, action('other.md'), 'turn1')

    def test_interrupted_call_remains_unknown(self):
        self.journal.claim(self.run, action(), 'turn1')
        with self.assertRaises(RuntimeError):
            self.journal.claim(self.run, action(), 'turn1')
        self.assertTrue(self.journal.unresolved(self.run))

    def test_legacy_records_still_readable(self):
        self.journal.claim(self.run, action())
        self.journal.finish(self.run, action(), {'legacy':True})
        self.assertEqual(self.journal.lookup(self.run, action()), {'legacy':True})
        self.assertIsNone(self.journal.lookup(self.run, action(), 'newturn'))

    def test_persistence_after_reopen(self):
        self.journal.claim(self.run, action(), 'turn1')
        self.journal.finish(self.run, action(), {'ok':1}, 'turn1')
        self.journal.close()
        self.journal = Journal(self.temp.name)
        self.assertEqual(self.journal.lookup(self.run, action(), 'turn1'), {'ok':1})

    def test_loop_explicitly_passes_host_turn_to_every_journal_call(self):
        tree = ast.parse((ROOT/'agent_core.py').read_text(encoding='utf-8'))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {'lookup','claim','finish'} and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'journal']
        self.assertEqual(len(calls),3)
        for call in calls:
            self.assertTrue(any(k.arg=='turn_id' and isinstance(k.value,ast.Name) and k.value.id=='turn_id' for k in call.keywords))

class ApprovalTests(unittest.IsolatedAsyncioTestCase):
    async def confirm(self, answers, plan=None):
        # Execute the exact production method without starting a browser,
        # constructing a PromptSession, or importing unrelated entry points.
        tree = ast.parse((ROOT/'cli.py').read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='AgentShell')
        method = next(node for node in cls.body if isinstance(node,ast.AsyncFunctionDef) and node.name=='confirm')
        module = ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
        output = io.StringIO()
        scope = {'json':json,'Panel':Panel,'Syntax':Syntax,'console':Console(file=output,width=120,force_terminal=False)}
        exec(compile(module, str(ROOT/'cli.py'), 'exec'), scope)
        shell = SimpleNamespace(approving=False,stop_status=Mock(),status=Mock(),prompt=SimpleNamespace(prompt_async=AsyncMock(side_effect=answers)))
        plan = plan if plan is not None else [{'action':'file_read','path':'README.md'}]
        result = await scope['confirm'](shell,plan)
        self.assertFalse(shell.approving)
        shell.status.start.assert_called_once()
        return result, output.getvalue(), shell

    async def test_yes_variants(self):
        for answer in ['y','Y','yes','YES',' Yes ']:
            with self.subTest(answer=answer):
                self.assertTrue((await self.confirm([answer]))[0])

    async def test_no_and_blank_decline(self):
        for answer in ['','n','N','no','NO']:
            with self.subTest(answer=answer):
                self.assertFalse((await self.confirm([answer]))[0])

    async def test_invalid_answer_reprompts(self):
        approved, text, shell = await self.confirm(['maybe','yes'])
        self.assertTrue(approved)
        self.assertEqual(shell.prompt.prompt_async.await_count,2)
        self.assertIn('Enter y/yes',text)

    async def test_new_file_content_only_printed_once(self):
        plan = [{'action':'file_write','path':'article.md','base_sha256':None,'content':'UNIQUE_ARTICLE_BODY','diff':'+UNIQUE_ARTICLE_BODY'}]
        approved, text, _ = await self.confirm(['yes'],plan)
        self.assertTrue(approved)
        self.assertEqual(text.count('UNIQUE_ARTICLE_BODY'),1)

    async def test_edit_shows_diff_not_complete_file_again(self):
        plan = [{'action':'file_write','path':'a.py','base_sha256':'a'*64,'content':'WHOLE_FILE_BODY','diff':'-old\n+NEW_DIFF_BODY','replacements':[{'old_text':'old','new_text':'NEW_DIFF_BODY'}]}]
        _, text, _ = await self.confirm(['yes'],plan)
        self.assertEqual(text.count('NEW_DIFF_BODY'),1)
        self.assertNotIn('WHOLE_FILE_BODY',text)

    async def test_full_arguments_can_be_requested_without_approval(self):
        approved, text, shell = await self.confirm(['v','no'])
        self.assertFalse(approved)
        self.assertEqual(shell.prompt.prompt_async.await_count,2)
        self.assertIn('README.md',text)
