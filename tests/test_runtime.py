import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from protocol import parse_response, ProtocolError
from journal import Journal
from tools import ToolExecutor, digest
from agent_core import AgentLoop
from privacy import outbound_text
from provider_adapters import Snapshot
from reliable_browser import BrowserBridge

def envelope(actions=None, message=None, turn='t'):
    obj = {'protocol': 'browser-agent/1', 'turn_id': turn, 'kind': 'final' if message is not None else 'actions'}
    obj['message' if message is not None else 'actions'] = message if message is not None else actions
    return json.dumps(obj)

def action(tool, args, id='a1'):
    return {'id': id, 'tool': tool, 'arguments': args}

class ProtocolTests(unittest.TestCase):
    def test_opaque_markers(self):
        payload = '[[[SHELL]]]\necho not-an-action\n[[[END]]]'
        obj = parse_response(envelope([action('write_file', {'path':'README.md', 'content':payload, 'expected_sha256':None})]), 't')
        self.assertEqual(len(obj['actions']), 1)
        self.assertEqual(obj['actions'][0]['arguments']['content'], payload)

    def test_bad_envelopes(self):
        good = envelope([action('read_file', {'path':'a'})])
        for bad in [good[:-1], good+good, 'example: '+good, good.replace('"t"', '"other"'), good.replace('"path": "a"', '"path":"a","extra":0'), good.replace('"path": "a"', '"path":"a","path":"b"')]:
            with self.subTest(bad=bad), self.assertRaises(ProtocolError):
                parse_response(bad, 't')

    def test_completion_is_typed(self):
        with self.assertRaises(ProtocolError):
            parse_response('TASK_COMPLETE', 't')
        obj = parse_response(envelope(message='Quoted TASK_COMPLETE'), 't')
        self.assertEqual(obj['kind'], 'final')

    def test_no_partial_batch(self):
        with self.assertRaises(ProtocolError):
            parse_response(envelope([action('read_file', {'path':'a'}), action('unknown', {}, 'b')]), 't')

    def test_duplicate_ids(self):
        with self.assertRaises(ProtocolError):
            parse_response(envelope([action('read_file', {'path':'a'})]*2), 't')

    def test_integer_not_boolean(self):
        with self.assertRaises(ProtocolError):
            parse_response(envelope([action('read_file', {'path':'a','start_line':True})]), 't')

class ToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tools = ToolExecutor({'workspace': str(self.root), 'max_output_bytes': 10000})

    def tearDown(self):
        self.temp.cleanup()

    async def test_exact_write_and_empty(self):
        for i, text in enumerate(['python\n\nhello  \n```', '', '\r\nhello\r\n', '[[[SHELL]]]\necho no\n[[[END]]]']):
            a = action('write_file', {'path':str(i), 'content':text, 'expected_sha256':None})
            result = await self.tools.execute(a)
            self.assertNotIn('error', result)
            self.assertEqual((self.root/str(i)).read_bytes(), text.encode())

    async def test_stale_and_approved_version(self):
        path = self.root/'a'
        path.write_bytes(b'one\ntwo\n')
        a = action('edit_file', {'path':'a','expected_sha256':digest(path.read_bytes()),'replacements':[{'old_text':'two','new_text':'three'}]})
        prepared = self.tools.prepare(a)
        path.write_bytes(b'external')
        result = await self.tools.execute(a, prepared)
        self.assertIn('error',result)
        self.assertEqual(path.read_bytes(),b'external')

    async def test_unique_and_nonoverlap(self):
        path = self.root/'a'
        path.write_text('abcabc',encoding='utf-8')
        with self.assertRaises(ValueError):
            self.tools.prepare(action('edit_file', {'path':'a','expected_sha256':digest(path.read_bytes()),'replacements':[{'old_text':'abc','new_text':'x'}]}))
        path.write_text('abcdef',encoding='utf-8')
        with self.assertRaises(ValueError):
            self.tools.prepare(action('edit_file', {'path':'a','expected_sha256':digest(path.read_bytes()),'replacements':[{'old_text':'abc','new_text':'x'},{'old_text':'cde','new_text':'y'}]}))

    async def test_edit_and_backup(self):
        path = self.root/'a'
        path.write_bytes(b'abc\r\ndef')
        a = action('edit_file',{'path':'a','expected_sha256':digest(path.read_bytes()),'replacements':[{'old_text':'abc','new_text':'xyz'}]})
        result = await self.tools.execute(a)
        self.assertEqual(path.read_bytes(), b'xyz\r\ndef')
        self.assertEqual((self.tools.state_dir/'backups'/result['backup_id']).read_bytes(), b'abc\r\ndef')

    async def test_paths_and_sensitive_reads(self):
        for p in ['../escape', str(self.root/'absolute'), '.env', '.browser-agent/journal.sqlite3']:
            self.assertIn('error', await self.tools.execute(action('read_file',{'path':p})))

    async def test_ranges(self):
        for newline in (b'\n', b'\r\n'):
            with self.subTest(newline=newline):
                raw = newline.join([b'one', b'two', b'three', b''])
                (self.root / 'a').write_bytes(raw)

                result = await self.tools.execute(
                    action(
                        'read_file',
                        {'path': 'a', 'start_line': 2, 'end_line': 2},
                    )
                )

                self.assertNotIn('error', result)
                self.assertEqual(
                    result['content'],
                    '2|two' + newline.decode('ascii'),
                )
                self.assertEqual(result['next_line'], 3)
                self.assertEqual(result['sha256'], digest(raw))
                self.assertEqual((self.root / 'a').read_bytes(), raw)

    async def test_search_and_list(self):
        (self.root/'a.py').write_text('one\ntwo\n',encoding='utf-8')
        found = await self.tools.execute(action('search_text',{'pattern':'two','glob':'*.py'}))
        self.assertEqual(found['matches'][0]['line'],2)
        listing = await self.tools.execute(action('list_files',{}))
        self.assertEqual(listing['files'],['a.py'])

    async def test_bounded_stream_and_output_paging(self):
        chunks = []
        async def emit(text): chunks.append(text)
        result = await self.tools.execute(action('run_process',{'argv':[sys.executable,'-c','print("x"*20000)']}),emit=emit)
        self.assertEqual(result['stored_bytes'],10000)
        self.assertGreater(result['discarded_bytes'],0)
        self.assertTrue(chunks)
        page = await self.tools.execute(action('read_output',{'output_id':result['output_id'],'offset':50,'limit':10}))
        self.assertEqual(page['bytes'],10)
        self.assertEqual(page['next_offset'],60)

    async def test_timeout_and_cancel(self):
        args = {'argv':[sys.executable,'-c','import time; time.sleep(20)'],'timeout':1}
        result = await self.tools.execute(action('run_process',args))
        self.assertTrue(result['timed_out'])
        task = asyncio.create_task(self.tools.execute(action('run_process',{**args,'timeout':30})))
        await asyncio.sleep(.2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task

    async def test_shell_is_explicit(self):
        result = await self.tools.execute(action('run_shell',{'command':'echo shell-ok'}))
        self.assertNotIn('error',result)
        self.assertIn('shell-ok',result['output'])

class JournalTests(unittest.TestCase):
    def test_replay_and_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            j = Journal(directory)
            run = j.start('task')
            a = action('run_shell',{'command':'echo hi'})
            self.assertIsNone(j.claim(run,a))
            with self.assertRaises(RuntimeError): j.claim(run,a)
            j.finish(run,a,{'exit_code':0})
            self.assertEqual(j.claim(run,a),{'exit_code':0})
            j.close()
            j = Journal(directory)
            self.assertEqual(j.start('task',run),run)
            self.assertEqual(j.lookup(run,a),{'exit_code':0})
            with self.assertRaises(ValueError): j.lookup(run,action('run_shell',{'command':'different'}))
            j.close()

class FakeBridge:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []
    async def send_message(self,text): self.messages.append(text)
    async def wait_for_response(self,**kwargs):
        turn = json.loads(self.messages[-1].split('\n')[-1])['turn_id']
        response = self.responses.pop(0)
        return response(turn) if callable(response) else response
    async def new_chat(self): pass

class LoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_fail_closed_and_no_guessing(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {'workspace':directory, 'protocol_repair_limit':0}
            tools = ToolExecutor(config)
            bridge = FakeBridge(['def main():\n    print("hello.py")'])
            loop = AgentLoop(bridge,tools,config)
            events = [e async for e in loop.run_turn('Create hello.py')]
            self.assertFalse((Path(directory)/'hello.py').exists())
            self.assertTrue(any(e.kind=='error' for e in events))
            loop.close()

    async def test_approval_and_event_order(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {'workspace':directory}
            a = action('write_file',{'path':'a','content':'python\n','expected_sha256':None})
            bridge = FakeBridge([lambda t:envelope([a],turn=t),lambda t:envelope(message='done',turn=t)])
            loop = AgentLoop(bridge,ToolExecutor(config),config)
            events = [e async for e in loop.run_turn('create')]
            self.assertFalse((Path(directory)/'a').exists())
            loop.close()
            async def approve(plan):
                self.assertEqual(plan[0]['content'],'python\n')
                return True
            bridge = FakeBridge([lambda t:envelope([a],turn=t),lambda t:envelope(message='done',turn=t)])
            loop = AgentLoop(bridge,ToolExecutor(config),config,confirm=approve)
            gen = loop.run_turn('create')
            kinds = []
            async for e in gen:
                kinds.append(e.kind)
                if e.kind == 'tool_start': self.assertFalse((Path(directory)/'a').exists())
            self.assertEqual((Path(directory)/'a').read_bytes(),b'python\n')
            self.assertLess(kinds.index('tool_start'),kinds.index('tool_result'))
            loop.close()

    async def test_completed_write_replays_without_stale_error(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {'workspace':directory,'confirm_changes':False}
            a = action('write_file',{'path':'a','content':'x','expected_sha256':None})
            bridge = FakeBridge([lambda t:envelope([a],turn=t),lambda t:envelope([a],turn=t),lambda t:envelope(message='done',turn=t)])
            loop = AgentLoop(bridge,ToolExecutor(config),config)
            events = [e async for e in loop.run_turn('create')]
            self.assertEqual(sum(e.kind=='tool_start' for e in events),1)
            self.assertTrue(events[-1].data['complete'])
            loop.close()

    def test_outbound_credentials(self):
        for text in ['api_key=abcdef', '-----BEGIN PRIVATE KEY-----', 'ghp_'+'a'*30]:
            with self.assertRaises(ValueError): outbound_text(text)

class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_error_is_ignored(self):
        from provider_adapters import ProviderAdapter
        class Adapter(ProviderAdapter):
            async def errors(self): return {'old rejection'}
        adapter = Adapter(None,{})
        await adapter.prepare()
        self.assertEqual(await adapter.rejection(),'')

    async def test_requires_completion_evidence(self):
        class Adapter:
            async def rejection(self): return ''
            async def snapshot(self): return Snapshot(1,'new','{"a":1}',False,False)
        bridge = object.__new__(BrowserBridge)
        bridge.config = {'response_settle_seconds':0}
        bridge._locked = True
        bridge._saw_stop = False
        bridge._baseline = Snapshot(0,'','',False,False)
        bridge._adapter = Adapter()
        bridge.debug = False
        with self.assertRaises(TimeoutError): await bridge.wait_for_response(timeout_seconds=.1)
        self.assertTrue(bridge._locked)

if __name__ == '__main__':
    unittest.main()
