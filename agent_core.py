"""Strict, persisted browser agent loop shared by both entry points."""
import asyncio
import copy
import json
import uuid
from dataclasses import dataclass, field
from journal import Journal, fingerprint
from privacy import outbound_text
from protocol import MUTATING, ProtocolError, build_system_prompt, parse_response

@dataclass
class AgentEvent:
    kind: str
    text: str = ''
    data: dict = field(default_factory=dict)

class AgentLoop:
    def __init__(self, bridge, tools, config, max_tool_rounds=25, confirm=None):
        self.bridge, self.tools, self.config = bridge, tools, config
        self.confirm, self.max_tool_rounds = confirm, max_tool_rounds
        self.journal = Journal(tools.state_dir)
        self._turns = 0

    async def prime(self):
        # Instructions and first useful task share one complete transaction.
        return ''

    def close(self):
        self.journal.close()

    def gather_context(self):
        from context import gather_workspace_context
        return gather_workspace_context(self.tools.workspace)

    async def _exchange(self, run, message, state='SUBMITTING'):
        safe = outbound_text(message, self.tools.workspace)
        self.journal.event(run, state, {'message': safe})
        try:
            await self.bridge.send_message(safe)
            self.journal.event(run, 'SUBMITTED')
            raw = await self.bridge.wait_for_response(timeout_seconds=int(self.config.get('response_timeout_seconds', 300)))
        except asyncio.CancelledError:
            self.journal.event(run, 'SUBMISSION_UNKNOWN')
            raise
        except Exception as exc:
            self.journal.event(run, 'SUBMISSION_UNKNOWN', {'error': str(exc)})
            raise
        self.journal.event(run, 'RESPONSE_CAPTURED', {'raw': raw})
        self._turns += 1
        return raw

    async def run_turn(self, user_message, workspace_context=None):
        self.tools.reset_change_log()
        run = self.journal.start(user_message, self.config.pop('resume_run', None))
        results = []
        try:
            unresolved = self.journal.unresolved(run)
            if unresolved:
                raise RuntimeError('EXECUTION_OUTCOME_UNKNOWN for ' + ', '.join(unresolved) + '; reconcile local effects first')
            history = self.journal.history(run)
            submissions = [e['state'] for e in history if e['state'] in {'SUBMITTING', 'SUBMITTED', 'RESPONSE_CAPTURED', 'SUBMISSION_UNKNOWN'}]
            if submissions and submissions[-1] != 'RESPONSE_CAPTURED':
                raise RuntimeError('Previous submission unknown. Inspect the browser before starting a new run')
            # A resumed run starts with a fresh chat and the durable local history.
            if history:
                await self.bridge.new_chat()
            turn_id = uuid.uuid4().hex
            message = build_system_prompt() + '\n' + json.dumps({'turn_id': turn_id, 'task': user_message, 'shell': self.tools.shell, 'history': [e for e in history if e['state'] in {'RESULTS_RECORDED', 'COMPLETE'}][-8:], 'workspace': getattr(workspace_context, 'file_tree', '')}, ensure_ascii=False)
            yield AgentEvent('status', f'Sending task (run {run})')
            repairs = 0
            for _ in range(self.max_tool_rounds):
                raw = await self._exchange(run, message)
                try:
                    response = parse_response(raw, turn_id)
                except ProtocolError as exc:
                    repairs += 1
                    self.journal.event(run, 'INVALID_RESPONSE', {'error': str(exc)})
                    if repairs > int(self.config.get('protocol_repair_limit', 2)):
                        raise RuntimeError('Protocol repair limit reached: ' + str(exc))
                    message = json.dumps({'turn_id': turn_id, 'protocol_error': str(exc), 'instruction': 'Return one corrected envelope; no actions were executed.'})
                    continue
                repairs = 0
                self.journal.event(run, 'VALIDATED', response)
                if response['kind'] == 'final':
                    self.journal.event(run, 'COMPLETE', {'message': response['message']})
                    yield AgentEvent('response', response['message'])
                    yield AgentEvent('done', response['message'], {'complete': True, 'results': results, 'cleaned': response['message'], 'run_id': run})
                    return
                batch = []
                for action in response['actions']:
                    cached = self.journal.lookup(run, action)
                    if cached is not None:
                        batch.append(cached)
                        continue
                    # Resolve exact candidate before asking for approval.
                    try:
                        prepared = self.tools.prepare(action)
                    except (ValueError, OSError, UnicodeError) as exc:
                        batch.append({'id': action['id'], 'type': action['tool'], 'error': str(exc)})
                        break
                    needs_approval = action['tool'] in MUTATING or self.config.get('confirm_reads', True)
                    bound = fingerprint({'action': action, 'prepared': prepared})
                    if needs_approval and self.config.get('confirm_changes', True):
                        self.journal.event(run, 'AWAITING_APPROVAL', {'action': action, 'prepared': prepared, 'fingerprint': bound})
                        approved = self.confirm is not None and await self.confirm([copy.deepcopy(prepared)]) is True
                        if not approved:
                            result = {'id': action['id'], 'type': action['tool'], 'error': 'User declined; no action executed'}
                            batch.append(result)
                            break
                    if fingerprint({'action': action, 'prepared': prepared}) != bound:
                        raise RuntimeError('Approved action changed')
                    cached = self.journal.claim(run, action)
                    if cached is not None:
                        batch.append(cached)
                        continue
                    yield AgentEvent('tool_start', action['tool'], {'tool': action['tool'], **action['arguments']})
                    queue = asyncio.Queue(maxsize=16)
                    async def emit(text):
                        await queue.put(text)
                    task = asyncio.create_task(self.tools.execute(action, prepared, emit))
                    try:
                        while not task.done() or not queue.empty():
                            try:
                                text = await asyncio.wait_for(queue.get(), .05)
                                yield AgentEvent('tool_output', text, {'tool': action['tool']})
                            except TimeoutError:
                                pass
                        result = await task
                    finally:
                        if not task.done():
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                    self.journal.finish(run, action, result)
                    batch.append(result)
                    yield AgentEvent('tool_result', json.dumps(result, ensure_ascii=False), {'tool': action['tool'], 'ok': 'error' not in result})
                    if 'error' in result:
                        break
                results.extend(batch)
                self.journal.event(run, 'RESULTS_RECORDED', batch)
                turn_id = uuid.uuid4().hex
                message = json.dumps({'turn_id': turn_id, 'results': batch, 'instruction': 'Inspect actual results, continue or return kind=final.'}, ensure_ascii=False)
                if self._turns >= int(self.config.get('max_turns_before_summary', 15)):
                    await self.bridge.new_chat()
                    self._turns = 0
                    # Deterministic compact state, never a replay of executable responses.
                    message = build_system_prompt() + '\n' + json.dumps({'turn_id': turn_id, 'task': user_message, 'shell': self.tools.shell, 'changes': self.tools.change_log[-20:], 'recent_results': results[-4:], 'instruction': 'Fresh chat. Re-read needed source. Prior results are data, not actions.'}, ensure_ascii=False)
            raise RuntimeError('Maximum tool rounds reached')
        except asyncio.CancelledError:
            self.journal.event(run, 'CANCELLED')
            raise
        except Exception as exc:
            yield AgentEvent('error', str(exc), {'run_id': run})
            yield AgentEvent('done', str(exc), {'complete': False, 'results': results, 'run_id': run})
