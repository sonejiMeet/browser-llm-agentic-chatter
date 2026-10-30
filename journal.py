"""Durable calls scoped by host turn, with legacy-record compatibility."""
import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

class Journal:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.directory / "journal.sqlite3")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, task TEXT NOT NULL, state TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, run TEXT, state TEXT, data TEXT, at REAL);
        CREATE TABLE IF NOT EXISTS calls(run TEXT, id TEXT, fingerprint TEXT, state TEXT, result TEXT, PRIMARY KEY(run,id));
        """)
        self.db.commit()

    def start(self, task, resume=None):
        if resume:
            row = self.db.execute("SELECT task FROM runs WHERE id=?", (resume,)).fetchone()
            if not row or row[0] != task:
                raise ValueError("Resume requires the original run ID and exact task")
            return resume
        run = uuid.uuid4().hex
        self.db.execute("INSERT INTO runs VALUES(?,?,?)", (run, task, "CREATED"))
        self.db.commit()
        return run

    def event(self, run, state, data=None):
        self.db.execute("UPDATE runs SET state=? WHERE id=?", (state, run))
        self.db.execute("INSERT INTO events(run,state,data,at) VALUES(?,?,?,?)", (run, state, canonical(data), time.time()))
        self.db.commit()

    @staticmethod
    def _call_key(action, turn_id):
        # Existing unscoped records remain readable; active runs pass the
        # host-generated, validated turn ID explicitly. No history is erased.
        if turn_id is None:
            return action["id"]
        if not isinstance(turn_id, str) or not turn_id:
            raise ValueError("A nonempty host turn ID is required")
        return canonical([turn_id, action["id"]])

    def lookup(self, run, action, turn_id=None):
        fp = fingerprint(action)
        key = self._call_key(action, turn_id)
        row = self.db.execute("SELECT fingerprint,state,result FROM calls WHERE run=? AND id=?", (run, key)).fetchone()
        if row:
            if row[0] != fp:
                raise ValueError("Action ID reused with different arguments in the same turn")
            if row[1] != "DONE":
                raise RuntimeError("Execution outcome unknown; inspect local effects before continuing")
            return json.loads(row[2])
        return None

    def claim(self, run, action, turn_id=None):
        cached = self.lookup(run, action, turn_id)
        if cached is not None:
            return cached
        fp = fingerprint(action)
        self.db.execute("INSERT INTO calls VALUES(?,?,?,?,NULL)", (run, self._call_key(action, turn_id), fp, "EXECUTING"))
        self.db.commit()
        return None

    def finish(self, run, action, result, turn_id=None):
        self.db.execute("UPDATE calls SET state='DONE',result=? WHERE run=? AND id=?", (canonical(result), run, self._call_key(action, turn_id)))
        self.db.commit()

    def history(self, run):
        return [{"state": state, "data": json.loads(data)} for state, data in self.db.execute("SELECT state,data FROM events WHERE run=? ORDER BY seq", (run,))]

    def unresolved(self, run):
        return [r[0] for r in self.db.execute("SELECT id FROM calls WHERE run=? AND state!='DONE'", (run,))]

    def close(self):
        self.db.close()
