"""A session's hook callbacks must not wait behind the kernel-wide registry lock (2026-10-09).

Each session runs its own asyncio loop on its own thread, and that loop is the only place the SDK answers the CLI's
hook callbacks (UserPromptSubmit, PostToolUse, Stop, SubagentStart, ...). The host transport acknowledges every record
it hands to that loop, and the acknowledgement wrote `hostAck` into the registry SYNCHRONOUSLY, from inside the record
hand-over (_read_socket -> _take -> _advance -> on_ack -> SdkBackend._write_host_ack -> _update_reg). _update_reg takes
SdkBackend._reg_lock, ONE lock for every session's registry read-modify-write, so while any other thread held it (another
session's write of a large registry, an HTTP handler, a long garbage collection inside a holder) the whole session loop
stood still: no hook was dispatched or answered until the CLI gave up on it (540 s for prompt, tool and stop hooks, 180 s
for subagent hooks), and queued prompts were refused one after another. A stack read of a live kernel showed most of its
session threads parked at exactly that frame.

These tests reproduce the stall with synthetic data only: one thread holds _reg_lock (standing in for the other
session's long write) while a session loop hands over a record and then has a hook to answer. The hook must be answered
while the lock is still held, and the acknowledgement must still land once the lock frees.
"""
import asyncio
import importlib.util
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

from romp_load import load_source

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "bin")
os.environ["XDG_STATE_HOME"] = tempfile.mkdtemp()
os.environ.pop("ROMP_STATE_DIR", None)  # a live kernel's export outranks the XDG floor
os.environ["ROMP_CLI_SCOPE"] = "0"
if importlib.util.find_spec("claude_agent_sdk") is None:
    _tag = "python%d.%d" % sys.version_info[:2]
    for _sp in sorted(Path(os.path.expanduser("~/.local/state/romp/sdkvenv/lib")).glob(_tag + "/site-packages")):
        sys.path.insert(0, str(_sp))
sb = load_source("romp_sdk_backend", os.path.join(BIN, "romp_sdk_backend.py"))
ht = sb._ht()

SID = "5e1f0a77-2222-4333-8444-0000000000c1"     # private synthetic sid (never a real session)
LOOP_PREFIX = getattr(sb, "SESSION_LOOP_THREAD_PREFIX", "sdk:")   # the session loop threads' name prefix (SdkSession.start)
HELLO = {"t": "hello", "host": {"pid": 4242, "start": "h1"}, "cli": {"pid": 4343, "start": "c1"}, "journal": {"next": 0}}


class HookCallbackNotBehindRegLock(unittest.TestCase):

    def _be(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        Path(d, "session-hosts").write_text("off")      # this test runs no host process
        logs = []
        be = sb.SdkBackend(d, "/bin/true", lambda *a, **k: None, log=logs.append)
        sb.write_reg(Path(d), SID, {"sid": SID, "name": "web", "alive": True})
        return d, be

    def _session_and_transport(self, be):
        """A session stand-in with a real HostTransport whose on_ack is the backend's own wiring (_new_host_transport),
        hello already received, as on a live attach."""
        s = types.SimpleNamespace(sid=SID, name="web", _host=None, _host_ack_t=0.0, _host_end_grace=None,
                                  _on_cli_stderr=lambda line: None)
        t = be._new_host_transport(s, "/nonexistent.sock", -1)
        t.hello = dict(HELLO)
        t.replay_end = 0
        s._host = t
        return s, t

    def _run_session_loop(self, t, answered, handed):
        """The session's loop thread: the SDK spawns a hook handler for a control request, then its reader pulls the next
        record from the transport, which hands it over (and acknowledges it) synchronously, as _read_socket does."""
        def run():
            async def main():
                async def hook_handler():                     # SdkSession's hook coroutine: answers at once
                    answered.set()
                    return {}
                task = asyncio.ensure_future(hook_handler())
                t._take({"t": "out", "offset": 7, "data": {"type": "assistant", "message": {"content": []}}})
                handed.set()
                await task
            asyncio.run(main())
        th = threading.Thread(target=run, name="sdk:web-test", daemon=True)
        th.start()
        return th

    def test_a_hook_is_answered_while_another_thread_holds_the_registry_lock(self):
        d, be = self._be()
        s, t = self._session_and_transport(be)
        answered, handed = threading.Event(), threading.Event()
        be._reg_lock.acquire()                               # another session's long registry write, in flight
        try:
            th = self._run_session_loop(t, answered, handed)
            ok = answered.wait(3.0)
            handed_ok = handed.is_set()
        finally:
            be._reg_lock.release()
        th.join(10.0)
        self.assertTrue(handed_ok, "the record hand-over blocked the session loop on the kernel-wide registry lock")
        self.assertTrue(ok, "the hook callback was not answered while another thread held _reg_lock")

    def test_the_acknowledgement_still_lands_once_the_lock_frees(self):
        d, be = self._be()
        s, t = self._session_and_transport(be)
        answered, handed = threading.Event(), threading.Event()
        be._reg_lock.acquire()
        try:
            th = self._run_session_loop(t, answered, handed)
            answered.wait(3.0)
            self.assertNotIn("hostAck", sb.read_reg(Path(d), SID) or {}, "nothing can be written while the lock is held")
        finally:
            be._reg_lock.release()
        th.join(10.0)
        deadline = time.time() + 10.0
        ack = None
        while time.time() < deadline:
            ack = (sb.read_reg(Path(d), SID) or {}).get("hostAck")
            if ack:
                break
            time.sleep(0.02)
        self.assertEqual(ack, {"host": "4242:h1", "cli": "4343:c1", "offset": 7},
                         "the deferred acknowledgement is written with the transport's offset once the lock frees")

    def test_a_forced_acknowledgement_is_still_written_before_it_returns(self):
        """The detach path's last ack (force=True) runs as the session leaves its loop; it stays synchronous, so the
        registry names the final offset by the time the session thread moves on."""
        d, be = self._be()
        s, t = self._session_and_transport(be)
        t.ack_offset = 11
        be._write_host_ack(s, force=True)
        self.assertEqual((sb.read_reg(Path(d), SID) or {}).get("hostAck"),
                         {"host": "4242:h1", "cli": "4343:c1", "offset": 11})

    def test_a_deferred_write_never_lands_for_a_transport_that_left(self):
        """The deferred writer reads the session's CURRENT transport under the registry lock: once the session has dropped
        its host (detach, exit), an acknowledgement that waited on the lock writes nothing rather than an older offset from
        a transport that is gone (which would land over the detach's forced, final ack)."""
        d, be = self._be()
        s, t = self._session_and_transport(be)
        be._reg_lock.acquire()
        try:
            # from a thread of its own: before the fix this call blocks on the held lock
            w = threading.Thread(target=be._write_host_ack, args=(s,), daemon=True)
            w.start()
            w.join(1.0)
        finally:
            s._host = None                                    # the session left its host before the write could run
            be._reg_lock.release()
        w.join(10.0)
        _drain_writer(be)
        self.assertNotIn("hostAck", sb.read_reg(Path(d), SID) or {},
                         "a write that waited on the lock while the session left its host lands nothing stale")


def _drain_writer(be, timeout=10.0):
    """Wait for the backend's registry writer (whatever the version calls it) to finish what is queued."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        w = getattr(be, "_reg_writer", None) or getattr(be, "_host_ack_thread", None)
        if w is None:
            return
        w.join(max(0.0, deadline - time.time()))


class HooksNotBehindRegLock(unittest.TestCase):
    """Every hook romp registers that writes the registry: answered while another thread holds _reg_lock, when it runs
    on a session loop thread (the name prefix the kernel gives them), and its write lands once the lock frees."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        Path(self.d, "session-hosts").write_text("off")
        self.logs = []
        self.be = sb.SdkBackend(self.d, "/bin/true", lambda *a, **k: None, log=self.logs.append)
        sb.write_reg(Path(self.d), SID, {"sid": SID, "name": "web", "cwd": self.d, "host": "TESTHOST", "alive": True})
        self.s = sb.SdkSession(self.be, sb.read_reg(Path(self.d), SID))

    def _reg(self):
        return sb.read_reg(Path(self.d), SID) or {}

    def _answer_under_held_lock(self, make_coro):
        """Hold _reg_lock, run the hook on a session-loop-named thread, and return (answered within 3 s, its answer).
        Releases the lock and drains the writer before returning, so the caller can read what landed."""
        out = {}

        def run():
            out["answer"] = asyncio.run(make_coro())
        th = threading.Thread(target=run, name=LOOP_PREFIX + "web-hooks", daemon=True)
        self.be._reg_lock.acquire()
        try:
            th.start()
            th.join(3.0)
            answered = not th.is_alive()
        finally:
            self.be._reg_lock.release()
        th.join(10.0)
        _drain_writer(self.be)
        return answered, out.get("answer")

    def test_post_tool_use_ledger_hook(self):
        inp = {"hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_use_id": "toolu_synthetic_1",
               "tool_input": {"command": "sleep 100", "run_in_background": True, "description": "a synthetic job"},
               "tool_response": {"backgroundTaskId": "bgtask-synthetic-1"}}
        answered, ans = self._answer_under_held_lock(lambda: self.s._ledger_tool_hook(inp, "toolu_synthetic_1", None))
        self.assertTrue(answered, "the PostToolUse ledger hook waited on the registry lock")
        self.assertEqual(ans, {})
        self.assertEqual([e.get("tid") for e in self._reg().get("bgLedger") or []], ["bgtask-synthetic-1"])

    def test_post_tool_use_failure_ledger_hook(self):
        launch = {"tool_name": "Bash", "tool_use_id": "toolu_synthetic_2",
                  "tool_input": {"command": "sleep 100", "run_in_background": True},
                  "tool_response": {"backgroundTaskId": "bgtask-synthetic-2"}}
        asyncio.run(self.s._ledger_tool_hook(launch, "toolu_synthetic_2", None))   # off any session loop: at once
        self.assertEqual(len(self._reg().get("bgLedger") or []), 1)
        fail = {"hook_event_name": "PostToolUseFailure", "tool_name": "Bash", "tool_use_id": "toolu_synthetic_2"}
        answered, ans = self._answer_under_held_lock(lambda: self.s._ledger_fail_hook(fail, "toolu_synthetic_2", None))
        self.assertTrue(answered, "the PostToolUseFailure hook waited on the registry lock")
        self.assertEqual(ans, {})
        self.assertEqual(self._reg().get("bgLedger"), [])
        self.assertEqual([e.get("why") for e in self._reg().get("bgLedgerEnded") or []], ["launch-failed"])

    def test_stop_hook(self):
        inp = {"hook_event_name": "Stop", "session_crons": [], "background_tasks": [], "transcript_path": "/x/t.jsonl"}
        answered, ans = self._answer_under_held_lock(lambda: self.s._stop_hook(inp, None, None))
        self.assertTrue(answered, "the Stop hook waited on the registry lock")
        self.assertEqual(ans, {})
        self.assertTrue(self._reg().get("lastStopAt"), "the turn-end stamp lands once the lock frees")

    def test_schedule_tool_hook(self):
        inp = {"hook_event_name": "PostToolUse", "tool_name": "ScheduleWakeup",
               "tool_input": {"delaySeconds": 600, "prompt": "check the synthetic build", "reason": "poll"}}
        answered, ans = self._answer_under_held_lock(lambda: self.s._sched_tool_hook(inp, None, None))
        self.assertTrue(answered, "the scheduling-tool hook waited on the registry lock")
        self.assertEqual(ans, {})
        self.assertEqual([c.get("prompt") for c in self._reg().get("sessionCrons") or []], ["check the synthetic build"])

    def test_facts_tool_hook(self):
        inp = {"hook_event_name": "PostToolUse", "tool_name": "TaskCreate", "tool_input": {"subject": "a task"},
               "tool_response": {"taskId": "7"}}
        answered, ans = self._answer_under_held_lock(lambda: self.s._facts_tool_hook(inp, None, None))
        self.assertTrue(answered, "the interaction-facts hook waited on the registry lock")
        self.assertEqual(ans, {})
        self.assertEqual([w.get("taskId") for w in self._reg().get("taskWrites") or []], ["7"])

    def test_worktree_tool_hook_transcript_path(self):
        inp = {"hook_event_name": "PostToolUse", "tool_name": "EnterWorktree", "transcript_path": "/x/moved.jsonl"}
        answered, ans = self._answer_under_held_lock(lambda: self.s._worktree_tool_hook(inp, None, None))
        self.assertTrue(answered, "the worktree hook waited on the registry lock")
        self.assertEqual(ans, {})
        self.assertEqual(self._reg().get("transcriptPath"), "/x/moved.jsonl")

    def test_background_task_mirror_never_writes_back_a_cleared_set(self):
        """The bgTasks mirror reads the live set when its write runs: a mirror queued while a task ran, then a clear
        (_drop_live_work), lands [] even though the earlier mirror was queued first."""
        with self.s._sub_lock:
            self.s._bg_tasks["task-synthetic"] = {"desc": "a synthetic task", "since": 1}

        async def mirror_then_drop():
            self.s._mirror_bg_tasks()                     # queued behind the held lock, the task still live
            self.s._drop_live_work("reconnect")          # clears the set, queues the [] write
            return {}
        answered, _ = self._answer_under_held_lock(mirror_then_drop)
        self.assertTrue(answered)
        self.assertEqual(self._reg().get("bgTasks"), [], "the cleared set is what lands, never the task it cleared")


class QueueMirrorOnTheLoop(HooksNotBehindRegLock):
    """_persist_queue on a session loop: written before it returns when the lock is free (a popped text leaves the disk
    queue before it is fed, as before), queued when the lock is busy, landing the queue as it stands then."""

    def _on_loop(self, fn):
        th = threading.Thread(target=fn, name=LOOP_PREFIX + "web-queue", daemon=True)
        th.start()
        th.join(3.0)
        return not th.is_alive()

    def test_lock_free_writes_before_returning(self):
        with self.s._lock:
            self.s._pending[:] = ["first synthetic text"]
            self.s._pending_meta[:] = [{}]
        seen = {}

        def persist():
            self.s._persist_queue()
            seen["queue"] = (sb.read_reg(Path(self.d), SID) or {}).get("queue")
        self.assertTrue(self._on_loop(persist))
        self.assertEqual(seen["queue"], ["first synthetic text"], "with the lock free the write is synchronous")

    def test_lock_busy_queues_and_lands_the_latest_queue(self):
        with self.s._lock:
            self.s._pending[:] = ["first synthetic text"]
            self.s._pending_meta[:] = [{}]
        self.be._reg_lock.acquire()
        try:
            self.assertTrue(self._on_loop(self.s._persist_queue), "the queue mirror waited on the registry lock")
            with self.s._lock:
                self.s._pending.append("second synthetic text")
                self.s._pending_meta.append({})
        finally:
            self.be._reg_lock.release()
        _drain_writer(self.be)
        self.assertEqual(self._reg().get("queue"), ["first synthetic text", "second synthetic text"])

    # the inherited hook tests run once, in HooksNotBehindRegLock
    test_post_tool_use_ledger_hook = test_post_tool_use_failure_ledger_hook = test_stop_hook = None
    test_schedule_tool_hook = test_facts_tool_hook = test_worktree_tool_hook_transcript_path = None
    test_background_task_mirror_never_writes_back_a_cleared_set = None


class SessionLoopRegLockGuard(unittest.TestCase):
    """The test-time rule: a session loop thread that takes the registry lock raises (ROMP_REG_LOCK_GUARD=raise, which
    tests/conftest.py sets for the suite), so a hook or handler that writes the registry on the loop fails its test."""

    def setUp(self):
        self._before = os.environ.get(sb.REG_LOCK_GUARD_ENV)
        self._log_before = os.environ.get(sb.REG_LOCK_GUARD_LOG_ENV)
        os.environ[sb.REG_LOCK_GUARD_ENV] = "raise"
        self.addCleanup(self._restore)
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        # these tests take the lock on a loop ON PURPOSE: their hits go to a private file, never the run's
        self.hits = os.path.join(self.d, "hits.tsv")
        os.environ[sb.REG_LOCK_GUARD_LOG_ENV] = self.hits
        Path(self.d, "session-hosts").write_text("off")
        self.be = sb.SdkBackend(self.d, "/bin/true", lambda *a, **k: None, log=lambda *a, **k: None)

    def _restore(self):
        for name, before in ((sb.REG_LOCK_GUARD_ENV, self._before), (sb.REG_LOCK_GUARD_LOG_ENV, self._log_before)):
            if before is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = before

    def _on_thread(self, name, fn):
        out = {}

        def run():
            try:
                fn()
                out["ok"] = True
            except Exception as e:
                out["error"] = e
        th = threading.Thread(target=run, name=name, daemon=True)
        th.start()
        th.join(10.0)
        return out

    def test_a_registry_write_on_a_session_loop_raises(self):
        out = self._on_thread(LOOP_PREFIX + "web", lambda: self.be._update_reg(SID, hostAck={"offset": 1}))
        self.assertIsInstance(out.get("error"), sb.RegLockOnSessionLoop)
        self.assertNotIn("hostAck", sb.read_reg(Path(self.d), SID) or {})
        self.assertTrue(Path(self.hits).read_text().strip(), "the hit is on file too, so a swallowed raise still fails the run")

    def test_an_owed_writer_is_let_through_and_a_try_never_raises(self):
        """The writers named in REG_LOCK_LOOP_WRITERS_OWED pass (a name per writer still to move), and a non-blocking
        try (the queue mirror's) never waits, so it is allowed on a loop."""
        self.assertIn("_persist_cost_state", sb.REG_LOCK_LOOP_WRITERS_OWED)
        self.assertNotIn("_ledger_tool_apply", sb.REG_LOCK_LOOP_WRITERS_OWED)
        out = self._on_thread(LOOP_PREFIX + "web", lambda: self.be._update_reg_try(SID, lastStopAt=3))
        self.assertTrue(out.get("ok"), out)
        self.assertEqual((sb.read_reg(Path(self.d), SID) or {}).get("lastStopAt"), 3)

    def test_the_same_write_elsewhere_is_allowed(self):
        out = self._on_thread("romp-test-worker", lambda: self.be._update_reg(SID, hostAck={"offset": 1}))
        self.assertTrue(out.get("ok"), out)
        self.assertEqual((sb.read_reg(Path(self.d), SID) or {}).get("hostAck"), {"offset": 1})

    def test_queued_work_from_a_session_loop_does_not_raise(self):
        out = self._on_thread(LOOP_PREFIX + "web",
                              lambda: self.be._reg_job(None, lambda: self.be._update_reg(SID, lastStopAt=5)))
        self.assertTrue(out.get("ok"), out)
        _drain_writer(self.be)
        self.assertEqual((sb.read_reg(Path(self.d), SID) or {}).get("lastStopAt"), 5)


if __name__ == "__main__":
    unittest.main()
