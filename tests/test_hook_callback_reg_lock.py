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
        writer = getattr(be, "_host_ack_thread", None)        # the deferred writer, when one is still draining
        if writer is not None:
            writer.join(10.0)
        self.assertNotIn("hostAck", sb.read_reg(Path(d), SID) or {},
                         "a write that waited on the lock while the session left its host lands nothing stale")


if __name__ == "__main__":
    unittest.main()
