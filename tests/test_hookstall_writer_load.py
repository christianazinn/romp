"""Review 5 of hook-callback-stall, must-fix 2 (synthetic data only): the registry writer under the overload review 5
measured. Twenty sessions stream records (the real _write_host_ack: its once-a-second throttle and carry), each queues a
hook write every 3 s and a keyed queue mirror every 2 s, and every writer item costs 60 ms (review 4's measured cost under
load), so the read-position saves alone want more than the writer has.

Two things must hold, and the bounds are generous on purpose (this asserts that starvation is gone, not a latency, so it
holds at a load average of 40 to 60):
- other writes keep landing: at least half of the hook writes issued have run, and none waited 10 s or more (the
  back-of-queue tree, fb5c44b65, measured a median of about 1.6 s; with the saves at the front, 5420787e7 ran none);
- every session's read position keeps being saved: each session's saved offset trails what it consumed by under 10 s of
  records (fb5c44b65 saved none of them while the writer lagged; 5420787e7 never saved a few of them).
"""
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.realpath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from romp_load import load_source  # noqa: E402

sb = load_source("hookstall_sb_writer_load", os.path.join(TREE, "bin", "romp_sdk_backend.py"))

NSESS, RATE, HOOK_EVERY, MIRROR_EVERY, ITEM_S, SECS = 20, 50.0, 3.0, 2.0, 0.06, 15.0
WAIT_BOUND_S = 10.0
LAG_BOUND = int(RATE * 10.0)          # records: ten seconds of streaming


class FakeHost:
    def __init__(self, i):
        self.hello = {"host": {"pid": 5000 + i, "start": "h%d" % i}, "cli": {"pid": 6000 + i, "start": "c%d" % i}}
        self.exit_info = None
        self.ack_offset = -1


class WriterUnderLoad(unittest.TestCase):
    def test_saves_and_other_writes_both_keep_moving(self):
        d = tempfile.mkdtemp(prefix="hookstall-load-")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        Path(d, "session-hosts").write_text("off")
        be = sb.SdkBackend(d, "/bin/true", lambda *a, **k: None, log=lambda *a, **k: None)
        sids = ["11111111-2222-4333-8444-%012d" % (300 + i) for i in range(NSESS)]
        filler = [{"md": "synthetic text " * 20, "qid": "q%d" % j} for j in range(20)]
        for s in sids:
            sb.write_reg(Path(d), s, {"sid": s, "name": "web", "alive": True, "echoes": filler})
        orig_merge = be._reg_merge_locked

        def slow_merge(sid, make_fields):     # every WRITER item costs ITEM_S inside the lock (a slow RMW under load)
            if threading.current_thread().name == "romp-reg-writer":
                time.sleep(ITEM_S)
            return orig_merge(sid, make_fields)
        be._reg_merge_locked = slow_merge

        sessions = [types.SimpleNamespace(sid=s, name="web", _host=FakeHost(i), _host_ack_t=0.0)
                    for i, s in enumerate(sids)]
        stop = threading.Event()
        lk = threading.Lock()
        pending = {}      # token -> time queued
        waits = []        # how long each hook write that ran had waited

        def hook_item(tok, sid):
            t = time.monotonic()
            with lk:
                q = pending.pop(tok, None)
            if q is not None:
                waits.append(t - q)
            be._update_reg(sid, lastSkill={"at": tok, "name": "x"})

        def loop(i):
            sess = sessions[i]
            n = 0
            now = time.monotonic()
            next_hook = now + HOOK_EVERY * i / NSESS
            next_mirror = now + MIRROR_EVERY * i / NSESS
            while not stop.is_set():
                sess._host.ack_offset += 1
                be._write_host_ack(sess)                  # the transport's per-record on_ack
                now = time.monotonic()
                if now >= next_hook:
                    next_hook = now + HOOK_EVERY
                    n += 1
                    tok = i * 1_000_000 + n
                    with lk:
                        pending[tok] = now
                    be._reg_job(None, lambda t=tok, s=sess.sid: hook_item(t, s), sid=sess.sid)
                if now >= next_mirror:                    # a keyed mirror (the queue mirror's shape)
                    next_mirror = now + MIRROR_EVERY
                    be._reg_job(("queue", sess.sid), lambda s=sess.sid, q=now: be._update_reg(s, queueMirror={"q": q}))
                time.sleep(1.0 / RATE)

        ths = [threading.Thread(target=loop, args=(i,), name="sdk:web%d" % i, daemon=True) for i in range(NSESS)]
        for t in ths:
            t.start()
        time.sleep(SECS)
        lags = []
        for sess in sessions:
            ack = (sb.read_reg(Path(d), sess.sid) or {}).get("hostAck") or {}
            lags.append(sess._host.ack_offset - int(ack.get("offset", -1)))
        stop.set()
        t_end = time.monotonic()
        with lk:
            still = list(pending.values())
            ran = list(waits)
        for t in ths:
            t.join(5)
        issued = len(ran) + len(still)
        oldest = max([t_end - q for q in still] + ran + [0.0])
        print("\n[writer load] %d sessions, %.0f ms an item, %.0f s: other writes ran %d of %d, oldest wait %.2f s; "
              "per-session save lag (records, sorted) %s" % (NSESS, ITEM_S * 1000, SECS, len(ran), issued, oldest,
                                                            sorted(lags)))
        self.assertGreater(issued, 0)
        self.assertGreaterEqual(len(ran), issued / 2.0, "other writes starved: %d of %d ran" % (len(ran), issued))
        self.assertLess(oldest, WAIT_BOUND_S, "another write waited %.1f s" % oldest)
        trailing = [x for x in lags if x >= LAG_BOUND]
        self.assertEqual(trailing, [], "%d sessions' saved read position trails by %d or more records: %s"
                         % (len(trailing), LAG_BOUND, sorted(lags)))


if __name__ == "__main__":
    unittest.main()
