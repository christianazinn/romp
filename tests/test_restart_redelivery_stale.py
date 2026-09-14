#!/usr/bin/env python3
"""Re-delivery has an AGE LINE and a REFUSED flag (2026-09-12). The boot and dead-spawn re-delivery arm
(_mark_dropped_echoes) re-feeds a human send whose text the transcript scan cannot find. A landing the scan
cannot see is re-fed at EVERY restart, forever: a kernel whose scan read only native user records (never the
queued_command attachment a mid-turn feed lands as) and only the last 2 MB of the file re-fed the same texts,
some days old, at two restarts in one night, one text landing six times. So: a send older than
REDELIVER_MAX_AGE_S at the restart is not re-fed and not scanned — it takes the flag path (dropped, kept in
the chat as never-delivered) and ONE notice names the count and the oldest stamp; a send the prompt gate
REFUSED is flagged dropped + refused at the refusal, so it is never re-fed; both flags ride the registry
mirror across restarts. The queue proper (sends never fed) is not under the line. SYNTHETIC."""
import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock
from romp_load import load_source

HERE = os.path.dirname(os.path.realpath(__file__))
BIN = os.path.join(os.path.dirname(HERE), "bin")

os.environ["XDG_STATE_HOME"] = tempfile.mkdtemp()
os.environ.pop("ROMP_STATE_DIR", None)
sb = load_source("romp_sdk_backend_stale0", os.path.join(BIN, "romp-event-model"))
sb = load_source("romp_sdk_backend_stale", os.path.join(HERE, "..", "kernel", "sdk_backend.py"))

SID = "11111111-2222-3333-4444-666666666666"
NOW = FRESH_T = STALE_T = 0              # set per test by Fixture.setUp (_reset_clock), never at import


def _reset_clock():
    """Every stamp is relative to the clock the TEST runs on, read in setUp — never at import. pytest imports
    each module at collection and runs this one minutes later under the full suite, while the code under test
    reads time.time() at the call: an import-time NOW put the send planted 120 s INSIDE the line past it by the
    time the arm ran (CI 2026-09-12: both sends dropped, the standalone run green at every timezone)."""
    global NOW, FRESH_T, STALE_T
    NOW = int(time.time())
    FRESH_T = NOW - 60                       # a minute before the restart: inside any sane line
    STALE_T = NOW - 3 * 86400                # three days before it: the pile the field measured


class Fixture(unittest.TestCase):
    def setUp(self):
        _reset_clock()
        self.td = tempfile.mkdtemp()
        os.environ["CLAUDE_CONFIG_DIR"] = os.path.join(self.td, "claude")
        self.cwd = os.path.join(self.td, "proj")
        os.makedirs(self.cwd, exist_ok=True)
        tp = sb.transcript_path(self.cwd, SID)
        os.makedirs(os.path.dirname(tp), exist_ok=True)
        self.tpath = tp
        open(tp, "w").close()                       # an empty transcript: nothing has landed

        class BE:
            state_dir = None
            _live = {}
            _reg_lock = __import__("threading").RLock()
            _live_lock = __import__("threading").RLock()
            _persisted = []
            _logs = []
            _problems = []
            _forgotten = []

            def _log(self, msg, problem=False, **kw):
                self._logs.append(msg)
                if problem:
                    self._problems.append(msg)

            def _persist_echoes(self, sid):
                self._persisted.append(sid)

            def _wake_push(self):
                pass

            def _touch_live(self, sid):
                pass

            def forget_fed(self, sid, uuid_):
                self._forgotten.append(uuid_)
        self.be = BE()
        import pathlib
        self.be.state_dir = pathlib.Path(self.td)
        for name in ("_mark_dropped_echoes", "_text_landed", "mark_echo_refused"):
            setattr(self.be, name, getattr(sb.SdkBackend, name).__get__(self.be))
        sb.write_reg(self.be.state_dir, SID, {"sid": SID, "alive": True, "cwd": self.cwd,
                                              "lastSid": SID, "queue": []})
        self._line = sb.REDELIVER_MAX_AGE_S

    def tearDown(self):
        self.be._live.clear()
        sb.REDELIVER_MAX_AGE_S = self._line
        os.environ.pop("CLAUDE_CONFIG_DIR", None)

    def _echo(self, text, t, author="human"):
        d = self.be._live.setdefault(SID, {})
        key = "echo:%d" % (len(d) + 1)               # one key per stash: two echoes may wear one text
        d[key] = {"_echo_text": text, "author": author, "t": t, "uuid": key}
        return d[key]

    def _reg_queue(self):
        return (sb.read_reg(self.be.state_dir, SID) or {}).get("queue") or []


class TheAgeLine(Fixture):
    def test_a_send_past_the_line_is_not_refed_and_one_notice_names_the_count_and_the_oldest(self):
        fresh = self._echo("fresh typed words", FRESH_T)
        stale = self._echo("three day old words", STALE_T)
        self.be._mark_dropped_echoes(SID, [])
        q = self._reg_queue()
        self.assertEqual(q[0], "fresh typed words", "the fresh send is re-delivered exactly as before")
        self.assertEqual(len(q), 2, "…followed by ONE notice, and nothing else: %r" % (q,))
        notice = q[1]
        self.assertIn("[romp]", notice)
        self.assertIn("1 queued message from before the restart was dropped as stale", notice)
        self.assertIn(time.strftime("%Y-%m-%d %H:%M", time.localtime(STALE_T)), notice,
                      "the notice names the oldest dropped send's stamp")
        self.assertNotIn("three day old words", q, "the stale send never re-enters the queue")
        self.assertTrue(stale.get("dropped") and stale.get("stale"),
                        "the stale send takes the flag path and says why")
        self.assertNotIn("dropped", fresh, "the fresh send is queued, not lost")
        self.assertEqual(len([p for p in self.be._problems if "age line" in p]), 1,
                         "one problem row for the whole drop, not one per send")

    def test_two_stale_sends_make_one_notice_with_the_oldest_stamp(self):
        self._echo("older stale words", STALE_T - 3600)
        self._echo("newer stale words", STALE_T)
        self.be._mark_dropped_echoes(SID, [])
        q = self._reg_queue()
        self.assertEqual(len(q), 1, "one notice, no re-delivery: %r" % (q,))
        self.assertIn("2 queued messages from before the restart were dropped as stale", q[0])
        self.assertIn(time.strftime("%Y-%m-%d %H:%M", time.localtime(STALE_T - 3600)), q[0])

    def test_a_stale_send_is_not_scanned(self):
        # the scan is the expensive step (a mark-less echo streams the whole transcript); the line runs first
        self._echo("three day old words", STALE_T)
        calls = []
        self.be._text_landed = lambda *a, **k: calls.append(a) or False
        with mock.patch.object(sb, "_input_landed_after", side_effect=AssertionError("scanned a stale send")):
            self.be._mark_dropped_echoes(SID, [])
        self.assertEqual(calls, [], "no transcript scan for a send past the line")

    def test_the_line_is_measured_from_the_send_stamp_not_the_text(self):
        # a send just inside the line is re-fed; one just outside is dropped — the stamp decides
        inside = self._echo("inside the line", NOW - int(sb.REDELIVER_MAX_AGE_S) + 120)
        outside = self._echo("outside the line", NOW - int(sb.REDELIVER_MAX_AGE_S) - 120)
        self.be._mark_dropped_echoes(SID, [])
        self.assertIn("inside the line", self._reg_queue())
        self.assertNotIn("outside the line", self._reg_queue())
        self.assertTrue(outside.get("stale"))
        self.assertNotIn("dropped", inside)

    def test_the_line_can_be_switched_off(self):
        sb.REDELIVER_MAX_AGE_S = 0
        self._echo("three day old words", STALE_T)
        self.be._mark_dropped_echoes(SID, [])
        self.assertEqual(self._reg_queue(), ["three day old words"], "0 restores the unbounded re-feed")

    def test_the_line_never_touches_the_queue_proper(self):
        # a send still in reg['queue'] (never fed) is the person's words waiting their turn: it stays, whatever its age
        self._echo("waiting its turn", STALE_T)
        sb.write_reg(self.be.state_dir, SID, {"sid": SID, "alive": True, "cwd": self.cwd,
                                              "lastSid": SID, "queue": ["waiting its turn"]})
        self.be._mark_dropped_echoes(SID, ["waiting its turn"])
        self.assertEqual(self._reg_queue(), ["waiting its turn"])
        self.assertFalse(any(a.get("dropped") for a in self.be._live[SID].values()))

    def test_a_stale_romp_authored_echo_makes_no_notice(self):
        self._echo("an old nudge body", STALE_T, author="romp")
        self.be._mark_dropped_echoes(SID, [])
        self.assertEqual(self._reg_queue(), [], "a nudge was never re-fed; nothing to announce")
        self.assertTrue(all(a.get("dropped") for a in self.be._live[SID].values()))

    def test_a_live_session_gets_the_notice_through_its_own_queue(self):
        class S:
            def __init__(self):
                self.q = []
            def pending(self):
                return list(self.q)
            def enqueue(self, text, qid=None, qts=None):
                self.q.append(text)
        s = S()
        self.be.sessions = {SID: s}
        self.be._lock = __import__("threading").RLock()
        self._echo("fresh typed words", FRESH_T)
        self._echo("three day old words", STALE_T)
        self.be._mark_dropped_echoes(SID, [])
        self.assertEqual(s.q[0], "fresh typed words")
        self.assertEqual(len(s.q), 2)
        self.assertIn("dropped as stale", s.q[1])
        self.assertEqual(self._reg_queue(), [], "the live session's queue is authoritative; the reg is not written")


class TheRefusedFlag(Fixture):
    def test_a_refused_send_is_flagged_and_never_refed(self):
        a = self._echo("a prompt the gate refused", FRESH_T)
        self.assertEqual(self.be.mark_echo_refused(SID, "a prompt the gate refused", "replayed schedule slot"), 1)
        self.assertTrue(a.get("dropped") and a.get("refused"))
        self.assertEqual(self.be._persisted, [SID], "the flags ride the mirror at once")
        self.assertEqual(self.be._forgotten, [a["uuid"]], "its landing will never come")
        self.be._mark_dropped_echoes(SID, [])
        self.assertEqual(self._reg_queue(), [], "a refused send is not a lost send: nothing re-fed, no notice")

    def test_the_mark_matches_under_the_echo_key_and_skips_landed_and_other_texts(self):
        a = self._echo("  a prompt the gate refused \n", FRESH_T)
        b = self._echo("another message entirely", FRESH_T)
        c = self._echo("a prompt the gate refused", FRESH_T - 5)
        c["_landed"] = True
        self.assertEqual(self.be.mark_echo_refused(SID, "a prompt the gate refused"), 1)
        self.assertTrue(a.get("refused"))
        self.assertNotIn("refused", b)
        self.assertNotIn("refused", c, "a landed echo is not refused: its record exists")
        self.assertEqual(self.be.mark_echo_refused(SID, ""), 0)
        self.assertEqual(self.be.mark_echo_refused(SID, "nothing wears this"), 0)

    def _replay(self, logs=None):
        """The real gate over a real backend (tests/test_cron_replay_dedupe.py's shape): the first fire records the
        slot, and a FRESH session's second fire of the same prompt is the restart replay. An echo wearing the prompt
        is stashed live before that fire. Returns (backend, the fresh session, the prompt, the state dir)."""
        import asyncio
        from pathlib import Path
        d = tempfile.mkdtemp()
        be = sb.SdkBackend(d, "/bin/true", lambda *a, **k: None, log=(logs.append if logs is not None else
                                                                      lambda *a, **k: None))
        cron, prompt = "* * * * *", "ping me every minute"
        sb.write_reg(Path(d), SID, {
            "sid": SID, "name": "web", "cwd": "/tmp", "alive": True,
            "sessionCrons": [{"id": "c1", "cron": cron, "prompt": prompt, "kind": "cron", "recurring": True,
                              "armedAt": time.time() - 3600, "dueEpoch": None, "procGen": "gen-A"}]})
        first = sb.SdkSession(be, sb.read_reg(Path(d), SID))
        self.assertEqual(asyncio.run(first._prompt_submit_hook({"prompt": prompt}, None, None)), {},
                         "the first fire of the slot runs and is recorded")
        be._stash_live(SID, "echo:refused1", {"type": "user", "uuid": "echo:refused1", "session_id": SID,
                                              "t": FRESH_T, "author": "human", "_echo_text": prompt,
                                              "message": {"role": "user", "content": [{"type": "text", "text": prompt}]}})
        return be, sb.SdkSession(be, sb.read_reg(Path(d), SID)), prompt, Path(d)

    @staticmethod
    def _mark_waits(be, stall_s=0.0):
        """Wrap the backend's mark so a test can WAIT for it (and stall it): the gate issues the mark beside its
        verdict, on a thread of its own, so the verdict comes back before the flag is on. Returns the Event the
        wrapped mark sets when it has run."""
        real, done = be.mark_echo_refused, threading.Event()

        def wrapped(sid, text, reason=""):
            try:
                if stall_s:
                    time.sleep(stall_s)
                return real(sid, text, reason)
            finally:
                done.set()
        be.mark_echo_refused = wrapped
        return done

    def _timed_fire(self, sess, prompt):
        import asyncio

        async def run():
            t0 = time.monotonic()
            out = await sess._prompt_submit_hook({"prompt": prompt}, None, None)
            return out, time.monotonic() - t0
        return asyncio.run(run())

    def test_the_gates_block_marks_the_echo(self):
        # The block flags the echo wearing that prompt refused, so the next boot's re-delivery never treats it as a
        # lost send; the flag rides the mirror at once.
        be, again, prompt, d = self._replay()
        done = self._mark_waits(be)
        out, _ = self._timed_fire(again, prompt)
        self.assertEqual(out.get("decision"), "block", "the replayed slot is refused")
        self.assertTrue(done.wait(5), "the mark ran")
        a = be._live[SID]["echo:refused1"]
        self.assertTrue(a.get("dropped") and a.get("refused"), "…and the echo wearing it is flagged refused")
        mirror = (sb.read_reg(d, SID) or {}).get("echoes") or []
        self.assertTrue(any(e.get("refused") for e in mirror), "the flag is on the mirror already")

    def test_a_stalled_mark_never_turns_the_block_into_an_allow(self):
        """The mark is a reg write. Awaited INSIDE _prompt_submit_hook's cap (asyncio.wait_for, {} = allowed on a
        timeout), a stalled write ran the cap out AFTER the verdict was decided: the block came back as an allow and
        the replayed schedule fired anyway (the pull request review, 2026-09-14). So the verdict is returned first
        and the mark runs beside it, outside the cap: with the mark stalled far past a 50 ms cap, the block still
        arrives inside the cap, the cap never trips, and the flag lands once the mark is through."""
        logs = []
        be, again, prompt, d = self._replay(logs)
        done = self._mark_waits(be, stall_s=0.6)
        before = os.environ.get("ROMP_PROMPT_HOOK_TIMEOUT_S")
        os.environ["ROMP_PROMPT_HOOK_TIMEOUT_S"] = "0.05"
        self.addCleanup(lambda: os.environ.pop("ROMP_PROMPT_HOOK_TIMEOUT_S", None) if before is None
                        else os.environ.__setitem__("ROMP_PROMPT_HOOK_TIMEOUT_S", before))
        out, took = self._timed_fire(again, prompt)
        self.assertEqual(out.get("decision"), "block", "the verdict does not wait on the mark")
        self.assertLess(took, 0.5, "…and arrives inside the cap, not after the stalled write")
        self.assertFalse(any("ran past its" in str(m) for m in logs), "the cap never tripped: nothing was under it")
        self.assertTrue(done.wait(5), "the mark still ran, outside the cap")
        a = be._live[SID]["echo:refused1"]
        self.assertTrue(a.get("dropped") and a.get("refused"), "…and flagged the echo once it was through")
        self.assertTrue(any(e.get("refused") for e in ((sb.read_reg(d, SID) or {}).get("echoes") or [])))


class TheMirror(Fixture):
    def test_stale_and_refused_ride_the_mirror_both_ways(self):
        writes = []
        self.be._update_reg = lambda sid, **kw: writes.append(kw)
        self.be._persist_echoes = sb.SdkBackend._persist_echoes.__get__(self.be)
        a = self._echo("three day old words", STALE_T)
        a["dropped"] = a["stale"] = True
        b = self._echo("a prompt the gate refused", FRESH_T)
        b["dropped"] = b["refused"] = True
        self.be._persist_echoes(SID)
        snap = {e["text"]: e for e in writes[-1]["echoes"]}
        self.assertTrue(snap["three day old words"].get("stale") and snap["three day old words"].get("dropped"))
        self.assertTrue(snap["a prompt the gate refused"].get("refused"))
        self.assertNotIn("refused", snap["three day old words"])
        stashed = []
        self.be._stash_live = lambda sid, key, atom: stashed.append(atom)
        self.be._reseed_echoes = sb.SdkBackend._reseed_echoes.__get__(self.be)
        self.be._live.clear()                         # nothing live → the reseed stashes and stops
        self.be._reseed_echoes([{"sid": SID, "alive": True, "echoes": writes[-1]["echoes"], "queue": []}])
        by = {x["_echo_text"]: x for x in stashed}
        self.assertTrue(by["three day old words"].get("dropped") and by["three day old words"].get("stale"))
        self.assertTrue(by["a prompt the gate refused"].get("dropped") and by["a prompt the gate refused"].get("refused"))


if __name__ == "__main__":
    unittest.main()
