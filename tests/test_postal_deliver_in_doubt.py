#!/usr/bin/env python3
"""A /deliver the kernel took but answered too late must not deliver the same message again (2026-09-29).

The bus claims a recipient's mail (new/ -> cur/, an exec row), POSTs the banner to the kernel's /deliver and waits
DELIVER_TIMEOUT (12 s) for the answer. The kernel does not stop when the bus stops waiting: it finishes and queues
the banner. Before this fix the bus read the missing answer as "kernel unreachable" and put every message back
(restore: cur/ -> new/, an unexec row, the pending marker raised), so the same message came in again three ways:
the next Stop-hook drain claimed it and fed it, the retry pass claimed it and posted it again, and each new claim
took the whole box, so the banner grew and the next post was slower still. Under a slow kernel one message was
delivered thirty-odd times in eighteen minutes, as exec/unexec pairs about every half minute.

The fix has two halves, and these tests pin both:
- the bus tells "the request never left" (a refused connect: put the mail back, as before) from "the request was
  sent and no answer came" (a read timeout or reset: the outcome is unknown). An unknown outcome leaves the mail
  claimed in cur/ with no unexec row, records the chunk as IN DOUBT, claims nothing more for that recipient, and
  the retry pass re-posts the SAME chunk under the same message ids until the kernel answers for it;
- the kernel's /deliver is safe to repeat: SdkBackend.deliver answers True without queueing a second copy of a
  banner whose message ids the session has all taken already (a bounded list, persisted in the registry so it
  outlives a kernel restart), and forgets the ids it hands back to the bus when a teardown strands a banner.

SYNTHETIC fixtures only: invented text, placeholder uuids, TESTHOST; no real session names or message ids.
"""
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from romp_load import load_source

HERE = os.path.dirname(os.path.realpath(__file__))
BIN = os.path.join(os.path.dirname(HERE), "bin")
# Hermetic state BEFORE the loads: both modules resolve their state root at import time.
os.environ["XDG_STATE_HOME"] = tempfile.mkdtemp()
os.environ.pop("ROMP_STATE_DIR", None)  # a live kernel's export outranks the XDG floor
sb = load_source("romp_sdk_backend_in_doubt", os.path.join(BIN, "romp_sdk_backend.py"))
pm = load_source("romp_postal_in_doubt", os.path.join(BIN, "romp-postal-service"))

TO = "66666666-7777-8888-9999-aaaaaaaaaaaa"       # the recipient session (the kernel-side tests)
_N = [0]


def _fresh_sid(prefix="66666666-7777-8888-9999-"):
    """A placeholder recipient id of its own per test: the maildir and ledger are module-wide, so a shared id would
    carry one test's claims into the next."""
    _N[0] += 1
    return prefix + "%012d" % _N[0]


SENDER = "11111111-2222-3333-4444-555555555555"
MID1 = "1700000000.111111.TESTHOST"
MID2 = "1700000001.222222.TESTHOST"

CLIENT_WAIT = 1.0      # the bus's wait for an answer, shortened for the test (production: DELIVER_TIMEOUT, 12 s)
SLOW_ANSWER = 2.0      # how long the slow kernel takes to answer AFTER it has queued the banner


def _banner(*mids, body="the pod is up; please take the next fire"):
    """A bus banner exactly as _push hands it to /deliver: one message per id, the bus's own formatter."""
    return pm.format_push([{"from": "web", "from_id": SENDER, "body": body, "id": m, "date": ""} for m in mids])


def _timeline(mid):
    """The ledger events for one message id, in order."""
    p = pm.TLDIR / "messages.jsonl"
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        try:
            row = json.loads(ln)
        except ValueError:
            continue
        if row.get("id") == mid:
            out.append(row.get("ev"))
    return out


class _SdkWorld:
    """A registry + empty transcript for one live SDK session registered without its thread (no loop, no CLI): the
    harness the stranded-mail tests use. _ensure is stubbed to hand back this session, the way the running kernel's
    would for a live one."""

    def _make_backend(self, to=TO):
        self.to = to
        self.state = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.state, "sdk"))
        with open(os.path.join(self.state, "session-hosts"), "w") as f:
            f.write("off")                            # a self-minted state root pins per-session hosts off
        self._cfg = os.environ.get("CLAUDE_CONFIG_DIR")
        os.environ["CLAUDE_CONFIG_DIR"] = os.path.join(self.state, "claude")
        self.cwd = os.path.join(self.state, "proj")
        os.makedirs(self.cwd, exist_ok=True)
        tp = sb.transcript_path(self.cwd, self.to)
        os.makedirs(os.path.dirname(tp), exist_ok=True)
        open(tp, "w").close()
        self.klog = []
        self.be = sb.SdkBackend(self.state, "/bin/true", lambda *a, **k: None, log=self.klog.append)
        self.sess = self._new_session()
        self.be._ensure = lambda sid, on_boot_settled=None: self.be.sessions.get(sid)

    def _new_session(self):
        """A fresh SdkSession seeded from the registry on disk, as a kernel restart seeds it."""
        reg = sb.read_reg(self.be.state_dir, self.to) or {"sid": self.to, "name": "api", "mode": "acceptEdits",
                                                           "alive": True, "cwd": self.cwd, "lastSid": self.to}
        sb.write_reg(self.be.state_dir, self.to, reg)
        s = sb.SdkSession(self.be, dict(reg))
        self.be.sessions[self.to] = s
        return s

    def _drop_backend(self):
        if self._cfg is None:
            os.environ.pop("CLAUDE_CONFIG_DIR", None)
        else:
            os.environ["CLAUDE_CONFIG_DIR"] = self._cfg

    def copies(self, mid):
        """How many banners queued in the session carry `mid`."""
        return sum(1 for t in self.sess.pending() if ("<!-- romp-msg-id: %s -->" % mid) in t)


class _SlowKernel(BaseHTTPRequestHandler):
    """The kernel's /deliver in shape: read the body, hand the banner to the backend's deliver (which queues it),
    and only THEN answer, after `delays` seconds (one per post, popped in order; none left means at once). A client
    that stopped waiting has closed its end by then; the write fails and is ignored, as the real handler's would."""
    protocol_version = "HTTP/1.1"
    backend = None
    delays = []
    posts = []
    done = None

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            injected = bool(_SlowKernel.backend.deliver(body["id"], body["text"]))
            _SlowKernel.posts.append(sb.postal_mids(body["text"]))
            time.sleep(_SlowKernel.delays.pop(0) if _SlowKernel.delays else 0)
            out = json.dumps({"ok": True, "injected": injected}).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            _SlowKernel.done.release()

    def log_message(self, *a):
        pass


class SlowKernelDoesNotRedeliver(_SdkWorld, unittest.TestCase):
    """The loop, end to end: the real bus (_push, _drain, _retry_pending, the real _kernel_post over a real socket, a
    real maildir and ledger) against a kernel that queues each banner on a real SdkBackend and answers too late."""

    def setUp(self):
        self._make_backend(_fresh_sid())
        self._seam = os.environ.pop("ROMP_SESSIONS_FILE", None)   # not a seam test: let _push actually post
        self.saved = (pm.KERNEL_BASE, pm._push_disabled, pm._log, pm.local_agents, urllib.request.urlopen)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), _SlowKernel)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        pm.KERNEL_BASE = "http://127.0.0.1:%d" % self.srv.server_address[1]
        pm._push_disabled = lambda: False
        self.logged = []
        pm._log = self.logged.append
        pm.local_agents = lambda threads=False: [{"id": self.to, "name": "api", "state": "idle"}]
        real = self.saved[4]
        # the bus's own wait, shortened: whatever timeout the caller passes, the socket gives up after CLIENT_WAIT
        urllib.request.urlopen = lambda req, timeout=None, **k: real(req, timeout=CLIENT_WAIT, **k)
        _SlowKernel.backend, _SlowKernel.posts, _SlowKernel.done = self.be, [], threading.Semaphore(0)
        _SlowKernel.delays = [SLOW_ANSWER]                        # the first post is answered late; later ones at once
        pm.STREAKS.pop(self.to, None)

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        if self._seam is not None:
            os.environ["ROMP_SESSIONS_FILE"] = self._seam
        pm.KERNEL_BASE, pm._push_disabled, pm._log, pm.local_agents, urllib.request.urlopen = self.saved
        pm.STREAKS.pop(self.to, None)
        self._drop_backend()

    def _kernel_finished(self, n=1):
        for _ in range(n):
            self.assertTrue(_SlowKernel.done.acquire(timeout=10), "the fake kernel never finished a post")

    def test_a_late_answer_leaves_the_mail_claimed_not_back_in_new_for_a_second_feed(self):
        mid = pm.deliver(self.to, "web", SENDER, "the pod is up; please take the next fire", kind="coordinate")
        self.assertFalse(pm._push(self.to, {"id": self.to, "state": "idle"}), "no answer: not a landed push")
        self._kernel_finished()
        self.assertEqual(self.copies(mid), 1, "the kernel queued the banner although the bus stopped waiting")
        self.assertEqual(pm.read_box(self.to, consume=False), [],
                         "the mail stays claimed in cur/: before the fix it went back to new/, where the next "
                         "drain or retry claimed it again")
        self.assertEqual(pm._drain(self.to)["messages"], [],
                         "the session's Stop-hook drain does not feed it a second time")
        self.assertEqual(_timeline(mid), ["sent", "exec"], "no unexec row: the claim is not rolled back")
        self.assertTrue((pm.MAILROOT / self.to / "cur" / mid).is_file())
        self.assertTrue(any("no answer" in ln and self.to in ln for ln in self.logged),
                        "the log names the missing answer, never 'kernel unreachable': %r" % (self.logged,))

    def test_the_retry_pass_re_posts_the_same_ids_and_the_session_gets_one_copy(self):
        mid = pm.deliver(self.to, "web", SENDER, "the pod is up; please take the next fire", kind="coordinate")
        pm._push(self.to, {"id": self.to, "state": "idle"})
        self._kernel_finished()
        pm._retry_pending()                                   # the bus's 5 s pass
        self.assertEqual(self.copies(mid), 1,
                         "one message, one copy in the session: before the fix the retry claimed it again and the "
                         "kernel queued a second banner (and a third, and ... while the kernel stayed slow)")
        self.assertEqual(_timeline(mid), ["sent", "exec"], "one claim, never rolled back")
        self.assertEqual(_SlowKernel.posts[-1], [mid], "the re-post carries the same message id")
        pm._retry_pending()
        self.assertEqual(len(_SlowKernel.posts), 2, "answered for: nothing in doubt, nothing more is posted")
        self.assertEqual(pm._doubt_read(self.to), [], "the in-doubt record is retired once the kernel answers")
        self.assertEqual(pm.read_box(self.to, consume=False), [])

    def test_mail_that_arrives_while_a_chunk_is_in_doubt_waits_behind_it(self):
        first = pm.deliver(self.to, "web", SENDER, "first", kind="coordinate")
        pm._push(self.to, {"id": self.to, "state": "idle"})
        self._kernel_finished()
        second = pm.deliver(self.to, "web", SENDER, "second", kind="coordinate")
        _SlowKernel.delays = [SLOW_ANSWER]                    # the kernel is still slow: the re-post goes unanswered too
        self.assertFalse(pm._push(self.to, {"id": self.to, "state": "idle"}))
        self._kernel_finished()
        self.assertEqual(_SlowKernel.posts, [[first], [first]],
                         "only the in-doubt chunk is re-posted; the new mail is not claimed on top of it, so the "
                         "banner no longer grows with every retry")
        self.assertEqual([m["id"] for m in pm.read_box(self.to, consume=False)], [second])
        self.assertEqual(self.copies(first), 1, "the repeat was recognised by id and not queued")
        pm._retry_pending()                                   # the kernel answers now: the doubt clears, then the new mail
        self.assertEqual(_SlowKernel.posts[-2:], [[first], [second]])
        self.assertEqual((self.copies(first), self.copies(second)), (1, 1))
        self.assertEqual(pm.read_box(self.to, consume=False), [])


class InDoubtResolution(unittest.TestCase):
    """The retry side of an in-doubt chunk, with the kernel stubbed at _kernel_post: an answer settles it, a missing
    answer or a refused connect keeps holding it, and a message that left cur/ meanwhile is dropped from it."""

    def setUp(self):
        self.SID = _fresh_sid("33333333-4444-5555-6666-")
        self._seam = os.environ.pop("ROMP_SESSIONS_FILE", None)
        self.saved = (pm._kernel_post, pm._push_disabled, pm._log)
        pm._push_disabled = lambda: False
        self.logged, self.posted = [], []
        pm._log = self.logged.append
        pm.STREAKS.pop(self.SID, None)
        self.mid = pm.deliver(self.SID, "web", SENDER, "an invented note", kind="coordinate")
        self._answer(pm.NO_ANSWER)
        pm._push(self.SID, {"id": self.SID, "state": "idle"})
        self.assertEqual(pm._doubt_read(self.SID), [[self.mid]], "the unanswered chunk is recorded in doubt")

    def tearDown(self):
        if self._seam is not None:
            os.environ["ROMP_SESSIONS_FILE"] = self._seam
        pm._kernel_post, pm._push_disabled, pm._log = self.saved
        pm.STREAKS.pop(self.SID, None)

    def _answer(self, resp):
        def post(path, body, timeout=2, no_answer=None):
            self.posted.append(body)
            return no_answer if resp is pm.NO_ANSWER else resp
        pm._kernel_post = post

    def test_no_answer_again_keeps_holding_it(self):
        self._answer(pm.NO_ANSWER)
        self.assertFalse(pm._push(self.SID, {"id": self.SID, "state": "idle"}))
        self.assertEqual(pm._doubt_read(self.SID), [[self.mid]])
        self.assertEqual(_timeline(self.mid), ["sent", "exec"])

    def test_a_refused_connect_on_the_re_post_keeps_holding_it(self):
        # the first post may have landed; a kernel that is down now (mid-restart) says nothing about it
        self._answer(None)
        self.assertFalse(pm._push(self.SID, {"id": self.SID, "state": "idle"}))
        self.assertEqual(pm._doubt_read(self.SID), [[self.mid]])
        self.assertEqual(pm.read_box(self.SID, consume=False), [], "never put back while the outcome is unknown")
        self.assertEqual(_timeline(self.mid), ["sent", "exec"])

    def test_an_answer_that_it_was_taken_retires_it(self):
        self._answer({"ok": True, "injected": True})
        self.assertTrue(pm._push(self.SID, {"id": self.SID, "state": "idle"}))
        self.assertEqual(pm._doubt_read(self.SID), [])
        self.assertTrue((pm.MAILROOT / self.SID / "cur" / self.mid).is_file(), "consumed: it stays in cur/")
        self.assertEqual(_timeline(self.mid), ["sent", "exec"])

    def test_an_answer_that_it_was_not_taken_puts_it_back_for_the_drain(self):
        self._answer({"ok": True, "injected": False})
        self.assertFalse(pm._push(self.SID, {"id": self.SID, "state": "idle"}))
        self.assertEqual(pm._doubt_read(self.SID), [])
        self.assertEqual([m["id"] for m in pm.read_box(self.SID, consume=False)], [self.mid],
                         "an ANSWERED not-taken is the old roll-back: back in new/ under its own id")
        self.assertEqual(_timeline(self.mid), ["sent", "exec", "unexec"])

    def test_a_message_restored_meanwhile_leaves_the_record(self):
        # the kernel handed it back (a teardown stranded the banner): restore() moves it to new/ and the doubt drops it
        self.assertEqual(pm.restore(self.SID, self.mid), pm.RESTORED)
        self.assertEqual(pm._doubt_read(self.SID), [])

    def test_a_record_that_cannot_be_written_falls_back_to_the_put_back_and_says_so(self):
        # nothing would ever re-post a chunk with no record, so it goes back to new/ (a possible second copy, which the
        # kernel's repeat check mostly absorbs, beats mail stranded in cur/ for good)
        mid = pm.deliver(self.SID, "web", SENDER, "another invented note", kind="coordinate")
        saved = pm._doubt_add
        pm._doubt_add = lambda sid, mids: False
        try:
            self._answer({"ok": True, "injected": True})      # the old chunk settles first
            self.assertTrue(pm._resolve_in_doubt(self.SID, {"id": self.SID})[0])
            self._answer(pm.NO_ANSWER)
            self.assertFalse(pm._push(self.SID, {"id": self.SID, "state": "idle"}))
        finally:
            pm._doubt_add = saved
        self.assertEqual([m["id"] for m in pm.read_box(self.SID, consume=False)], [mid])
        self.assertEqual(_timeline(mid), ["sent", "exec", "unexec"])
        self.assertTrue(any("could not be written" in ln for ln in self.logged), self.logged)

    def test_the_record_survives_a_bus_restart(self):
        # the record is a file: a fresh bus process (code-change re-exec) reads the same chunk
        self.assertTrue((pm.MAILDOUBT / self.SID).is_file())
        self.assertEqual((pm.MAILDOUBT / self.SID).read_text().split(), [self.mid])


class KernelPostTellsUnsentFromUnanswered(unittest.TestCase):
    """_kernel_post: a request that never left (urllib wraps a connect or send failure in URLError) is None, as
    before; a request sent in full and then not answered is the caller's `no_answer`, and None for every caller
    that does not ask (the working-note, the notice door, the redial keep their contract)."""

    def setUp(self):
        self._seam = os.environ.pop("ROMP_SESSIONS_FILE", None)
        self.saved = urllib.request.urlopen

    def tearDown(self):
        urllib.request.urlopen = self.saved
        if self._seam is not None:
            os.environ["ROMP_SESSIONS_FILE"] = self._seam

    def _raise(self, exc):
        def urlopen(req, timeout=None, **k):
            raise exc
        urllib.request.urlopen = urlopen

    def test_a_refused_connect_is_none_even_when_the_caller_asks(self):
        self._raise(urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")))
        self.assertIsNone(pm._kernel_post("/deliver", {"id": TO, "text": "x"}, timeout=12, no_answer=pm.NO_ANSWER))

    def test_a_read_timeout_is_no_answer_for_a_caller_that_asks_and_none_for_the_rest(self):
        self._raise(TimeoutError("timed out"))
        self.assertIs(pm._kernel_post("/deliver", {"id": TO, "text": "x"}, timeout=12, no_answer=pm.NO_ANSWER),
                      pm.NO_ANSWER)
        self.assertIsNone(pm._kernel_post("/working", {"id": TO, "text": ""}))

    def test_a_reset_after_the_send_is_no_answer(self):
        import http.client
        self._raise(http.client.RemoteDisconnected("Remote end closed connection without response"))
        self.assertIs(pm._kernel_post("/deliver", {"id": TO, "text": "x"}, no_answer=pm.NO_ANSWER), pm.NO_ANSWER)

    def test_no_answer_is_falsy_so_a_truth_test_never_reads_it_as_taken(self):
        self.assertFalse(pm.NO_ANSWER)


class KernelDeliverIsSafeToRepeat(_SdkWorld, unittest.TestCase):
    """SdkBackend.deliver: a banner whose message ids the session has all taken is answered True and not queued again."""

    def setUp(self):
        self._make_backend()

    def tearDown(self):
        self._drop_backend()

    def test_the_same_banner_twice_is_queued_once(self):
        b = _banner(MID1, MID2)
        self.assertTrue(self.be.deliver(TO, b))
        self.assertTrue(self.be.deliver(TO, b), "a repeat is answered taken, so the bus retires its claim")
        self.assertEqual(self.sess.pending(), [b], "one copy: before the fix every repeat queued another banner")
        self.assertTrue(any("taken already" in ln and MID1 in ln for ln in self.klog), self.klog)

    def test_a_repeat_is_recognised_across_a_kernel_restart(self):
        b = _banner(MID1)
        self.assertTrue(self.be.deliver(TO, b))
        self.sess = self._new_session()                       # a restarted kernel seeds the session from the registry
        self.assertEqual(self.sess.pending(), [b], "the queue mirror carried the banner")
        self.assertTrue(self.be.deliver(TO, b))
        self.assertEqual(self.sess.pending(), [b], "and the taken ids came with it")

    def test_racing_posts_of_one_banner_queue_it_once(self):
        b = _banner(MID1)
        start = threading.Barrier(8)

        def post():
            start.wait()
            self.be.deliver(TO, b)
        ts = [threading.Thread(target=post) for _ in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10)
        self.assertEqual(self.copies(MID1), 1)

    def test_a_banner_with_a_new_id_beside_a_taken_one_is_queued_whole_and_said(self):
        self.be.deliver(TO, _banner(MID1))
        both = _banner(MID1, MID2)
        self.assertTrue(self.be.deliver(TO, both))
        self.assertEqual(self.sess.pending(), [_banner(MID1), both],
                         "the banner cannot be split here, and a message delivered twice beats one never delivered")
        self.assertTrue(any("1 of 2" in ln for ln in self.klog), self.klog)

    def test_a_text_with_no_message_ids_is_queued_every_time(self):
        self.be.deliver(TO, "plain wake text")
        self.be.deliver(TO, "plain wake text")
        self.assertEqual(self.sess.pending(), ["plain wake text", "plain wake text"])

    def test_the_taken_list_is_bounded(self):
        for i in range(sb.POSTAL_TAKEN_KEEP + 20):
            self.sess.take_postal(["1700%06d.000000.TESTHOST" % i])
        reg = sb.read_reg(self.be.state_dir, TO) or {}
        self.assertEqual(len(reg.get("postalTaken") or []), sb.POSTAL_TAKEN_KEEP)
        self.assertEqual(reg["postalTaken"][-1], "1700%06d.000000.TESTHOST" % (sb.POSTAL_TAKEN_KEEP + 19),
                         "the newest ids are kept")

    def test_ids_handed_back_to_the_bus_are_forgotten_so_their_re_delivery_is_queued(self):
        b = _banner(MID1, MID2)
        self.be.deliver(TO, b)
        self.sess._pending.clear()                            # fed to the CLI...
        self.be.postal_restore = lambda sid, mids: set(mids)  # ...then stranded by a teardown and handed back
        self.sess.inflight = 1
        self.sess._inflight_texts.append(b)
        self.sess._reconcile_stranded()
        self.assertTrue(self.be.deliver(TO, b), "the bus re-delivers the mail it was handed back")
        self.assertEqual(self.sess.pending(), [b], "queued again: a forgotten id is not a repeat")


if __name__ == "__main__":
    unittest.main()
