#!/usr/bin/env python3
"""A text fed into a running turn is TAKEN when the CLI records its queue removal as absorbed_mid_turn, even with no
queued_command attachment; a removal for any other reason releases the hold but is never a delivery.

What happened (2026-10-09): romp feeds a running session one queued text at a time and holds the rest until it sees
the CLI take the current one (SdkSession._untaken, released by _untaken_taken). For a text fed mid-turn the take was
read only from the queued_command attachment the CLI writes when it splices the text into the turn. On one CLI
version some mid-turn takes left only the CLI's queue bookkeeping record, a queue-operation remove with reason
absorbed_mid_turn, and no attachment, so the hold never saw the take: busy sessions whose turns never end kept every
later text queued for over an hour.

The fix reads the remove too (sdk_backend._queue_removal, through _text_take):
  * an absorbed_mid_turn removal naming the fed text is a take: the hold releases, the next text feeds into the same
    turn, the echoes it speaks for land by id (no record carries their text for the by-text retire), and it is
    counted (feed_take_counts);
  * a joined text (peer mail and tagged sends fed as one) matches its removal the same way, and every part's echo
    lands;
  * a removal for another reason (dropped_by_hook: a hook dropped the text) releases the hold, since the text left the
    CLI's queue, but the text is reported as removed unread: its echo is flagged never delivered (dropped and refused,
    like the prompt gate's refusal), it is not re-fed, its mail gets no read stamp, and one problem line names the
    session;
  * a remove with no reason (an older CLI's discard) or for another text releases nothing;
  * the queued_command attachment still releases the hold as before, and the boot re-delivery guard (_text_landed)
    reads an absorbed removal as a landing, so a restart cannot re-feed a text the turn already read.
The REAL _amain runs (its inputs() closure and _on_message) against the stand-in SDK client of
test_queued_sends_not_fused. SYNTHETIC fixtures only: invented text, placeholder uuids, the TESTHOST hostname."""
import os
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from romp_load import load_source          # noqa: E402
os.environ["XDG_STATE_HOME"] = tempfile.mkdtemp()
os.environ.pop("ROMP_STATE_DIR", None)
import test_queued_sends_not_fused as _q   # noqa: E402  (borrowed harness: the stand-in SDK client and helpers)

sb = _q.sb
pm = load_source("romp_postal_absorbedtake", os.path.join(_q.BIN, "romp-postal-service"))
SID, FSID = _q.SID, _q.FSID
PEER = "22222222-3333-4444-5555-777777777777"
SEP = "\n\n"


def _mid(n):
    return "17100%05d.%06d.TESTHOST" % (n, n)


def _banner(mid, body="the index rebuild finished; your turn"):
    return pm.format_push([{"from": "api", "from_id": PEER, "body": body + " (" + mid[:10] + ")", "id": mid, "date": ""}])


def _tagged(body, label="watch"):
    return body + "\n\n<!-- romp-tag: " + label + " -->"


def _enqueue_op(text):
    """The CLI's queue bookkeeping when it receives a fed text: no uuid, the text as `content`."""
    return {"type": "queue-operation", "operation": "enqueue", "timestamp": _q._iso(time.time()),
            "sessionId": FSID, "content": text}


def _remove_op(text, reason="absorbed_mid_turn"):
    """The CLI's queue bookkeeping when the text leaves its queue. reason None writes the older, reason-less shape."""
    rec = {"type": "queue-operation", "operation": "remove", "timestamp": _q._iso(time.time()), "sessionId": FSID}
    if text is not None:
        rec["content"] = text
    if reason is not None:
        rec["reason"] = reason
    return rec


def _splice(text):
    return {"type": "attachment", "uuid": "11111111-2222-3333-4444-" + ("%012d" % (abs(hash(text)) % 10 ** 12)),
            "timestamp": _q._iso(time.time()), "attachment": {"type": "queued_command", "prompt": text}}


class TakeOnTheCLIsQueueRemoval(unittest.TestCase):
    """End to end through the real inputs() closure and _on_message, with real sends (echo and id)."""
    _H = _q.OneFedTextAtATime   # the borrowed harness, bound in the class body so it is not collected a second time here
    _Client, _Options = _H._Client, _H._Options
    setUp, tearDown = _H.setUp, _H.tearDown
    _wait, _settle, _push, _uid = _H._wait, _H._settle, _H._push, _H._uid
    _init, _assistant, _result_frame = _H._init, _H._assistant, _H._result_frame
    _user_record, _first_turn, _transcript, _append = _H._user_record, _H._first_turn, _H._transcript, _H._append

    def _send(self, text, user=False):
        """A real send (SdkBackend.send): the queued copy and its echo share one minted id, which this returns."""
        self.assertTrue(self.be.send(SID, text, user=user))
        hit = [k for k, a in (self.be._live.get(SID) or {}).items() if a.get("_echo_text") == text]
        self.assertTrue(hit, "the send's echo, stashed under its id")
        return hit[-1]

    def _echo(self, qid):
        return (self.be._live.get(SID) or {}).get(qid)

    def _reports(self):
        reports = []
        self.be.postal_taken = lambda sid, mids: reports.append((sid, list(mids)))
        return reports

    def _counts(self):
        return self.be.feed_take_counts()

    def test_an_absorbed_removal_with_no_attachment_releases_the_hold_and_the_next_text_feeds(self):
        """The incident: the CLI took the fed text into its running turn and wrote only the remove. The hold
        releases on the next turn frame and the text behind it follows into the SAME turn."""
        s, c = self.s, self._first_turn()
        q1 = self._send("please also rerun the lint step", user=True)
        self._wait(lambda: len(c.writes) == 2, "the first mid-turn send forwarded")
        self._append(_enqueue_op("please also rerun the lint step"))
        q2 = self._send("and then summarise what changed", user=True)
        self._settle()
        self.assertEqual(len(c.writes), 2, "the second text is held while the first is untaken")
        self._assistant(c)                                   # a turn frame with only the enqueue on disk: no take
        self._settle()
        self.assertEqual(len(c.writes), 2, "an enqueue is not a take")
        self._append(_remove_op("please also rerun the lint step"))
        self._assistant(c)
        self._wait(lambda: len(c.writes) == 3, "the next text fed after the absorbed removal")
        self.assertEqual(c.writes[2], ("and then summarise what changed", "turn-1"), "into the same running turn")
        self.assertEqual(self._counts().get("absorbed_removal"), 1, "the take is counted")
        self._wait(lambda: (self._echo(q1) or {}).get("_landed"), "the taken text's echo landed by id")
        self.assertFalse((self._echo(q2) or {}).get("_landed"), "the text fed behind it is still owed its own take")
        self.assertFalse((self._echo(q1) or {}).get("dropped"), "a take is never a loss")
        self.be.prune_live(SID, set(), {})
        self.assertIsNone(self._echo(q1), "the landed echo retires with no record of its text (prune_live's _landed exit)")
        self.assertIsNotNone(self._echo(q2))

    def test_joined_mail_and_tagged_sends_match_their_removal(self):
        """Mail and tagged sends queued back to back go in as one joined text; the CLI's removal names that joined
        text, so it releases the hold, every part's echo lands, and every mail id is reported read."""
        s, c = self.s, self._first_turn()
        reports = self._reports()
        s.enqueue("hold this")
        self._wait(lambda: len(c.writes) == 2, "the holding send forwarded")
        b1, b2 = _banner(_mid(1)), _banner(_mid(2), body="the cache is warm")
        t1, t2 = _tagged("build 7 finished green on TESTHOST"), _tagged("deploy queued", label="ci")
        s.enqueue_postal(b1, [_mid(1)])
        qa = self._send(t1)
        s.enqueue_postal(b2, [_mid(2)])
        qb = self._send(t2)
        s.enqueue("the person's own words")                 # not joinable: waits behind the joined text
        self._append(_splice("hold this"))
        self._assistant(c)
        joined = SEP.join([b1, t1, b2, t2])
        self._wait(lambda: len(c.writes) == 3, "the joined text fed")
        self.assertEqual(c.writes[2][0], joined)
        self._append(_enqueue_op(joined))
        self._append(_remove_op(joined))
        self._assistant(c)
        self._wait(lambda: len(c.writes) == 4, "the text behind the joined one fed after its absorbed removal")
        self.assertEqual(c.writes[3], ("the person's own words", "turn-1"))
        self._wait(lambda: len(reports) == 1, "the joined text's mail reported read")
        self.assertEqual(reports, [(SID, [_mid(1), _mid(2)])], "every mail id, once")
        for q in (qa, qb):
            self.assertTrue((self._echo(q) or {}).get("_landed"), "each tagged part's echo landed by id")
        self.assertEqual(self._counts().get("absorbed_removal"), 1, "one removal take: the holding send's attachment take is not one")

    def test_a_dropped_by_hook_removal_releases_the_hold_but_is_reported_never_delivered(self):
        """A hook dropped the fed text: it left the CLI's queue (so the next text may feed) but the turn never read
        it. Not a take: no landing, no read stamp, not re-fed; its echo is flagged never delivered and one problem
        line names the session and the reason."""
        s, c = self.s, self._first_turn()
        reports = self._reports()
        b = _banner(_mid(3))
        s.enqueue_postal(b, [_mid(3)])
        self._wait(lambda: len(c.writes) == 2, "the mail forwarded")
        q = self._send("a note the hook will drop", user=True)
        self._wait(lambda: len(c.writes) == 2 and s.pending() == ["a note the hook will drop"], "the note waits")
        self._append(_enqueue_op(b))
        self._append(_remove_op(b, reason="dropped_by_hook"))
        self._assistant(c)
        self._wait(lambda: len(c.writes) == 3, "the note fed once the dropped mail left the CLI's queue")
        self._settle()
        self.assertEqual(reports, [], "a dropped banner's mail is never reported read")
        self.assertNotIn(b, s.pending(), "a hook's drop is not re-headed (the same hook would drop it again)")
        self.assertEqual(self._counts().get("removed:dropped_by_hook"), 1)
        self.assertIsNone(self._counts().get("absorbed_removal"), "a drop is never counted as a take")
        lines = [p["text"] for p in self.be._problems if "removed a fed text" in p.get("text", "")]
        self.assertEqual(len(lines), 1, "one problem line")
        self.assertIn("web", lines[0], "naming the session")
        self.assertIn("dropped_by_hook", lines[0], "and the reason")
        self.assertIn(_mid(3), lines[0], "and the mail that stays unread")

        # the note the person sent is dropped too: its echo is flagged never delivered, and never re-fed
        self._append(_enqueue_op("a note the hook will drop"))
        self._append(_remove_op("a note the hook will drop", reason="dropped_by_hook"))
        self._assistant(c)
        self._wait(lambda: (self._echo(q) or {}).get("dropped"), "the dropped note's echo flagged")
        e = self._echo(q)
        self.assertTrue(e.get("refused"), "flagged refused like a prompt-gate refusal: dismissable, never re-delivered")
        self.assertIn("dropped_by_hook", e.get("refusedWhy", ""))
        self.assertFalse(e.get("_landed"), "a drop is never a landing")
        self.assertIsNone(s._untaken, "the hold released")
        self.assertEqual(s.pending(), [], "nothing re-headed")
        self.assertEqual(self._counts().get("removed:dropped_by_hook"), 2)
        self.be.prune_live(SID, set(), {})
        self.assertIsNotNone(self._echo(q), "a dropped echo stays visible until the person dismisses it")

    def test_a_reasonless_or_foreign_removal_releases_nothing(self):
        """A remove with no reason (an older CLI's discard), one with no text, and one naming another text say nothing
        about the fed text: the hold stays until its own take."""
        s, c = self.s, self._first_turn()
        s.enqueue("first note")
        self._wait(lambda: len(c.writes) == 2, "the first note forwarded")
        s.enqueue("second note")
        self._append(_remove_op("first note", reason=None))
        self._append(_remove_op(None))
        self._append(_remove_op("some other text"))
        self._append(_remove_op("some other text", reason="dropped_by_hook"))
        self._assistant(c)
        self._settle()
        self.assertEqual(len(c.writes), 2, "none of these is the first note's take")
        self.assertEqual(self._counts(), {})
        self._append(_remove_op("first note"))
        self._assistant(c)
        self._wait(lambda: len(c.writes) == 3, "its own absorbed removal is")

    def test_the_queued_command_attachment_still_releases_the_hold_and_counts_no_removal(self):
        s, c = self.s, self._first_turn()
        q = self._send("check the release notes", user=True)
        self._wait(lambda: len(c.writes) == 2, "forwarded")
        s.enqueue("then tag the release")
        self._append(_splice("check the release notes"))
        self._assistant(c)
        self._wait(lambda: len(c.writes) == 3, "the attachment is the take, as before")
        self.assertEqual(self._counts(), {}, "no removal take counted")
        self.assertFalse((self._echo(q) or {}).get("_landed"), "its echo retires by text on the record, as before")
        # the removal the CLI writes after the attachment changes nothing for the text now held
        self._append(_remove_op("check the release notes"))
        self._assistant(c)
        self._settle()
        self.assertEqual(len(c.writes), 3)
        self.assertEqual(self._counts(), {})

    def test_the_cli_exiting_after_a_dropped_removal_does_not_re_head_the_text(self):
        """The CLI removed the text unread and exited before any turn frame acted on it: the exit release reads the
        removal and takes the same road (flagged, not re-fed), where a text with no record at all is re-headed."""
        s, c = self.s, self._first_turn()
        q = self._send("a note the hook will drop", user=True)
        self._wait(lambda: len(c.writes) == 2, "forwarded")
        self._append(_enqueue_op("a note the hook will drop"))
        self._append(_remove_op("a note the hook will drop", reason="dropped_by_hook"))
        u = s._untaken
        s._release_hold_at_exit()
        self.assertIsNone(s._untaken)
        self.assertEqual(s.pending(), [], "not re-headed")
        self.assertTrue((self._echo(q) or {}).get("dropped"), "flagged never delivered")
        self.assertEqual(self._counts().get("removed:dropped_by_hook"), 1)
        self.assertIsNotNone(u)


class TheLandingScanReadsRemovals(unittest.TestCase):
    """_text_landed / _text_take straight against a synthetic transcript: the boot re-delivery guard counts an absorbed
    removal as a landing (a restart must not re-feed a text the turn read), and a drop as not landed."""
    _H = _q.OneFedTextAtATime
    _Client, _Options = _H._Client, _H._Options
    setUp, tearDown = _H.setUp, _H.tearDown

    def _write(self, recs):
        sb.write_reg(self.state, SID, {**sb.read_reg(self.state, SID), "lastSid": FSID})
        p = sb.transcript_path(self.cwd, FSID)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        import json
        with open(p, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    def test_verdicts(self):
        self._write([_enqueue_op("alpha"), _remove_op("alpha"),
                     _enqueue_op("beta"), _remove_op("beta", reason="dropped_by_hook"),
                     _enqueue_op("gamma"), _remove_op("gamma", reason=None),
                     _splice("delta")])
        self.assertEqual(self.be._text_take(SID, "alpha", 0, 0, FSID), "absorbed")
        self.assertEqual(self.be._text_take(SID, "beta", 0, 0, FSID), ("removed", "dropped_by_hook"))
        self.assertIs(self.be._text_take(SID, "gamma", 0, 0, FSID), False, "a reason-less remove says nothing")
        self.assertEqual(self.be._text_take(SID, "delta", 0, 0, FSID), "record")
        self.assertIs(self.be._text_landed(SID, "alpha", 0, 0, FSID), True, "absorbed: landed, never re-fed at boot")
        self.assertIs(self.be._text_landed(SID, "beta", 0, 0, FSID), False, "dropped: not landed")
        cur = {}
        self.assertIs(self.be._text_landed(SID, "beta", 0, 0, FSID, cursor=cur), False)
        self.assertEqual(cur.get("take"), ("removed", "dropped_by_hook"), "the verdict rides back in the cursor")
        future = int(time.time()) + 3600
        self.assertIs(self.be._text_take(SID, "alpha", future, 0, FSID), False,
                      "a removal stamped before the send is an older copy's, like any record")

    def test_a_content_block_list_removal_matches_too(self):
        rec = _remove_op(None)
        rec["content"] = [{"type": "text", "text": "epsilon"}]
        self._write([rec])
        self.assertEqual(self.be._text_take(SID, "epsilon", 0, 0, FSID), "absorbed")


if __name__ == "__main__":
    unittest.main()
