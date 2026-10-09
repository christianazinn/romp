"""Tail-only record cache, the third review of round two (2026-10-09): four defects, each reproduced, and their pins.

1. A tail record stamped in the gap between the document's watermark (its last pre-cut conversational stamp) and the cut's
   first record: the cold parse sorts it into the last pre-cut turn, the restore gave it a turn of its own (after a
   compaction on the append path, and at a boot).
2. A reply or tool_result parented on the settled pre-cut tip: the cold parse extends the last pre-cut turn (and drops the
   tail as rewound); the restore opened a turn with no prompt.
   One rule answers both: a restore over a turns section refuses unless the tail's first atom, in the cold parse's sort
   order, opens a fresh turn after the frozen last pre-cut turn (a compaction boundary, or a prompt after a turn that ended)
   and sorts at or after every pre-cut atom.
3. A document written before preGates (every document on disk at the deploy): the restore skipped the pre-cut prompt-id
   and Skill-link checks, so a command wrapper wearing the pre-cut twin's prompt id, or a Skill tool_use on the pre-cut
   payload's id, restored a wrong tree, and the restore after a compaction kept it. It now checks against the document's
   whole-adapter gates (a superset of the pre-cut sets): one over-refusal at most, after which the rewrite carries preGates.
4. A typed slash command at the cut: its raw twin makes no atom, so the cut fell on the wrapper with the twin before it,
   and every restore refused the wrapper's prompt id: two whole parses at every restart, then the entry held whole. The
   cut now takes the twin with its wrapper.
Should-fix pins: a content refusal whose rewrite would publish the same cut declines and marks the leaf (later boots pay
one whole parse and no writes); the writer's cut search is linear; the converge pass's floor and owed-mark changes; the
rest-of-append scan's watermark for a restored entry (docMaxPpt).

Every defect test fails on d088e830b; the pins and the gap-prompt control pass there, and each pin fails with its change
reverted (the docMaxPpt line, the converge loop body of c8219856f). Synthetic only: invented text, placeholder uuids."""
import gzip
import json
import os
import sys
import time
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import test_record_cache_tail_r2 as R                   # noqa: E402  one kernel copy: R2Base, knobs, fresh
from test_asm_checkpoint_served import iso              # noqa: E402

T, em, km, SID, NOW, WINDOW = R.T, R.em, R.km, R.SID, R.NOW, R.WINDOW
WRAP = "<command-name>/review</command-name>\n<command-message>review</command-message>\n<command-args></command-args>"


def _ts(r):
    return em.parse_z(r["timestamp"])


class _Base(R._Roads):
    """Helpers over the 160-turn leaf: its document cuts at u158 (the last two turns are the tail), so the watermark is
    a157's stamp and the gap runs 40 s to u158's."""

    def _rec(self, u):
        return next(r for r in self.recs if r.get("uuid") == u)

    def _reply(self, u, parent, t):
        return {"type": "assistant", "uuid": u, "parentUuid": parent, "timestamp": iso(t), "cwd": "/w/notes-api",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "a reply " + u}], "stop_reason": "end_turn"}}

    def _tool_result(self, u, parent, t):
        return {"type": "user", "uuid": u, "parentUuid": parent, "timestamp": iso(t), "cwd": "/w/notes-api",
                "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_gone", "content": "ok"}]}}

    def _counted(self, fn):
        s0, w0 = dict(em._ASM_STATS), em._ASM_CKPT_STATS.get("written", 0)
        out = fn()
        d = {k: v - s0.get(k, 0) for k, v in em._ASM_STATS.items() if v != s0.get(k, 0)}
        if em._ASM_CKPT_STATS.get("written", 0) != w0:
            d["_written"] = em._ASM_CKPT_STATS.get("written", 0) - w0
        return out, d

    def _doc_path(self):
        docs = [p for p in (self.td / "checkpoints").iterdir() if p.name.endswith(".json.gz") or p.suffix == ".gz"]
        self.assertEqual(len(docs), 1, [p.name for p in (self.td / "checkpoints").iterdir()])
        return docs[0]

    def _strip_pre_gates(self):
        """The document as a writer before preGates left it: the same bytes but that one field (the deploy's first boot)."""
        p = self._doc_path()
        st = os.stat(p)
        doc = json.loads(gzip.decompress(p.read_bytes()).decode("utf-8"))
        self.assertIn("preGates", doc)
        doc.pop("preGates")
        p.write_bytes(gzip.compress(json.dumps(doc, separators=(",", ":")).encode("utf-8"), compresslevel=6))
        os.utime(p, (st.st_atime, st.st_mtime))


class AGapStampedRecord(_Base):
    """Defect 1: a reply stamped one second after the watermark, before the cut's first record."""

    def test_after_a_compaction_in_the_same_append(self):
        wm = _ts(self._rec("a157"))
        def mutate():
            b = self._boundary(1, self.t)
            self._write(b + [self._reply("gap1", b[-1]["uuid"], wm + 1)])
        d, got = self._go(mutate)
        self.assertEqual(got, self._ref(), d)

    def test_at_a_boot(self):
        wm = _ts(self._rec("a157"))
        d, got = self._go(lambda: self._write([self._reply("gap2", self.parent, wm + 1)]), boot=True)
        self.assertEqual(got, self._ref(), d)

    def test_control_a_prompt_in_the_gap_still_restores(self):
        """A prompt in the gap opens its own turn in the cold parse too: the restore answers it (no refusal)."""
        wm = _ts(self._rec("a157"))
        def mutate():
            self._write([{"type": "user", "uuid": "gapu", "parentUuid": self.parent, "timestamp": iso(wm + 1), "promptSource": "typed",
                          "cwd": "/w/notes-api", "message": {"role": "user", "content": "a prompt in the gap"}}])
        d, got = self._go(mutate, boot=True)
        self.assertEqual((d.get("restore"), d.get("full", 0)), (1, 0), d)
        self.assertEqual(got, self._ref(), d)


class ARecordOnTheSettledPreCutTip(_Base):
    """Defect 2: a record that opens no turn, parented on a157 (the pre-cut tip), stamped current."""

    def test_a_reply_after_an_append(self):
        d, got = self._go(lambda: self._write([self._reply("tip1", "a157", self.t)]))
        self.assertEqual(got, self._ref(), d)

    def test_a_reply_at_a_boot(self):
        d, got = self._go(lambda: self._write([self._reply("tip2", "a157", self.t)]), boot=True)
        self.assertEqual(got, self._ref(), d)

    def test_a_tool_result_at_a_boot(self):
        d, got = self._go(lambda: self._write([self._tool_result("tip3", "a157", self.t)]), boot=True)
        self.assertEqual(got, self._ref(), d)


class AnOldDocumentWithoutPreGates(_Base):
    """Defect 3: the document written before preGates, a pre-cut twin (u10, pid-pre) and payload (u12, toolu_pre_sk), and a
    tail record re-classifying one of them written after the document."""
    def setUp(self):
        _Base.setUp(self)
        for r in self.recs:
            if r.get("uuid") == "u10":
                r["promptId"] = "pid-pre"
                r["message"]["content"] = "/review"
            if r.get("uuid") == "u12":
                r["sourceToolUseID"] = "toolu_pre_sk"
                r["message"]["content"] = "instructions for the deploy skill"
        Path(self.path).write_text("".join(json.dumps(x) + "\n" for x in self.recs))

    _wrapper = R.APreCutTwinAndPayload._wrapper
    _skill = R.APreCutTwinAndPayload._skill
    WRAP = WRAP

    def _old(self, rec, compaction_after=False):
        with T.knobs(WINDOW, 4, roots=[str(self.proj)]):
            tree = self.parse()
            self.assertTrue(em.asm_checkpoint_write(self.path, SID, tree=tree), em.asm_checkpoint_stats())
            del tree
            self._strip_pre_gates()
            r_ = rec(self.parent)
            self._write([r_])
            self._reset()
            tree, d = self._counted(self.parse)
            em.hydrate(tree, SID)
            got = [T._strip_tree(tree)]
            if compaction_after:                          # the review: the restore after the compaction kept the wrong tree
                self._write(R.tail_turn(50, r_["uuid"], self.t + 100, boundary=True))
                tree = self.parse()
                em.hydrate(tree, SID)
                got.append(T._strip_tree(tree))
        return d, got

    def test_a_wrapper_at_the_first_boot(self):
        d, got = self._old(self._wrapper)
        self.assertEqual(got[0], self._ref(), d)

    def test_a_skill_link_at_the_first_boot(self):
        d, got = self._old(self._skill)
        self.assertEqual(got[0], self._ref(), d)

    def test_a_wrapper_then_a_compaction(self):
        d, got = self._old(self._wrapper, compaction_after=True)
        self.assertEqual(got[1], self._ref(), d)


class TheCutSearchIsLinear(R.R2Base):
    """Should-fix: a kept prompt in the last turn stamped near the start steps the writer's cut back one turn per candidate;
    each candidate re-scanned every kept record and every record (d088e830b: 2.1 s at 1,000 turns, 69.8 s at 4,000 in the
    review's probe; 5.9 s against 0.5 s for the same leaf without the stale stamp at 2,000 turns here). The search now reads
    per-candidate facts precomputed once, so the stale leaf's write costs about what the plain leaf's does."""

    def _time_write(self, turns, stale):
        from test_asm_checkpoint_served import transcript
        recs = transcript(NOW - 86400, turns=turns, compact_every=100000)
        last, t = recs[-1]["uuid"], NOW - 86400 + turns * 60
        recs.append({"type": "user", "uuid": "stale", "parentUuid": last, "promptSource": "typed", "cwd": "/w/notes-api",
                     "timestamp": iso(NOW - 86400 + 30 if stale else t + 10),
                     "message": {"role": "user", "content": "a prompt stamped near the start"}})
        recs.append({"type": "assistant", "uuid": "stalea", "parentUuid": "stale", "timestamp": iso(t + 99), "cwd": "/w/notes-api",
                     "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}})
        recs += R.tail_turn(9000, "stalea", t + 200)
        Path(self.path).write_text("".join(json.dumps(r) + "\n" for r in recs))
        old = time.time() - 600
        os.utime(self.path, (old, old))
        ck = self.td / "checkpoints"
        for p in (list(ck.iterdir()) if ck.exists() else []):
            p.unlink()
        self._reset()
        with T.knobs(0, 4, roots=[str(self.proj)]):
            tree = self.parse()
            t0 = time.perf_counter()
            ok = em.asm_checkpoint_write(self.path, SID, tree=tree)
            dt = time.perf_counter() - t0
            self.assertTrue(ok, em.asm_checkpoint_stats().get("skipped"))
            return dt

    def test_a_stale_stamp_costs_about_a_plain_write(self):
        plain = min(self._time_write(2000, False) for _ in range(2))
        stale = min(self._time_write(2000, True) for _ in range(2))
        sys.stderr.write("cut search at 2,000 turns: plain %.2f s, stale %.2f s\n" % (plain, stale))
        self.assertLess(stale, 2 * plain + 0.25, (plain, stale))


if __name__ == "__main__":
    unittest.main()
