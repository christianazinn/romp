"""Tail-only record cache, round two (2026-10-08): the two defects the first live deploy measured.

1. Records were not freed, only moved out of the cache's count: a large leaf's assembly entry stayed WHOLE (its document
   was over the flat 16 MiB cap, and an idle leaf is never parsed again to re-seat), and the parse store's tree built from it
   held every body. Now a tail-only leaf's document may reach 1/16 of its pre-cut bytes, the writer releases the whole entry
   at once, and the kernel replaces the store's tree with a restore.
2. Four routine callers (tasks_for, _plan_session, _auto_nudge_session, _bg_placed_tops) read whole transcripts back from
   disk: each reached the records only through a WHOLE parse of the session, taken when a restored entry demoted. Now a
   restored entry of a tail-only leaf demoted for a reason the chain proof covers restores again (the boot's road), and its
   leaf's record window is pinned to the document's cut, so the pass reads nothing before the window.

Every test here fails on the round-one head (62e03751a). Synthetic only: invented text, placeholder uuids."""
import gc
import json
import os
import shutil
import sys
import tempfile
import time
import tracemalloc
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import test_record_cache_tail_only as T                 # noqa: E402  one kernel copy for both files: its km, jd, em and helpers
from test_asm_checkpoint_served import transcript, iso  # noqa: E402  the stage 4a fixture's builder (synthetic)

km, jd, em = T.km, T.jd, T.em
SID = T.SID
NOW = T.NOW
WINDOW = 16 * 1024


def cold():
    """(records read before a window, by caller) so far."""
    st = em.cold_read_stats()
    return st["records"], {k: v["records"] for k, v in st["byCaller"].items()}


def tail_turn(k, parent, t, boundary=False):
    """One appended turn chained on `parent`, with a compaction (boundary and summary) before it when asked."""
    out = []
    if boundary:
        out.append({"type": "system", "subtype": "compact_boundary", "uuid": "nb%d" % k, "parentUuid": None,
                    "logicalParentUuid": parent, "timestamp": iso(t),
                    "compactMetadata": {"trigger": "auto", "preTokens": 160000, "postTokens": 9000}})
        out.append({"type": "user", "uuid": "ns%d" % k, "parentUuid": "nb%d" % k, "timestamp": iso(t + 1), "isCompactSummary": True,
                    "message": {"role": "user", "content": "summary so far: retry budget cache window %d" % k}})
        parent, t = "ns%d" % k, t + 2
    out.append({"type": "user", "uuid": "nu%d" % k, "parentUuid": parent, "timestamp": iso(t), "promptSource": "typed",
                "cwd": "/w/notes-api", "message": {"role": "user", "content": "next step %d: tighten the retry cap" % k}})
    out.append({"type": "assistant", "uuid": "na%d" % k, "parentUuid": "nu%d" % k, "timestamp": iso(t + 20), "cwd": "/w/notes-api",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "done with step %d " % k + "ok " * 40}],
                            "stop_reason": "end_turn"}})
    return out


class R2Base(unittest.TestCase):
    """A rebound judge state, a projects root the tail rule windows, a checkpoint directory; a leaf of TURNS turns with a
    compaction every 40 (160 turns: 306 KB), so a 16 KiB window holds it tail-only. An appended turn is 581 bytes, 1,023
    with a compaction."""
    TURNS = 160

    def setUp(self):
        self.td = Path(tempfile.mkdtemp()).resolve()
        (self.td / "session-hosts").write_text("off\n")      # CLAUDE.md: a minted state root never spawns a session host
        self.proj = self.td / "projects" / "-w-notes-api"
        self.proj.mkdir(parents=True)
        self.saved = (jd.STATE, jd.PROJECTS, jd.plan_llm, jd.opener_llm, jd._group_store)
        jd._rebind_state(self.td)
        jd.PROJECTS = self.td / "projects"
        jd.GOALDIR.mkdir(parents=True, exist_ok=True)
        (self.td / "states").mkdir(exist_ok=True)
        (self.td / "judge-units-cache").mkdir(exist_ok=True)
        em.set_checkpoint_dir(lambda: self.td / "checkpoints")
        self._reset()
        self.path = str(self.proj / (SID + ".jsonl"))
        self.recs = transcript(NOW - 86400, turns=self.TURNS, compact_every=40)
        Path(self.path).write_text("".join(json.dumps(r) + "\n" for r in self.recs))
        self.parent, self.t, self.k = self.recs[-1]["uuid"], NOW - 86400 + self.TURNS * 60 + 600, 0

    def tearDown(self):
        em.set_checkpoint_dir(None)
        jd._rebind_state(self.saved[0])
        jd.PROJECTS, jd.plan_llm, jd.opener_llm, jd._group_store = self.saved[1:]
        self._reset()
        shutil.rmtree(self.td, ignore_errors=True)

    def _reset(self):
        T.fresh()
        getattr(em, "_ASM_RESEAT_DUE", {}).clear()       # a restart forgets a released entry's re-seat as it forgets the entry
        km._parse_cache.clear()
        jd._PARSE_CACHE.clear(); jd._CHAIN_MEMO.clear()
        km._BG_TOPS_CACHE.clear(); km._task_seg_cache.clear(); km._PLACEMENT_IDX.clear()
        for k in [k for k in em._ASM_STATS if k not in ("full", "fold", "serve", "restore", "bypass", "fallback")]:
            em._ASM_STATS.pop(k, None)

    def parse(self):
        return em.parse_session(self.path, rompuuid=SID, name="impl", dir="/TESTDIR", candidate_files=[self.path],
                                states=None, postal_log=[], now=NOW)

    def append(self, boundary=False):
        recs = tail_turn(self.k, self.parent, self.t, boundary=boundary)
        with open(self.path, "a") as f:
            f.write("".join(json.dumps(r) + "\n" for r in recs))
        self.parent, self.t, self.k = recs[-1]["uuid"], self.t + 120, self.k + 1
        return recs

    def restored(self):
        """The kernel's life before the measured one: a whole parse, its document written (a settle), a restart. The first
        parse after it restores from the document: the entry the routine callers' passes meet."""
        tree = self.parse()
        self.assertTrue(em.asm_checkpoint_write(self.path, SID, tree=tree), em.asm_checkpoint_stats())
        del tree
        self._reset()
        self.parse()
        key = (os.path.realpath(self.path), SID, False)
        self.assertIsNotNone(em._ASM_CACHE[key].get("docPre"), "the parse after the restart restored from the document")
        ent = em._JSONL_CACHE.get(self.path)
        self.assertTrue(ent is not None and T._tail(ent[4]) or (ent is not None and ent[5] > 0),
                        "the leaf is held from the document's cut or tail-only")
        self._reset_parse_store()

    def _reset_parse_store(self):
        km._parse_cache.clear()
        jd._PARSE_CACHE.clear(); jd._CHAIN_MEMO.clear()

    def save_store(self):
        g = {"id": SID + ":g1", "text": "Tighten the retry cap", "parentId": None, "nodeComplete": False, "blocked": False,
             "cleared": False, "trail": [], "t": NOW - 86400}
        (jd.GOALDIR / (SID + ".json")).write_text(json.dumps(
            {"rompUuid": SID, "seq": 1, "lastNode": g["id"], "closedTurns": [], "nodes": {g["id"]: g}, "placements": {},
             "status": {g["id"]: "working"}, "placementsV": jd.PLACEMENTS_V}))


class RoutineCallersReadNothingBeforeTheWindow(R2Base):
    """Defect 2, one test per caller. A pass: a turn whose compaction lands in the restored entry's tail (the fold cannot
    carry it, g:boundary), then the caller. On 62e03751a the demotion takes the WHOLE parse, which streams every record
    before the window off disk under the caller's name; now it restores from the document with the window pinned to the cut."""

    def _passes(self, call, name, n=2):
        with T.knobs(WINDOW, 4, roots=[str(self.proj)]):
            self.restored()
            call(-1)                                      # the caller's earlier pass: its own folds (the judges' background-task
            #                                               fold) hold cursors, as a fold checkpoint restores them at a live boot
            for i in range(n):
                self.append(boundary=True)
                c0, by0 = cold()
                call(i)
                c1, by1 = cold()
                self.assertEqual(by1.get(name, 0) - by0.get(name, 0), 0,
                                 "pass %d: %s read no record before the window (roads %r)" % (i, name, dict(em._ASM_STATS)))
                self.assertEqual(c1 - c0, 0, "pass %d: nothing at all was read before the window: %r" % (i, {k: v - by0.get(k, 0) for k, v in by1.items() if v != by0.get(k, 0)}))
            self.assertGreaterEqual(em._ASM_STATS.get("g:boundary", 0), n, "each pass demoted: %r" % dict(em._ASM_STATS))
            self.assertGreaterEqual(em._ASM_STATS.get("restore:afterDemote", 0), n,
                                    "each pass's demotion took the restore road: %r" % dict(em._ASM_STATS))

    def test_tasks_for(self):
        self.save_store()
        self._passes(lambda i: jd.tasks_for(SID, self.path, [self.path], NOW + i), "tasks_for")

    def test_plan_session(self):
        self.save_store()
        jd.plan_llm = jd.opener_llm = lambda *a, **k: '{"ops":[]}'
        jd._group_store = lambda *a, **k: None
        self._passes(lambda i: jd._plan_session(SID, self.path, NOW + i), "_plan_session")

    def test_auto_nudge_session(self):
        self.save_store()
        row = {"sid": SID, "path": self.path, "name": "api", "mtime": NOW}
        self._passes(lambda i: km._auto_nudge_session(row, NOW + i, {}, {}, {}, alive_ids={SID}), "_auto_nudge_session")

    def test_bg_placed_tops(self):
        self.save_store()
        self._passes(lambda i: km._bg_placed_tops(SID, self.path, ("toolu_none_%d" % i,)), "_bg_placed_tops")


def _decode_lines():
    """The (file, line) pairs of the reader's record decodes: a traced block allocated under one of them is a decoded record
    (the scan's, or a streaming pass's before a window)."""
    out = set()
    for code in (em._scan_jsonl_stream.__code__, em._TailRecords._decode_run.__code__):
        out |= {(code.co_filename, ln) for _a, _b, ln in code.co_lines() if ln}
    return out


def held_record_bytes(snap, lines):
    return sum(st.size for st in snap.statistics("traceback")
               if any((fr.filename, fr.lineno) in lines for fr in st.traceback))


class AnIdleLeafReleasesItsRecordsOnceItsDocumentStands(R2Base):
    """Defect 1. An idle leaf the boot parsed whole: the kernel's display parse holds its tree, the converge pass writes its
    document. On 62e03751a the whole assembly entry and the store's tree stayed (the next parse would re-seat, and an idle
    leaf is never parsed again), so every decoded record stayed alive though the record cache counted only the window. Now the
    writer releases the entry and the kernel replaces the tree with a restore: the decoded records still held fall to the
    window's (measured by tracemalloc over the reader's decode sites)."""

    def test_the_held_decoded_records_fall_to_the_window(self):
        old = time.time() - 600
        os.utime(self.path, (old, old))                  # quiescent: the converge pass's leaf
        lines = _decode_lines()
        with T.knobs(WINDOW, 4, roots=[str(self.proj)]):
            tracemalloc.start(40)
            try:
                km._parse(self.path, SID, NOW)
                gc.collect()
                before = held_record_bytes(tracemalloc.take_snapshot(), lines)
                for name, val in (("CKPT_CONVERGE_MS", 5000.0), ("CKPT_CONVERGE_BYTES", em._CKPT_CYCLE_CAP_DEFAULT),
                                  ("ASM_CONVERGE", True)):
                    saved = getattr(km, name); setattr(km, name, val); self.addCleanup(setattr, km, name, saved)
                km._ASM_CONVERGE_DONE.clear(); km._ASM_CONVERGE_BLIP.clear(); km._ASM_CONVERGE_NOENTRY.clear()
                km._begin_checkpoint_cycle()
                self.assertEqual(km._converge_assembly(time.time(), time.monotonic()), 1, em.asm_checkpoint_stats())
                gc.collect()
                after = held_record_bytes(tracemalloc.take_snapshot(), lines)
            finally:
                tracemalloc.stop()
            hot = em._JSONL_CACHE[self.path][4]
            hot_n = len(hot.hot) if T._tail(hot) else len(hot)
            tree = jd._parse_slot(SID, jd._pending_cut(SID), self.path, km._display_sdk_human(SID))
            self.assertIsNotNone(tree, "the store holds a tree again (the feed's cache-only read never meets an empty slot)")
        self.assertGreater(before, 0)
        self.assertLess(after, 0.3 * before, "decoded records held: %d bytes after the document, %d before (the window holds "
                                             "%d of %d records)" % (after, before, hot_n, len(self.recs)))


class ALargeLeafsDocumentIsWritten(R2Base):
    """Defect 1's cause on the devbox: a large leaf's document exceeded the flat cap, so it was never written and the entry
    stayed whole. Scaled here: the flat cap shrunk to 2 KiB, under this leaf's document and over 1/16 of its pre-cut bytes'
    worth of nothing. A tail-only leaf's document is written (the cap is 1/16 of its pre-cut bytes); with the rule off, the
    same leaf still refuses at the flat cap (the control)."""

    def _write(self, tail_bytes):
        saved = em._ASM_CKPT_CAP, getattr(em, "_ASM_CKPT_CAP_SHARE", None)
        em._ASM_CKPT_CAP = 2048
        em._ASM_CKPT_CAP_SHARE = 2                        # this fixture's records are tiny (its document is 7 percent of it; the
        #                                                   devbox's large leaves measured 0.7 to 3.3 percent), so the share scales too
        try:
            with T.knobs(tail_bytes, 4, roots=[str(self.proj)]):
                tree = self.parse()
                reasons = []
                ok = em.asm_checkpoint_write(self.path, SID, tree=tree, reason_out=reasons)
                cp = em._asm_ckpt_file(self.path)
                return ok, reasons, (cp.stat().st_size if cp.exists() else None)
        finally:
            em._ASM_CKPT_CAP = saved[0]
            if saved[1] is None:
                del em._ASM_CKPT_CAP_SHARE
            else:
                em._ASM_CKPT_CAP_SHARE = saved[1]

    def test_a_tail_only_leaf_writes_past_the_flat_cap(self):
        ok, reasons, size = self._write(WINDOW)
        self.assertTrue(ok, "written: %r" % reasons)
        self.assertGreater(size, 2048, "the document is over the flat cap")
        self.assertLessEqual(size, os.path.getsize(self.path) // 2)

    def test_control_the_rule_off_keeps_the_flat_cap(self):
        ok, reasons, size = self._write(0)
        self.assertFalse(ok)
        self.assertEqual(reasons, ["oversize"])


class TheNewRestoreRoadsEqualAWholeParse(R2Base):
    """The demotions a restored tail-only entry now takes to the restore road give the tree a cold whole parse gives (the
    rule off, no checkpoint directory), hydrated; and they take that road (on 62e03751a: the whole parse)."""

    def _ref(self):
        with T.knobs(0, 4, roots=[str(self.proj)]):
            self._reset()
            saved = em._CKPT_DIR_FN; em._CKPT_DIR_FN = None
            try:
                return T._strip_tree(self.parse())
            finally:
                em._CKPT_DIR_FN = saved

    def _check(self, mutate, reason):
        with T.knobs(WINDOW, 4, roots=[str(self.proj)]):
            self.restored()
            mutate()
            c0, _ = cold()
            s0 = dict(em._ASM_STATS)
            tree = self.parse()
            d = {k: v - s0.get(k, 0) for k, v in em._ASM_STATS.items() if v != s0.get(k, 0)}
            self.assertEqual(d.get("g:" + reason, 0), 1, "the fold demoted for %s: %r" % (reason, d))
            self.assertEqual(d.get("restore:afterDemote", 0), 1, "…and restored, never parsed whole: %r" % d)
            self.assertNotIn("full", d)
            self.assertEqual(cold()[0] - c0, 0, "nothing read before the window")
            em.hydrate(tree, SID)
            got = T._strip_tree(tree)
        self.assertEqual(got, self._ref())

    def test_a_compaction_in_the_tail(self):
        self._check(lambda: self.append(boundary=True), "boundary")

    def test_a_record_written_twice_in_the_tail(self):
        def mutate():
            recs = self.append()
            self.parse()                                  # folded: the record is in the entry's graph
            with open(self.path, "a") as f:
                f.write(json.dumps(recs[-1]) + "\n")      # the CLI re-writes a record (a verbatim duplicate)
            self.append()
        self._check(mutate, "uuid-known")


class TheWindowStaysAtARestoredCut(R2Base):
    """The record window of a leaf with a live restored entry never starts after the entry's cut, a window that slid past it
    already comes back once (the gap read once, counted), and the window slides again when the entry goes. A 400-turn leaf
    (768 KB): its churn bound (1/8 of the pre-cut bytes, about 95 KB of tail) stays far off while the tail passes the 32 KiB
    the rule windows from."""
    TURNS = 400

    def _cut_index(self):
        key = (os.path.realpath(self.path), SID, False)
        return em._ASM_CACHE[key]["docCutOff"]

    def test_appends_past_the_window_keep_the_cut_and_a_whole_parse_releases_it(self):
        with T.knobs(WINDOW, 4, roots=[str(self.proj)]):
            self.restored()
            cut = self._cut_index()
            for _ in range(70):                           # about 40 KB: past the 32 KiB span the rule windows, past the window
                self.append()
                self.parse()
            ent = em._JSONL_CACHE[self.path]
            first = ent[4].hot_offset() if T._tail(ent[4]) else int(ent[7][0])
            self.assertLessEqual(first, cut, "the window still starts at or before the restored cut")
            self.assertEqual(em.tail_pin_stats().get(self.path), cut)
            em.evict_document(self.path)                  # the entry goes (its release unpins)
            self.assertIsNone(em.tail_pin_stats().get(self.path))

    def test_a_window_that_slid_past_the_cut_comes_back_once(self):
        with T.knobs(WINDOW, 4, roots=[str(self.proj)]):
            tree = self.parse()
            self.assertTrue(em.asm_checkpoint_write(self.path, SID, tree=tree), em.asm_checkpoint_stats())
            del tree
            for _ in range(40):                           # about 23 KB: the window of a whole read starts past the cut
                self.append()
            self._reset()
            em._read_jsonl_entry(self.path)               # a whole reader first: a tail-only entry whose window starts past the cut
            ent = em._JSONL_CACHE[self.path]
            self.assertTrue(T._tail(ent[4]))
            self.parse()                                  # the restore pins and widens
            c0, _ = cold()
            for _ in range(3):
                self.append(boundary=True)                # each a demotion and a restore over the pinned span
                self.parse()
            self.assertEqual(cold()[0] - c0, 0, "after the one widening read, nothing before the window")


class TheHydrationMemoKeepsWhatAtomsRead(unittest.TestCase):
    """The hydration memo held every hydrated record whole: a tool result's output twice (its message block and
    toolUseResult) and its metadata, beside the message the atom shares. It keeps the fields the atom's kind reads; an atom of
    the same uuid that needs a dropped field reads the record again (on 62e03751a the helpers do not exist)."""

    REC = {"type": "user", "uuid": "u-tr", "parentUuid": "a-tr", "timestamp": "2026-10-07T00:00:00.000Z", "cwd": "/w/notes-api",
           "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "out " * 50}]},
           "toolUseResult": {"stdout": "out " * 50, "stderr": ""}}

    def setUp(self):
        with em._ASM_CKPT_LOCK:
            em._HYDRATED.pop("u-tr", None)

    tearDown = setUp

    def test_a_tool_result_without_tur_keeps_no_second_copy(self):
        keep = em._hydrated_keep(self.REC, {"k": "u"})
        self.assertNotIn("toolUseResult", keep)
        self.assertNotIn("cwd", keep)
        self.assertIs(keep["message"], self.REC["message"])
        with em._ASM_CKPT_LOCK:
            em._HYDRATED["u-tr"] = (keep, 100)
            self.assertIsNotNone(em._hydrated_hit("u-tr", {"uuid": "u-tr", "lazy": {"k": "u"}}))
            self.assertIsNone(em._hydrated_hit("u-tr", {"uuid": "u-tr", "lazy": {"k": "u", "tur": True}}),
                              "an atom that needs the dropped toolUseResult reads the record again")

    def test_with_tur_the_atom_gets_it(self):
        keep = em._hydrated_keep(self.REC, {"k": "u", "tur": True})
        a = {"uuid": "u-tr", "lazy": {"k": "u", "tur": True}}
        em._hydrate_one(a, keep)
        self.assertEqual(a["toolUseResult"], self.REC["toolUseResult"])
        self.assertEqual(a["message"], em._norm_message(self.REC["message"]))


if __name__ == "__main__":
    unittest.main()
