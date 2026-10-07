#!/usr/bin/env python3
"""The record cache holds a large transcript's TAIL only (2026-10-07).

The reader kept every live transcript whole, about 3 bytes of memory per file byte, so a machine whose live transcripts
were 20 GiB on disk held about 61 GiB of decoded records. A transcript (a file under the Claude projects root) whose held
span reaches _TAIL_MIN_FILE_BYTES now keeps only its newest records in memory (the last _TAIL_BYTES or the last
_TAIL_RECORDS, whichever is more); every older record stays indexed by byte offset and is read off disk when asked for,
through a read-only sequence with the whole list's length and order. Pinned here, on synthetic transcripts only:

  - a 200 MB transcript read through the whole reader holds only its tail: the cache's own weight and tracemalloc both
    say so, for what stays held and for the read's peak, and its first and last records are the file's;
  - an append extends the window and, past one and a half windows, slides it, under the same generation;
  - every index, slice, walk and reversed walk equals a plain read's; a fold from record 0, a fold whose cursor fell
    behind the window, a checkpoint write and a fresh process's restore all get every record they need;
  - nothing can see a truncated view: the C json encoder refuses the sequence, and a file rewritten under the index
    raises rather than serving a wrong record;
  - other large files (the postal log's case) stay whole, and ROMP_RECORD_CACHE_TAIL_MB=0 turns the rule off;
  - the assembly's whole parse, its document and a fresh process's restore over a tail-only leaf equal the rule-off
    parse, and the chat build's scroll-back pages over it equal the whole build.

Every test asserts the entry it reads is tail-only, so each fails on a base that holds every entry whole."""
import array
import json
import os
import random
import shutil
import sys
import tempfile
import time
import tracemalloc
import unittest
from contextlib import contextmanager
from pathlib import Path
from romp_load import load_source

HERE = os.path.dirname(os.path.realpath(__file__))
BIN = os.path.join(os.path.dirname(HERE), "bin")
os.environ["XDG_STATE_HOME"] = tempfile.mkdtemp()
os.environ.pop("ROMP_STATE_DIR", None)
os.environ["ROMP_KERNEL_NO_OPEN"] = "1"
os.environ.setdefault("ROMP_SERVE_TOKEN", "testtok")
for _k in ("ROMP_RECORD_CACHE_TAIL_MB", "ROMP_RECORD_CACHE_TAIL_RECORDS", "ROMP_RECORD_CACHE_TAIL_ROOTS"):
    os.environ.pop(_k, None)                     # the module's defaults, whatever the shell running the suite exports
km = load_source("romp_kernel_tail_only", os.path.join(BIN, "romp-kernel"))
jd, em = km.jd, km.em
sys.path.insert(0, HERE)
from test_asm_checkpoint_served import transcript    # noqa: E402  the stage 4a served fixture's builder (synthetic)

SID = "aaaaaaaa-4444-4222-8333-555555555555"
NOW = 1781200000
MIB = 1024 * 1024
WORDS = ("alpha beta gamma delta retry budget cache window tail fold cursor index offset append slide page scroll "
         "notes api web tests make build lint route handler schema migration").split()


def _tail(recs):
    return type(recs).__name__ == "_TailRecords"


@contextmanager
def knobs(tail_bytes, records=4, roots=None):
    """The rule's knobs for one test (module globals the reader reads at call time); restored after. On a base without the
    rule the attributes are set and ignored, so a test fails on its behaviour, not on a missing name."""
    names = ("_TAIL_BYTES", "_TAIL_MIN_FILE_BYTES", "_TAIL_RECORDS", "_COLD_CHUNK_BYTES")
    saved = {n: getattr(em, n, None) for n in names}
    em._TAIL_BYTES, em._TAIL_MIN_FILE_BYTES, em._TAIL_RECORDS = int(tail_bytes), 2 * int(tail_bytes), int(records)
    em._COLD_CHUNK_BYTES = max(4096, int(tail_bytes) // 2)       # several runs per streaming pass, so the run seams are crossed
    set_roots = getattr(em, "set_tail_roots", None)
    if set_roots is not None:
        set_roots(roots)
    try:
        yield
    finally:
        for n, v in saved.items():
            if v is None:
                try:
                    delattr(em, n)
                except AttributeError:
                    pass
            else:
                setattr(em, n, v)
        if set_roots is not None:
            set_roots(None)


def fresh():
    """A kernel restart's in-memory side (the record cache, the assembly cache, the checkpoints' pending restores)."""
    with em._JSONL_CACHE_LOCK:
        em._JSONL_CACHE.clear()
        em._JSONL_CACHE_BYTES[0] = 0
    with em._ASM_LOCK:
        em._ASM_CACHE.clear()
    em._TRAILING_CACHE.clear()
    with em._ASM_CKPT_LOCK:
        em._HYDRATED.clear(); em._HYDRATED_BYTES[0] = 0
    em._LAZY_FILES.clear()
    memo = getattr(em, "_ASM_DOC_MEMO", None)
    if memo is not None:
        with em._ASM_CKPT_LOCK:
            memo.clear()
            getattr(em, "_ASM_DOC_MEMO_BYTES", [0])[0] = 0
    with em._CKPT_LOCK:
        em._CKPT_PENDING.clear(); em._CKPT_SEQ.clear(); em._FOLD_DIRTY.clear(); em._CKPT_DOC_FOLDS.clear()
        em._COLD_FOLDS.clear(); em._COLD_REASONS.clear(); em._COLD_OVER_KB.clear()
        em._DOC_MEMO.clear(); em._DOC_MEMO_BYTES[0] = 0
    with em._READ_BYTES_LOCK:
        em._READ_BYTES.clear()
    em._LAST_ENTRY.ent = None


def _rec(i, rnd, big=False):
    """One synthetic transcript record of a coding session's shape (invented text, placeholder ids)."""
    k = i % 4
    u = "11111111-2222-4333-8444-%012d" % i
    p = "11111111-2222-4333-8444-%012d" % (i - 1) if i else None
    t = "2026-10-07T00:%02d:%02d.000Z" % ((i // 60) % 60, i % 60)
    if k == 0:
        return {"type": "user", "uuid": u, "parentUuid": p, "timestamp": t, "cwd": "/w/notes-api",
                "message": {"role": "user", "content": " ".join(rnd.choice(WORDS) for _ in range(12)) + " %d" % i}}
    if k == 1:
        return {"type": "assistant", "uuid": u, "parentUuid": p, "timestamp": t, "cwd": "/w/notes-api",
                "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_%d" % i, "name": "Bash",
                                                              "input": {"command": "make test # %d" % i}}],
                            "usage": {"input_tokens": 3 + i % 7, "output_tokens": 40}}}
    if k == 2:
        n = rnd.choice((40, 80, 160, 400, 1600, 6000)) * (8 if big else 1)
        return {"type": "user", "uuid": u, "parentUuid": p, "timestamp": t, "cwd": "/w/notes-api",
                "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_%d" % (i - 1),
                                                         "content": " ".join(rnd.choice(WORDS) for _ in range(n))}]}}
    return {"type": "assistant", "uuid": u, "parentUuid": p, "timestamp": t, "cwd": "/w/notes-api",
            "message": {"role": "assistant", "content": [{"type": "text", "text": " ".join(rnd.choice(WORDS) for _ in range(60))}],
                        "stop_reason": "end_turn"}}


def write_transcript(path, target_bytes, start=0, seed=3, big=False, mode="w"):
    """Records appended until the file holds `target_bytes` more; returns (records written, the next index). A blank line
    and a line that is not JSON sit among them, as the reader skips such lines."""
    rnd = random.Random(seed + start)
    out, i, n = [], start, 0
    with open(path, mode) as f:
        while n < target_bytes:
            r = _rec(i, rnd, big)
            line = json.dumps(r) + "\n"
            f.write(line); n += len(line)
            out.append(r)
            if i % 997 == 13:
                f.write("\n{not json\n"); n += 11
            i += 1
    return out, i


def plain(path):
    """The reference: every record of the file as the reader's own reference scanner decodes it."""
    return list(em._read_jsonl(path))


class Base(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp()).resolve()
        self.proj = self.td / "projects" / "-w-notes-api"
        self.proj.mkdir(parents=True)
        self.ck = self.td / "checkpoints"
        em.set_checkpoint_dir(lambda: self.ck)
        fresh()

    def tearDown(self):
        em.set_checkpoint_dir(None)
        fresh()
        shutil.rmtree(self.td, ignore_errors=True)

    def leaf(self, name=SID):
        return str(self.proj / (name + ".jsonl"))


class LargeTranscriptHoldsItsTail(Base):
    """The headline: a 200 MB transcript at the default window (32 MiB, 2,000 records)."""

    def test_a_200_mb_transcript_holds_only_its_tail_in_memory(self):
        with knobs(32 * MIB, 2000, roots=[str(self.proj)]):
            path = self.leaf()
            recs, _ = write_transcript(path, 200 * 1000 * 1000, big=True)
            size = os.path.getsize(path)
            first, last, count = recs[0], recs[-1], len(recs)
            del recs
            tracemalloc.start()
            try:
                base_now, _ = tracemalloc.get_traced_memory()
                tracemalloc.reset_peak()
                ent = em._read_jsonl_entry(path)              # the WHOLE reader (tail_ok False): what a parse asks for
                held, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            got = ent[4]
            self.assertTrue(_tail(got), "the entry of a %d-byte transcript is held tail-only" % size)
            self.assertEqual((ent[5], len(got)), (0, count), "base 0 and every record counted: the whole list's shape")
            self.assertEqual(got[0], first, "the first record comes off disk")
            self.assertEqual(got[-1], last, "the last record comes out of the window")
            window = size - got.hot_offset()
            self.assertLessEqual(window, 32 * MIB + 400 * 1024, "the window is the last 32 MiB (plus at most one record)")
            stats = em.record_cache_stats()
            self.assertLess(stats["bytes"], int(3 * (32 * MIB + 400 * 1024)) + 16 * count + 1,
                            "the cache weighs the window and the index, not the file: %d" % stats["bytes"])
            self.assertLess(held - base_now, 160 * MIB,
                            "tracemalloc: %.1f MB held after reading a %.1f MB file (whole: about 3 bytes a file byte)"
                            % ((held - base_now) / 1e6, size / 1e6))
            self.assertLess(peak - base_now, 200 * MIB, "tracemalloc: the read peaked at %.1f MB: the scan drops what it "
                                                        "decodes before the window" % ((peak - base_now) / 1e6))
            again = em._read_jsonl_entry(path)
            self.assertIs(again, ent, "an unchanged file is a hit: the same entry, nothing read")


class TheWindow(Base):
    def _read(self, path):
        return em._read_jsonl_entry(path)

    def test_appends_extend_the_window_and_slide_it_under_the_same_generation(self):
        with knobs(64 * 1024, 10, roots=[str(self.proj)]):
            path = self.leaf()
            _, nxt = write_transcript(path, 1 * MIB)
            e1 = self._read(path)
            self.assertTrue(_tail(e1[4]))
            c1, h1 = e1[4].ncold, len(e1[4].hot)
            _, nxt = write_transcript(path, 8 * 1024, start=nxt, mode="a")       # a small append: inside the hysteresis
            e2 = self._read(path)
            self.assertEqual(e2[6], e1[6], "an append keeps the generation (every fold cursor stands)")
            self.assertEqual(e2[4].ncold, c1, "a small append extends the window without sliding it")
            self.assertGreater(len(e2[4].hot), h1)
            _, nxt = write_transcript(path, 200 * 1024, start=nxt, mode="a")     # past one and a half windows: it slides
            e3 = self._read(path)
            self.assertEqual(e3[6], e1[6])
            self.assertGreater(e3[4].ncold, c1, "the window slid forward")
            self.assertLessEqual(e3[1] - e3[4].hot_offset(), 64 * 1024 + 64 * 1024, "back to about one window")
            self.assertEqual(list(e3[4]), plain(path), "the records are the file's, every one, in order")
            self.assertIsNot(e3[4].hot, e2[4].hot, "a NEW list per append: a served one never changes under its holder")

    def test_every_index_slice_and_walk_equals_a_plain_read(self):
        with knobs(32 * 1024, 4, roots=[str(self.proj)]):
            path = self.leaf()
            write_transcript(path, 600 * 1024)
            got, ref = self._read(path)[4], plain(path)
            self.assertTrue(_tail(got))
            n = len(ref)
            self.assertEqual(len(got), n)
            self.assertEqual(list(got), ref)
            self.assertEqual(list(reversed(got)), ref[::-1])
            rnd = random.Random(5)
            for _ in range(300):
                i = rnd.randrange(-n, n)
                self.assertEqual(got[i], ref[i])
                a, b = sorted(rnd.randrange(-n - 3, n + 3) for _ in range(2))
                sl = got[a:b]
                self.assertEqual(len(sl), len(ref[a:b]))
                self.assertEqual(list(sl), ref[a:b])
                if len(sl):
                    j = rnd.randrange(len(sl))
                    self.assertEqual(sl[j], ref[a:b][j])
                    self.assertEqual(list(sl[j:]), ref[a:b][j:], "a slice of a slice")
                    self.assertEqual(list(reversed(sl)), ref[a:b][::-1])
            self.assertEqual(got[::7], ref[::7])
            with self.assertRaises(IndexError):
                got[n]
            self.assertTrue(bool(got))

    def test_a_fold_from_record_0_streams_the_older_records_and_leaves_the_entry_tail_only(self):
        with knobs(32 * 1024, 4, roots=[str(self.proj)]):
            path = self.leaf()
            write_transcript(path, 600 * 1024)
            ent = self._read(path)
            self.assertTrue(_tail(ent[4]))
            c0 = em.cold_read_stats()
            cache = {}
            step = lambda st, r: st + [r.get("uuid")]
            state = em.fold_records(cache, path, list, step)          # no cursor: a refold over every record
            self.assertEqual(state, [r.get("uuid") for r in plain(path)])
            c1 = em.cold_read_stats()
            self.assertEqual(c1["records"] - c0["records"], ent[4].ncold, "every record before the window read once")
            self.assertLessEqual(c1["passes"] - c0["passes"], 1, "in one streaming pass, never a read per record")
            after = em._JSONL_CACHE.get(path)
            self.assertTrue(after is not None and _tail(after[4]) and after[6] == ent[6], "the entry stays tail-only, same gen")
            write_transcript(path, 4 * 1024, start=100000, mode="a")
            state2 = em.fold_records(cache, path, list, step)          # an append: the new records alone, from the window
            self.assertEqual(state2, [r.get("uuid") for r in plain(path)])
            self.assertEqual(em.cold_read_stats()["records"], c1["records"], "an append reads nothing before the window: %r" % em.cold_read_stats())

    def test_a_fold_whose_cursor_fell_behind_the_window_steps_the_gap_from_disk(self):
        with knobs(32 * 1024, 4, roots=[str(self.proj)]):
            path = self.leaf()
            write_transcript(path, 600 * 1024)
            ent = self._read(path)
            ref = plain(path)
            behind = ent[4].ncold // 2
            cache = {path: (behind, ent[6], [r.get("uuid") for r in ref[:behind]])}
            state = em.fold_records(cache, path, list, lambda st, r: st + [r.get("uuid")])
            self.assertEqual(state, [r.get("uuid") for r in ref])

    def test_checkpoint_write_and_a_fresh_processs_restore_over_a_tail_only_entry(self):
        with knobs(32 * 1024, 4, roots=[str(self.proj)]):
            path = self.leaf()
            write_transcript(path, 600 * 1024)
            cache = {}
            count = lambda st, r: st + 1
            em.fold_records(cache, path, int, count, ckpt="tailCount")
            self.assertTrue(_tail(em._JSONL_CACHE[path][4]))
            self.assertTrue(em.checkpoint_write(path))
            n = len(plain(path))
            fresh()
            write_transcript(path, 4 * 1024, start=100000, mode="a")
            cache2 = {}
            em.name_fold_cache(cache2, "tailCount")
            got = em.fold_records(cache2, path, int, count, ckpt="tailCount")
            self.assertEqual(got, len(plain(path)))
            self.assertGreater(got, n)
            self.assertLess(em.read_bytes_report().get(path, 0), 64 * 1024, "the fresh process read the tail past the cut only")

    def test_a_rewritten_file_refuses_rather_than_serving_a_wrong_record(self):
        with knobs(32 * 1024, 4, roots=[str(self.proj)]):
            path = self.leaf()
            write_transcript(path, 300 * 1024)
            ent = self._read(path)
            self.assertTrue(_tail(ent[4]))
            held = ent[4]
            write_transcript(path, 300 * 1024, seed=99)                          # a rewrite: other records, about the same size
            with self.assertRaises(em.TailRecordsRead):
                held[0]
            with self.assertRaises(em.TailRecordsRead):
                list(held)
            self.assertGreater(em.cold_read_stats()["rewrites"], 0)
            self.assertEqual(list(self._read(path)[4]), plain(path), "the next read through the reader sees the rewrite")

    def test_no_c_fast_path_sees_a_truncated_view(self):
        with knobs(32 * 1024, 4, roots=[str(self.proj)]):
            path = self.leaf()
            write_transcript(path, 300 * 1024)
            got = self._read(path)[4]
            self.assertTrue(_tail(got))
            self.assertNotIsInstance(got, list, "a list subclass would hand C readers its storage: the window alone")
            with self.assertRaises(TypeError):
                json.dumps(got)                                   # loud, never the window serialised as the whole
            self.assertEqual(len(list(got)), len(plain(path)))
            self.assertEqual(sum(1 for _ in got), len(plain(path)))

    def test_files_outside_the_transcripts_root_and_the_off_switch_stay_whole(self):
        other = self.td / "state" / "timeline"
        other.mkdir(parents=True)
        log = str(other / "messages.jsonl")                      # the postal log's place: walked whole by every parse
        with knobs(32 * 1024, 4, roots=[str(self.proj)]):
            path = self.leaf()
            write_transcript(path, 300 * 1024)
            shutil.copy(path, log)
            self.assertTrue(_tail(self._read(path)[4]), "the transcript is held tail-only")
            self.assertIs(type(self._read(log)[4]), list, "a large file outside the transcripts root stays whole")
            self.assertFalse(em.entry_whole_resident(path), "a fold from record 0 over a tail-only entry reads: not 'resident'")
            self.assertTrue(em.entry_indexed_whole(path), "its offsets start at record 0: the assembly writer's need")
            self.assertTrue(em.entry_whole_resident(log))
        with knobs(32 * 1024, 4, roots=None):
            fresh()
            self.assertIs(type(self._read(path)[4]), list, "the default root is PROJECTS, not this temp directory")
        with knobs(0, 4, roots=[str(self.proj)]):
            fresh()
            self.assertIs(type(self._read(path)[4]), list, "ROMP_RECORD_CACHE_TAIL_MB=0: every entry whole, as before")
        with knobs(32 * 1024, 4, roots=[str(self.proj)]):
            fresh()
            self.assertTrue(_tail(self._read(path)[4]), "on again (the guard: this test reads the rule, not a no-op)")


def _strip_tree(tree):
    t = json.loads(json.dumps(em.plain_tree(tree), default=lambda o: "<unserializable>"))
    t.pop("cutTurn", None)
    for turn in t["turns"]:
        for a in turn["atoms"]:
            a.pop("lazy", None)
    return t


class Assembly(Base):
    """The assembly's whole parse, its document, a fresh process's restore and a fold, over a tail-only leaf."""

    def parse(self, path):
        return em.parse_session(path, rompuuid=SID, name="impl", dir="/TESTDIR", candidate_files=[path],
                                states=None, postal_log=[], now=NOW)

    def test_the_whole_parse_the_document_and_the_restore_over_a_tail_only_leaf_equal_the_rule_off_parse(self):
        path = self.leaf()
        recs = transcript(NOW - 86400, turns=160, compact_every=40)
        Path(path).write_text("".join(json.dumps(r) + "\n" for r in recs))
        with knobs(0, 4, roots=[str(self.proj)]):
            saved = em._CKPT_DIR_FN; em._CKPT_DIR_FN = None
            try:
                ref = _strip_tree(self.parse(path))
            finally:
                em._CKPT_DIR_FN = saved
        fresh()
        with knobs(16 * 1024, 4, roots=[str(self.proj)]):
            tree = self.parse(path)
            ent = em._JSONL_CACHE.get(path)
            self.assertTrue(ent is not None and _tail(ent[4]), "the parse read the leaf through a tail-only entry")
            self.assertEqual(_strip_tree(tree), ref, "the whole parse over the tail-only entry equals the rule-off parse")
            key = (os.path.realpath(path), SID, False)
            ad = em._ASM_CACHE[key]["ad"]
            self.assertIs(type(ad._src[path]), list, "the whole adapter holds its own list: the writer never walks the disk")
            c0 = em.cold_read_stats()["records"]
            self.assertTrue(em.asm_checkpoint_write(path, SID, tree=tree), em.asm_checkpoint_stats())
            self.assertEqual(em.cold_read_stats()["records"], c0, "the document was written without reading before the window")
            more = [{"type": "user", "uuid": "u-more", "parentUuid": recs[-1]["uuid"], "timestamp": "2026-06-11T12:00:00.000Z",
                     "promptSource": "typed", "cwd": "/w/notes-api", "message": {"role": "user", "content": "one more"}}]
            fresh()
            restored = self.parse(path)
            em.hydrate(restored, SID)
            self.assertEqual(_strip_tree(restored), ref, "a fresh process's restore, hydrated, equals the rule-off parse")
            with open(path, "a") as f:
                f.write(json.dumps(more[0]) + "\n")
            fresh()
            folded = self.parse(path)
            em.hydrate(folded, SID)
        with knobs(0, 4, roots=[str(self.proj)]):
            fresh()
            saved = em._CKPT_DIR_FN; em._CKPT_DIR_FN = None
            try:
                ref2 = _strip_tree(self.parse(path))
            finally:
                em._CKPT_DIR_FN = saved
        self.assertEqual(_strip_tree(folded), ref2, "after an append, still equal")


class ChatScrollBack(unittest.TestCase):
    """The chat build's scroll-back pages (the kernel's _chat_history_page) over a tail-only leaf equal the whole build:
    the test_chat_pages harness's check, with the leaf's record entry tail-only throughout."""

    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.saved_state = jd.STATE
        jd._rebind_state(self.td / "state")
        for d in ("states", "goals", "sdk", "checkpoints"):
            (jd.STATE / d).mkdir(parents=True, exist_ok=True)
        em.set_checkpoint_dir(lambda: jd.STATE / "checkpoints")
        self.proj = self.td / "proj"; self.proj.mkdir()
        self.leaf = str(self.proj / (SID + ".jsonl"))
        self.rows = [{"sid": SID, "name": "web", "path": self.leaf, "mtime": NOW, "anchor": SID}]
        self.saved = (km._sessions, km._live_map)
        km._sessions = lambda now, **kw: list(self.rows)
        km._live_map = lambda: {}
        self.fresh()

    def tearDown(self):
        km._sessions, km._live_map = self.saved
        km._live_scope.chat_floor0 = None
        em.set_checkpoint_dir(None)
        jd._rebind_state(self.saved_state)
        self.fresh()
        shutil.rmtree(self.td, ignore_errors=True)

    def fresh(self):
        fresh()
        with km._chat_fold_lock:
            km._chat_fold.clear()
        jd._PARSE_CACHE.clear()
        km._parse_mode.clear()
        km._built_chat.clear() if hasattr(km, "_built_chat") else None
        km._prev_chat_events.clear()
        km._RENDER_FLOOR.clear()
        with km._page_lock:
            km._PAGE_CACHE.clear()
        km._live_scope.chat_floor0 = None
        with em._MAT_LOCK:
            em._MAT_LRU.clear()

    def test_scroll_back_pages_over_a_tail_only_leaf_equal_the_whole_build(self):
        recs = transcript(NOW - 86400, turns=120, compact_every=25)
        Path(self.leaf).write_text("".join(json.dumps(r) + "\n" for r in recs))
        strip = lambda ev: json.loads(json.dumps(ev, default=lambda o: "<unserializable>"))
        with knobs(0, 4, roots=[str(self.proj)]):                 # the reference: the rule off, a fully hydrated build
            self.fresh(); saved = em._CKPT_DIR_FN; em._CKPT_DIR_FN = None
            try:
                m = km.build_session(SID, NOW, {}, floor=0)
            finally:
                em._CKPT_DIR_FN = saved
            heads = strip(m.get("headCards") or [])
            whole = strip(m["events"])[len(heads):]
        with knobs(16 * 1024, 4, roots=[str(self.proj)]):
            self.fresh()
            km.build_session(SID, NOW, {}, floor=0)              # a whole parse over the tail-only entry, then its document
            ent = em._JSONL_CACHE.get(self.leaf)
            self.assertTrue(ent is not None and _tail(ent[4]), "the chat build read the leaf through a tail-only entry")
            self.assertTrue(em.asm_checkpoint_write(self.leaf, SID, tree=km._parse(self.leaf, SID, NOW)), em.asm_checkpoint_stats())
            self.fresh()
            km._live_scope.chat_floor0 = False
            try:
                m2 = km.build_session(SID, NOW, {})
            finally:
                km._live_scope.chat_floor0 = None
            floor = m2["floor"]
            self.assertGreater(floor, 0, "a restored parse renders from the cut")
            tail = strip(m2["events"])
            self.assertEqual(tail, whole[len(whole) - len(tail):], "the floor'd list is the whole build's tail")
            for size in (1, 7, 16):
                with self.subTest(page_turns=size):
                    pages = []
                    for lo in range(0, floor, size):
                        pages += km._chat_history_page(SID, lo, min(lo + size, floor), NOW)
                    self.assertEqual(strip(pages) + tail, whole, "scroll-back pages of %d turns plus the tail equal the whole" % size)
            ent = em._JSONL_CACHE.get(self.leaf)
            if ent is not None:
                self.assertTrue(_tail(ent[4]) or ent[5] > 0, "the leaf's record entry never became whole")


if __name__ == "__main__":
    unittest.main()
