"""Differential test of the assembly's roads over a tail-only leaf (2026-10-08, after two reviews in a row found wrong trees
in hand-picked cases): random appends mixing every record kind the fold's gates and the restore read, over a leaf with an
assembly document, and the hydrated tree after the road the code picks (fold, restore or whole parse), and again after a
restart (the boot's restore), must equal a cold whole parse (the rule off, no checkpoint directory).

The leaf's pre-cut part carries what the gates and the restore key on: prompt ids on every prompt, a typed slash command's raw
twin (its wrapper in the tail re-classifies it), a Skill payload record (a Skill tool_use in the tail links it), and a
compaction whose stitch target was never written (a record carrying it in the tail rebinds it). The appends draw from: prompts
(a fresh, a tail-reused or a pre-cut prompt id; a current, stale or missing stamp), replies, command wrappers wearing the
pre-cut twin's id or a fresh one, Skill tool_uses on the linked id or a fresh one, compactions (a boundary and its summary, the
summary a second earlier, the summary in the next append, a stale boundary), summaries off any boundary, verbatim duplicates,
the dangling target, and swaps of adjacent records.

ROMP_FUZZ_CASES sets the case count (default 120, under a minute); ROMP_FUZZ_SEED the seed (default 1). A longer sweep, run
locally: ROMP_FUZZ_CASES=2000 ROMP_FUZZ_SEED=7. Synthetic only: invented text, placeholder ids."""
import collections
import json
import os
import random
import shutil
import sys
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import test_record_cache_tail_r2 as R                   # noqa: E402  one kernel copy: R2Base, knobs, fresh
from test_asm_checkpoint_served import transcript, iso  # noqa: E402

T, em, SID, NOW, WINDOW = R.T, R.em, R.SID, R.NOW, R.WINDOW
CASES = int(os.environ.get("ROMP_FUZZ_CASES", "120"))
SEED = int(os.environ.get("ROMP_FUZZ_SEED", "1"))
WRAP = "<command-name>/review</command-name>\n<command-message>review</command-message>\n<command-args></command-args>"


class Gen:
    """One case's appends, chained on the leaf's current tip."""

    def __init__(self, rnd, parent, t, precut_t):
        self.rnd, self.parent, self.t, self.precut_t = rnd, parent, t, precut_t
        self.n, self.written, self.pids, self.pending = 0, [], [], []

    def _id(self, p):
        self.n += 1
        return "fz-%s%d" % (p, self.n)                  # never a base record's uuid (u1, a1, b40, s40, ...)

    def _stamp(self, kinds):
        x = self.rnd.random()
        if x < 0.04:
            kinds.append("stale")
            return iso(self.precut_t + self.rnd.randrange(0, 600))
        if x < 0.05:
            kinds.append("nostamp")
            return None
        self.t += self.rnd.randrange(5, 60)
        return iso(self.t)

    def _rec(self, r, ts):
        if ts is not None:
            r["timestamp"] = ts
        self.parent = r["uuid"]
        return r

    def append(self):
        kinds, out = [], list(self.pending)
        if self.pending:
            kinds.append("late-summary")
        self.pending = []
        for _ in range(self.rnd.randint(1, 5)):
            k = self.rnd.choices(("prompt", "reply", "compact", "wrapper", "skill", "orphan", "dup", "ghost"),
                                 weights=(30, 30, 18, 5, 5, 4, 5, 3))[0]   # mostly benign: the roads a hazard picks need company
            if k == "prompt":
                pid = self.rnd.choices(("fresh", "tail", "pre"), weights=(6, 3, 1))[0]
                pv = {"fresh": self._id("pf"), "tail": self.rnd.choice(self.pids) if self.pids else self._id("pf"),
                      "pre": "p%d" % self.rnd.randrange(1, 40)}[pid]
                self.pids.append(pv)
                kinds.append("prompt:" + pid)
                ts = self._stamp(kinds)
                out.append(self._rec({"type": "user", "uuid": self._id("u"), "parentUuid": self.parent, "promptSource": "typed",
                                      "promptId": pv, "cwd": "/w/notes-api",
                                      "message": {"role": "user", "content": "step %d" % self.n}}, ts))
            elif k == "reply":
                kinds.append("reply")
                ts = self._stamp(kinds)
                out.append(self._rec({"type": "assistant", "uuid": self._id("a"), "parentUuid": self.parent, "cwd": "/w/notes-api",
                                      "message": {"role": "assistant", "content": [{"type": "text", "text": "ok %d " % self.n * 5}],
                                                  "stop_reason": "end_turn"}}, ts))
            elif k == "wrapper":
                pre = self.rnd.random() < 0.6
                kinds.append("wrapper:" + ("pre" if pre else "fresh"))
                self.t += 3
                out.append(self._rec({"type": "user", "uuid": self._id("w"), "parentUuid": self.parent, "cwd": "/w/notes-api",
                                      "promptId": "pid-pre" if pre else self._id("pw"),
                                      "message": {"role": "user", "content": WRAP}}, iso(self.t)))
            elif k == "skill":
                pre = self.rnd.random() < 0.6
                kinds.append("skill:" + ("pre" if pre else "fresh"))
                self.t += 3
                out.append(self._rec({"type": "assistant", "uuid": self._id("k"), "parentUuid": self.parent, "cwd": "/w/notes-api",
                                      "message": {"role": "assistant", "content": [
                                          {"type": "tool_use", "id": "toolu_pre_sk" if pre else self._id("toolu_"), "name": "Skill",
                                           "input": {"skill": "deploy"}}], "stop_reason": "tool_use"}}, iso(self.t)))
            elif k == "compact":
                v = self.rnd.choice(("pair", "pair", "early-summary", "split", "stale-boundary", "bare"))
                kinds.append("compact:" + v)
                bu, su = self._id("b"), self._id("s")
                bt = (self.precut_t + 300) if v == "stale-boundary" else (self.t + 5)
                self.t = max(self.t, bt) + 2
                b = {"type": "system", "subtype": "compact_boundary", "uuid": bu, "parentUuid": None, "logicalParentUuid": self.parent,
                     "timestamp": iso(bt), "compactMetadata": {"trigger": "auto", "preTokens": 160000, "postTokens": 9000}}
                s = {"type": "user", "uuid": su, "parentUuid": bu, "isCompactSummary": True,
                     "timestamp": iso(bt - 1 if v == "early-summary" else bt + 1),
                     "message": {"role": "user", "content": "summary so far: %d" % self.n}}
                out.append(b)
                self.parent = bu
                if v == "split":
                    self.pending = [s]
                    self.parent = su
                elif v != "bare":
                    out.append(s)
                    self.parent = su
            elif k == "orphan":
                kinds.append("orphan-summary")
                self.t += 2
                out.append(self._rec({"type": "user", "uuid": self._id("o"), "parentUuid": self.parent, "isCompactSummary": True,
                                      "message": {"role": "user", "content": "summary so far: orphan %d" % self.n}}, iso(self.t)))
            elif k == "dup" and self.written:
                kinds.append("dup")
                out.append(dict(self.written[-1]))
            elif k == "ghost" and not any(r.get("uuid") == "ghost40" for r in self.written + out):
                kinds.append("ghost")
                self.t += 3
                out.append(self._rec({"type": "user", "uuid": "ghost40", "parentUuid": self.parent, "promptSource": "typed",
                                      "cwd": "/w/notes-api", "message": {"role": "user", "content": "the stitch target"}},
                                     iso(self.t)))
        if len(out) > 2 and self.rnd.random() < 0.15:
            i = self.rnd.randrange(0, len(out) - 1)
            out[i], out[i + 1] = out[i + 1], out[i]
            kinds.append("swap")
        self.written += [r for r in out if r.get("uuid")]
        return out, kinds


class DifferentialRoads(R.R2Base):
    TURNS = 80

    def setUp(self):
        super().setUp()
        for r in self.recs:                               # the pre-cut features the gates and the restore key on
            if r.get("type") == "user" and str(r.get("uuid", "")).startswith("u"):
                r["promptId"] = "p" + r["uuid"][1:]
            if r.get("uuid") == "u10":
                r["promptId"] = "pid-pre"
                r["message"]["content"] = "/review"       # a typed slash command's raw twin
            if r.get("uuid") == "u12":
                r["sourceToolUseID"] = "toolu_pre_sk"     # a Skill payload record
                r["message"]["content"] = "instructions for the deploy skill"
            if r.get("uuid") == "b40":
                real = r["logicalParentUuid"]
                r["logicalParentUuid"] = "ghost40"        # a stitch target never written, repaired through preservedSegment
                r["compactMetadata"]["preservedSegment"] = {"tailUuid": real, "anchorUuid": real, "headUuid": real}
        Path(self.path).write_text("".join(json.dumps(x) + "\n" for x in self.recs))

    def _ref(self):
        with T.knobs(0, 4, roots=[str(self.proj)]):
            self._reset()
            saved = em._CKPT_DIR_FN; em._CKPT_DIR_FN = None
            try:
                return T._strip_tree(self.parse())
            finally:
                em._CKPT_DIR_FN = saved

    def _parse_measured(self):
        s0 = dict(em._ASM_STATS)
        tree = self.parse()
        d = {k: v - s0.get(k, 0) for k, v in em._ASM_STATS.items() if v != s0.get(k, 0)}
        em.hydrate(tree, SID)
        road = "whole" if d.get("full") else "restore" if d.get("restore") else "fold" if d.get("fold") else "serve"
        why = sorted(k[2:] for k in d if k.startswith("g:")) + sorted(k for k in d if k.startswith("restore:") and k.endswith("Refused"))
        return road + ("(%s)" % ",".join(why) if why else ""), T._strip_tree(tree)

    def run_cases(self, n, seed):
        rnd = random.Random(seed)
        ck = self.td / "checkpoints"
        with T.knobs(WINDOW, 4, roots=[str(self.proj)]):
            tree = self.parse()
            self.assertTrue(em.asm_checkpoint_write(self.path, SID, tree=tree), em.asm_checkpoint_stats())
            del tree
        base_size, base_mtime = os.path.getsize(self.path), os.stat(self.path).st_mtime
        docs = {p.name: p.read_bytes() for p in ck.iterdir()}
        precut_t = NOW - 86400 + 5 * 60
        fails, roads = collections.Counter(), collections.Counter()
        examples = {}
        for c in range(n):
            with open(self.path, "r+b") as f:             # the leaf and the document as they were: every case starts restored
                f.truncate(base_size)
            os.utime(self.path, (base_mtime, base_mtime))
            for p in list(ck.iterdir()):
                if p.name not in docs:
                    p.unlink()
            for name, data in docs.items():
                (ck / name).write_bytes(data)
            self._reset()
            for m in ("_ASM_CHAIN_REFUSED_PATHS", "_ASM_CKPT_REFUSED"):
                getattr(em, m).clear()                    # a refusal of another case's tail is no fact about this one
            g = Gen(rnd, self.parent, self.t, precut_t)
            sig = []
            path_roads = []
            with T.knobs(WINDOW, 4, roots=[str(self.proj)]):
                path_roads.append(self._parse_measured()[0])   # the boot's restore
                for step in range(2):
                    recs, kinds = g.append()
                    with open(self.path, "a") as f:
                        f.write("".join(json.dumps(r) + "\n" for r in recs))
                    sig.append(tuple(sorted(set(kinds))))
                    road, got = self._parse_measured()
                    path_roads.append(road)
                self._reset()
                broad, bgot = self._parse_measured()      # a restart over the same tail
            ref = self._ref()
            roads["/".join(path_roads)] += 1
            for variant, r_, t_ in (("append", road, got), ("boot", broad, bgot)):
                if t_ != ref:
                    key = (variant, r_, sig[0], sig[1])
                    fails[key] += 1
                    examples.setdefault(key, c)
        return fails, roads, examples

    HAZARDS = ("stale", "nostamp", "wrapper:pre", "skill:pre", "orphan-summary", "dup", "ghost", "swap", "prompt:pre",
               "compact:early-summary", "compact:split", "compact:stale-boundary", "late-summary")

    def classes(self, fails):
        """Failing cases grouped by variant and the hazards present in the case's appends (a distinct failing shape)."""
        out = collections.Counter()
        for (variant, _road, a, b), v in fails.items():
            hz = tuple(sorted({k for k in a + b if k in self.HAZARDS})) or ("none",)
            out[(variant,) + hz] += v
        return out

    def test_random_appends_give_the_cold_parse(self):
        fails, roads, examples = self.run_cases(CASES, SEED)
        cls = self.classes(fails)
        sys.stderr.write("fuzz classes: %d failing cases, %d shapes (variant + hazards): %s\n" % (
            sum(fails.values()), len(cls), "; ".join("%s x%d" % ("+".join(k), v) for k, v in cls.most_common())))
        lines = ["%s road=%s first=%s second=%s x%d (case %d)" % (k[0], k[1], k[2], k[3], v, examples[k])
                 for k, v in sorted(fails.items(), key=lambda kv: -kv[1])]
        sys.stderr.write("fuzz: %d cases, seed %d, roads %r, %d failing cases in %d shapes\n%s\n" % (
            CASES, SEED, dict(roads), sum(fails.values()), len(fails), "\n".join(lines[:40])))
        self.assertEqual(sum(fails.values()), 0, "\n".join(lines[:40]))


if __name__ == "__main__":
    unittest.main()
