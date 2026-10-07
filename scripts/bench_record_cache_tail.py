#!/usr/bin/env python3
"""bench_record_cache_tail: the record cache holding a large transcript's TAIL only, against holding it whole, on a loop
shaped like the kernel's pusher, over SYNTHETIC transcripts (invented text, placeholder ids; never a real transcript).

The corpus: --files transcripts totalling --total-gb, each a chain of turns of the shape a coding session writes (a typed
prompt, one to six tool rounds whose results are long-tailed in size and carried twice as the CLI does, an occasional
background task and its completion notice, a closing reply, a compaction every couple of hundred turns). Generated once
(deterministic seed) and reused.

Three phases, each its own process so a measurement never inherits another's heap:

    python scripts/bench_record_cache_tail.py --phase gen  --corpus DIR
    python scripts/bench_record_cache_tail.py --phase prep --corpus DIR --repo CHECKOUT --ckpt CKPTDIR
    python scripts/bench_record_cache_tail.py --phase run  --corpus DIR --repo CHECKOUT --ckpt CKPTDIR --label after >> runs.jsonl
    python scripts/bench_record_cache_tail.py --phase summarize runs.jsonl --live-gib 20.2 --live-cache-gb 49.5 --live-rss-gb 74

`prep` is the kernel life before the measured one: a whole parse of every transcript, its assembly document and its fold
documents written (what a settle leaves on disk). `run` is a restart over those documents. Its cycle 0 is the boot: every
transcript parsed (restored from its document), its folds restored, and one whole read of each (the boot's whole readers:
an auto-nudge or interrupt parse that upgrades a restored tail entry). Then --cycles cycles of the pusher's shape:
  - appends to a rotating third of the transcripts (a tool round each cycle, a closing reply and a new prompt every third);
  - the folds over every transcript (the background-task views bgAll and bgRunning through the real fold, scan_bg_tasks_cached,
    and a small session-meta fold), as the jobs pass and every build run them;
  - a chat build near the end of every transcript (parse_session, the bodies of its last two turns hydrated);
  - every fifth cycle (cycles 2, 7, 12, ...) a scroll-back page at the top of one transcript (sixteen turns hydrated);
  - every tenth cycle a whole adapter over one transcript (FileAdapter with no document: the kernel's thread-message and
    anchor reads, and a parse demoted to whole), and, offset by five, a fold with no cursor over one transcript (a refold
    from record 0, as after a drop).
Per cycle it records the wall time, the bytes the reader pulled off disk, the record cache's weight (/perf recordCache.bytes,
the kernel's own estimate of resident bytes), the resident size and its high-water mark (VmRSS, VmHWM), and, where the
checkout has the rule, the reads before a window. --repo picks the checkout whose event model runs: the base for
"before", this branch for "after" (ROMP_RECORD_CACHE_TAIL_MB=0 on this branch is the same as the base for the cache).

Guards: refuses to start with under 20 GiB available, and stops with exit 3 once its own resident size passes --max-rss-gb
(default 12). One process at a time."""
import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time

GIB = 1024 ** 3
MIB = 1024 ** 2
SID_FMT = "bbbbbbbb-4444-4222-8333-%012d"


def _status_kb(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return 0


def _mem_available_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 ** 2)
    return 0.0


def _iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ".000Z"


class Writer:
    """One synthetic transcript, appended a turn or a tool round at a time, its uuid chain and clock carried."""

    def __init__(self, path, fi, rng, blob, t0):
        self.path, self.fi, self.rng, self.blob = path, fi, rng, blob
        self.sid = SID_FMT % fi
        self.n = 0
        self.parent = None
        self.t = t0
        self.pending_bg = []

    def _uid(self):
        self.n += 1
        return "11111111-2222-4333-%04d-%012d" % (self.fi % 10000, self.n)

    def _text(self, mean):
        n = max(8, int(self.rng.lognormvariate(0, 1.0) * mean))
        n = min(n, len(self.blob) - 1)
        i = self.rng.randrange(0, len(self.blob) - n)
        return self.blob[i:i + n]

    def _base(self, typ):
        u = self._uid()
        self.t += self.rng.uniform(0.5, 20)
        r = {"parentUuid": self.parent, "isSidechain": False, "userType": "external", "cwd": "/w/notes-api",
             "sessionId": self.sid, "version": "2.1.0", "gitBranch": "main", "type": typ, "uuid": u, "timestamp": _iso(self.t)}
        self.parent = u
        return r

    def prompt(self):
        r = self._base("user")
        r["promptSource"] = "typed"
        r["message"] = {"role": "user", "content": self._text(300)}
        return [r]

    def tool_round(self):
        out = []
        tid = "toolu_%d_%d" % (self.fi, self.n)
        bg = self.rng.random() < 0.03
        a = self._base("assistant")
        a["requestId"] = "req_%d_%d" % (self.fi, self.n)
        a["message"] = {"id": "msg_%d_%d" % (self.fi, self.n), "type": "message", "role": "assistant", "model": "model-x",
                        "content": [{"type": "tool_use", "id": tid, "name": self.rng.choice(("Bash", "Read", "Edit", "Grep")),
                                     "input": {"command": self._text(120), "description": self._text(30),
                                               **({"run_in_background": True} if bg else {})}}],
                        "stop_reason": "tool_use", "stop_sequence": None,
                        "usage": {"input_tokens": self.rng.randrange(1, 9), "cache_read_input_tokens": self.rng.randrange(0, 200000),
                                  "output_tokens": self.rng.randrange(1, 4000), "service_tier": "standard"}}
        out.append(a)
        body = self._text(2500)
        u = self._base("user")
        u["message"] = {"role": "user", "content": [{"tool_use_id": tid, "type": "tool_result", "content": body, "is_error": False}]}
        u["toolUseResult"] = {"stdout": body, "stderr": "", "interrupted": False, "isImage": False}
        out.append(u)
        if bg:
            self.pending_bg.append(tid)
        if self.pending_bg and self.rng.random() < 0.05:
            done = self.pending_bg.pop(0)
            nt = self._base("user")
            nt["message"] = {"role": "user", "content": "<task-notification>\n<task-id>b%s</task-id>\n<tool-use-id>%s</tool-use-id>\n"
                             "<status>completed</status>\n<summary>done</summary>\n</task-notification>" % (done, done)}
            out.append(nt)
        return out

    def reply(self):
        r = self._base("assistant")
        r["message"] = {"id": "msg_%d_%d" % (self.fi, self.n), "type": "message", "role": "assistant", "model": "model-x",
                        "content": [{"type": "text", "text": self._text(800)}], "stop_reason": "end_turn", "stop_sequence": None,
                        "usage": {"input_tokens": 3, "output_tokens": self.rng.randrange(1, 2000)}}
        return [r]

    def compact(self):
        b = self._base("system")
        b.update({"subtype": "compact_boundary", "parentUuid": None, "logicalParentUuid": b["parentUuid"],
                  "compactMetadata": {"trigger": "auto", "preTokens": 160000, "postTokens": 9000}})
        s = self._base("user")
        s["isCompactSummary"] = True
        s["message"] = {"role": "user", "content": "summary so far: " + self._text(2000)}
        return [b, s]

    def turn(self, k):
        recs = []
        if k and k % 220 == 0:
            recs += self.compact()
        recs += self.prompt()
        for _ in range(self.rng.randint(1, 6)):
            recs += self.tool_round()
        recs += self.reply()
        return recs

    def write(self, recs):
        with open(self.path, "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")


def gen(corpus, files, total_bytes, seed=11):
    os.makedirs(corpus, exist_ok=True)
    have = sorted(f for f in os.listdir(corpus) if f.endswith(".jsonl"))
    if len(have) >= files and sum(os.path.getsize(os.path.join(corpus, f)) for f in have[:files]) >= 0.95 * total_bytes:
        return [os.path.join(corpus, f) for f in have[:files]]
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz      \n.,_/-0123456789"
    blob = "".join(rng.choice(alphabet) for _ in range(1 << 21))
    per = total_bytes // files
    paths = []
    for fi in range(files):
        sid = SID_FMT % fi
        p = os.path.join(corpus, sid + ".jsonl")
        if os.path.exists(p):
            os.unlink(p)
        w = Writer(p, fi, random.Random(seed * 1000 + fi), blob, 1_780_000_000 + fi * 7)
        k = 0
        size = 0
        # sizes vary like the live set's (0.7 to 1.9 GiB there): from 0.5 to 1.5 times the mean
        target = int(per * (0.5 + fi / max(1, files - 1)))
        with open(p, "w") as f:
            while size < target:
                for r in w.turn(k):
                    line = json.dumps(r) + "\n"
                    f.write(line)
                    size += len(line)
                k += 1
        paths.append(p)
        sys.stderr.write("gen %s %.1f MB %d turns\n" % (os.path.basename(p), size / 1e6, k))
    return paths


def load(repo, corpus):
    os.environ.setdefault("XDG_STATE_HOME", tempfile.mkdtemp(prefix="bench-tail-state-"))
    os.environ.pop("ROMP_STATE_DIR", None)
    os.environ["ROMP_KERNEL_NO_OPEN"] = "1"
    os.environ.setdefault("ROMP_RECORD_CACHE_TAIL_ROOTS", os.path.realpath(corpus))   # a checkout without the rule ignores it
    sys.path.insert(0, os.path.join(repo, "tests"))
    from romp_load import load_source
    return load_source("romp_event_model", os.path.join(repo, "bin", "romp-event-model"))


def leaves(corpus, files):
    return sorted(os.path.join(os.path.realpath(corpus), f) for f in os.listdir(corpus) if f.endswith(".jsonl"))[:files]


def parse(em, path):
    sid = os.path.basename(path)[:-6]
    return em.parse_session(path, rompuuid=sid, name="bench", dir="/w/notes-api", candidate_files=[path], states=None,
                            postal_log=[], now=time.time())


META = {}


def meta_fold(em, path, cache):
    def step(st, r):
        st["n"] += 1
        if r.get("cwd"):
            st["cwd"] = r["cwd"]
        if r.get("type") == "assistant":
            m = (r.get("message") or {}).get("model")
            if m:
                st["model"] = m
        return st
    return em.fold_records(cache, path, lambda: {"n": 0, "cwd": None, "model": None}, step, ckpt="benchMeta")


def folds(em, path, caches):
    em.scan_bg_tasks_cached(path, caches["bgAll"], want_all=True, ckpt="bgAll")
    em.scan_bg_tasks_cached(path, caches["bgRunning"], want_all=False, ckpt="bgRunning")
    meta_fold(em, path, caches["meta"])


def prep(em, paths, ckpt):
    os.makedirs(ckpt, exist_ok=True)
    em.set_checkpoint_dir(lambda: ckpt)
    caches = {"bgAll": {}, "bgRunning": {}, "meta": {}}
    for p in paths:
        sid = os.path.basename(p)[:-6]
        tree = parse(em, p)
        folds(em, p, caches)
        ok = em.asm_checkpoint_write(p, sid, tree=tree)
        em.checkpoint_write(p)
        sys.stderr.write("prep %s doc=%s rss=%.2f GiB\n" % (os.path.basename(p), ok, _status_kb("VmRSS") / (1024 ** 2)))
        em.evict_document(p)


def run(em, paths, ckpt, cycles, label, max_rss_gb, seed=5):
    """The measured restart (see the module docstring). The corpus is put back as it was at the end (each file truncated to
    its starting size and given its starting mtime), so every run starts from the same bytes and the prep's documents
    still verify."""
    orig = {p: (os.path.getsize(p), os.stat(p).st_mtime) for p in paths}
    try:
        _run(em, paths, ckpt, cycles, label, max_rss_gb, seed)
    finally:
        for p, (size, mtime) in orig.items():
            with open(p, "r+b") as f:
                f.truncate(size)
            os.utime(p, (mtime, mtime))


def _run(em, paths, ckpt, cycles, label, max_rss_gb, seed):
    em.set_checkpoint_dir(lambda: ckpt)
    caches = {"bgAll": {}, "bgRunning": {}, "meta": {}}
    for name, c in caches.items():
        em.name_fold_cache(c, {"meta": "benchMeta"}.get(name, name))
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz      \n.,_/-0123456789"
    blob = "".join(rng.choice(alphabet) for _ in range(1 << 20))
    writers = {}
    for fi, p in enumerate(paths):                      # carry each file's chain on from its last record (a bounded tail read)
        with open(p, "rb") as f:
            f.seek(max(0, os.path.getsize(p) - 4 * MIB))
            last = json.loads(f.read().splitlines()[-1])
        w = Writer(p, fi, random.Random(seed * 100 + fi), blob, (em.parse_z(last["timestamp"]) or time.time()) + 60)
        w.parent, w.n = last["uuid"], 10 ** 7
        writers[p] = w
    has_cold = hasattr(em, "cold_read_stats")
    rows = []

    def guard():
        rss = _status_kb("VmRSS") / (1024 ** 2)
        if rss > max_rss_gb:
            sys.stderr.write("stopping: resident %.2f GiB over the %.1f GiB guard\n" % (rss, max_rss_gb))
            print(json.dumps({"label": label, "aborted": "rss", "rssGiB": rss, "rows": rows}))
            raise SystemExit(3)

    def sample(c, t0, r0, c0, what):
        st = em.record_cache_stats()
        cold = em.cold_read_stats() if has_cold else {"passes": 0, "records": 0, "bytes": 0}
        row = {"cycle": c, "what": what, "wallS": round(time.monotonic() - t0, 3), "readBytes": em.read_bytes_total() - r0,
               "cacheBytes": st["bytes"], "entries": st["entries"], "tailOnly": (st.get("tailOnly") or {}).get("entries", 0),
               "rssKiB": _status_kb("VmRSS"), "hwmKiB": _status_kb("VmHWM"),
               "coldPasses": cold["passes"] - c0["passes"], "coldBytes": cold["bytes"] - c0["bytes"]}
        rows.append(row)
        sys.stderr.write("%s c%-3d %-30s %6.2fs read %7.1f MB cache %6.2f GB rss %5.2f GiB cold %d passes %.1f MB\n" % (
            label, c, what, row["wallS"], row["readBytes"] / 1e6, row["cacheBytes"] / 1e9, row["rssKiB"] / (1024 ** 2),
            row["coldPasses"], row["coldBytes"] / 1e6))
        return row

    def cold0():
        return em.cold_read_stats() if has_cold else {"passes": 0, "records": 0, "bytes": 0}

    # cycle 0: the boot over the documents
    t0, r0, c0 = time.monotonic(), em.read_bytes_total(), cold0()
    for p in paths:
        tree = parse(em, p)
        em.hydrate([a for t in tree["turns"][-2:] for a in t["atoms"]])
        folds(em, p, caches)
        em._read_jsonl_incremental(p)                   # the boot's whole readers (an interrupt or auto-nudge parse's upgrade)
        guard()
    sample(0, t0, r0, c0, "boot")
    n = len(paths)
    for c in range(1, cycles + 1):
        t0, r0, c0 = time.monotonic(), em.read_bytes_total(), cold0()
        what = []
        for i, p in enumerate(paths):
            if (i + c) % 3 == 0:
                w = writers[p]
                recs = w.tool_round()
                if c % 3 == 0:
                    recs += w.reply() + w.prompt()
                w.write(recs)
        for p in paths:
            folds(em, p, caches)
            tree = parse(em, p)
            em.hydrate([a for t in tree["turns"][-2:] for a in t["atoms"]])
        guard()
        if c % 5 == 2:
            p = paths[(c // 5) % n]
            tree = parse(em, p)
            em.hydrate([a for t in tree["turns"][:16] for a in t["atoms"]])
            what.append("scroll-back")
        if c % 10 == 0:
            p = paths[(c // 10) % n]
            ad = em.FileAdapter([p], p)
            del ad
            what.append("whole adapter")
            guard()
        if c % 10 == 5:
            p = paths[(c // 10 + 3) % n]
            st = em.fold_records({}, p, int, lambda s, r: s + 1)
            what.append("refold from 0")
        sample(c, t0, r0, c0, "+".join(what) or "steady")
    disk = sum(os.path.getsize(p) for p in paths)
    out = {"label": label, "files": n, "diskBytes": disk, "cycles": cycles, "rows": rows,
           "recordCount": sum(len(em._JSONL_CACHE[p][4]) for p in paths if p in em._JSONL_CACHE),
           "tailBytes": getattr(em, "_TAIL_BYTES", None), "tailRecords": getattr(em, "_TAIL_RECORDS", None)}
    print(json.dumps(out))


def summarize(files, live_gib, live_cache_gb, live_rss_gb):
    runs = {}
    for fn in files:
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if line.startswith("{"):
                    d = json.loads(line)
                    runs[d["label"]] = d
    def agg(d):
        rows = d["rows"]
        boot, steady = rows[0], rows[1:]
        plain = [r for r in steady if r["what"] == "steady"]
        return {
            "disk GiB": d["diskBytes"] / GIB,
            "boot s": boot["wallS"], "boot read GB": boot["readBytes"] / 1e9,
            "steady cycle s (median)": statistics.median(r["wallS"] for r in plain) if plain else float("nan"),
            "steady cycle s (max)": max(r["wallS"] for r in plain) if plain else float("nan"),
            "steady read MB/cycle (median)": statistics.median(r["readBytes"] for r in plain) / 1e6 if plain else float("nan"),
            "scroll-back cycle s": statistics.median([r["wallS"] for r in steady if "scroll-back" in r["what"]] or [float("nan")]),
            "whole-adapter cycle s": statistics.median([r["wallS"] for r in steady if "whole adapter" in r["what"]] or [float("nan")]),
            "refold-from-0 cycle s": statistics.median([r["wallS"] for r in steady if "refold" in r["what"]] or [float("nan")]),
            "refold-from-0 read MB": statistics.median([r["readBytes"] / 1e6 for r in steady if "refold" in r["what"]] or [float("nan")]),
            "cache GB (end)": rows[-1]["cacheBytes"] / 1e9,
            "RSS GiB (end)": rows[-1]["rssKiB"] / (1024 ** 2),
            "peak RSS GiB": max(r["hwmKiB"] for r in rows) / (1024 ** 2),
            "tail-only entries (end)": rows[-1]["tailOnly"],
            "records": d.get("recordCount"),
        }
    labels = [l for l in ("before", "after") if l in runs] + [l for l in runs if l not in ("before", "after")]
    table = {l: agg(runs[l]) for l in labels}
    keys = list(next(iter(table.values())).keys())
    print("| measure | " + " | ".join(labels) + " |")
    print("|---|" + "---|" * len(labels))
    for k in keys:
        cells = []
        for l in labels:
            v = table[l][k]
            cells.append(("%.3f" % v) if isinstance(v, float) else str(v))
        print("| %s | %s |" % (k, " | ".join(cells)))
    if "after" in runs and "before" in runs:
        b, a = table["before"], table["after"]
        disk_gib = b["disk GiB"]
        recs = a["records"] or 0
        per_rec = (runs["after"]["diskBytes"] / recs) if recs else 5000
        tb = (runs["after"].get("tailBytes") or 32 * MIB)
        print()
        print("EXTRAPOLATION (not a measurement) to %.1f GiB of live transcripts in 23 files:" % live_gib)
        before_w = live_gib * GIB * 3.0
        live_records = live_gib * GIB / per_rec
        after_w = 23 * tb * 3.0 + live_records * 20          # offsets (16 bytes) and a CRC (4) a record
        print("  cache weight whole (3 bytes a file byte): %.1f GB (tonight's measured cache: %.1f GB in 59 entries)" % (before_w / 1e9, live_cache_gb))
        print("  cache weight tail-only: 23 windows of %d MiB x 3 + %.1f M records x 20 B of index = %.2f GB" % (tb // MIB, live_records / 1e6, after_w / 1e9))
        print("  bench ratio after/before (cache, end of run): %.4f -> %.2f GB of tonight's %.1f GB" % (
            a["cache GB (end)"] / b["cache GB (end)"], live_cache_gb * a["cache GB (end)"] / b["cache GB (end)"], live_cache_gb))
        print("  resident: tonight %.1f GB with a %.1f GB cache; the same heap with a tail-only cache: about %.1f GB "
              "(the other %.1f GB of heap unchanged, and growing separately)" % (
                  live_rss_gb, live_cache_gb, live_rss_gb - live_cache_gb + after_w / 1e9, live_rss_gb - live_cache_gb))
        print("  bench resident at the end: before %.2f GiB, after %.2f GiB over %.2f GiB on disk (peak %.2f vs %.2f GiB)" % (
            b["RSS GiB (end)"], a["RSS GiB (end)"], disk_gib, b["peak RSS GiB"], a["peak RSS GiB"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", required=True, choices=("gen", "prep", "run", "summarize"))
    ap.add_argument("--corpus", default=os.path.join(tempfile.gettempdir(), "bench-record-cache-tail"))
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    ap.add_argument("--ckpt")
    ap.add_argument("--files", type=int, default=12)
    ap.add_argument("--total-gb", type=float, default=2.4)
    ap.add_argument("--cycles", type=int, default=30)
    ap.add_argument("--label", default="run")
    ap.add_argument("--max-rss-gb", type=float, default=12.0)
    ap.add_argument("--live-gib", type=float, default=20.2)
    ap.add_argument("--live-cache-gb", type=float, default=49.5)
    ap.add_argument("--live-rss-gb", type=float, default=74.0)
    ap.add_argument("inputs", nargs="*")
    a = ap.parse_args()
    if a.phase == "summarize":
        return summarize(a.inputs, a.live_gib, a.live_cache_gb, a.live_rss_gb)
    if a.phase == "gen":
        gen(a.corpus, a.files, int(a.total_gb * 1e9))
        return
    if _mem_available_gib() < 20:
        sys.exit("refusing: under 20 GiB available (%.1f)" % _mem_available_gib())
    em = load(a.repo, a.corpus)
    paths = leaves(a.corpus, a.files)
    if a.phase == "prep":
        prep(em, paths, a.ckpt)
    else:
        run(em, paths, a.ckpt, a.cycles, a.label, a.max_rss_gb)


if __name__ == "__main__":
    main()
