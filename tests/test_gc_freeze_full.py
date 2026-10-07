#!/usr/bin/env python3
"""The gc-freeze controller's FULL-COLLECTION freeze (2026-10-07, kernel/gc_freeze.py).

The load fold-in keys on the record cache's inserts, so once the cache's decoded records were untracked the heap that
grows outside the cache was frozen only at start-up, and every organic full collection walked it (on the live kernel about
67 an hour, 5.9 s on average, 11% of wall time). The controller now freezes from the collector's own stop callback after an
organic full collection that paused at least `full_freeze_ms`, raises the third collection threshold to `full_t2`, and
owes a backstop reclaim once full-collection pauses reach `full_backstop_ratio` times the largest walk. These tests pin: the trigger (generation, pause, organic only),
the backstop, the safe knob parses and the OFF switch (no callback, thresholds untouched), that a freeze at the stop
callback pins no garbage while a cycle formed later among frozen objects waits for the backstop, and the kernel's wiring.
Every fixture is synthetic. Real-collector tests unfreeze and restore the thresholds in tearDown.
"""
import gc
import os
import re
import sys
import tempfile
import unittest
import weakref
from pathlib import Path

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "bin")
GC_FREEZE = os.path.join(ROOT, "kernel", "gc_freeze.py")

sys.path.insert(0, os.path.join(ROOT, "tests"))
from romp_load import load_source   # brought in before the state preamble (the mkdtemp hook and TMPDIR redirect ride it)

os.environ["XDG_STATE_HOME"] = tempfile.mkdtemp()
os.environ.pop("ROMP_STATE_DIR", None)
os.makedirs(os.path.join(os.environ["XDG_STATE_HOME"], "romp"), exist_ok=True)
Path(os.environ["XDG_STATE_HOME"], "romp", "session-hosts").write_text("off")
os.environ["ROMP_POSTAL_CLIENT_ONLY"] = "1"

gf = load_source("romp_gc_freeze_full", GC_FREEZE)


class FakeGc:
    def __init__(self):
        self.calls = []
        self.callbacks = []
        self.threshold = (2000, 10, 10)
    def collect(self, *a):
        self.calls.append("collect"); return 0
    def freeze(self):
        self.calls.append("freeze")
    def unfreeze(self):
        self.calls.append("unfreeze")
    def get_threshold(self):
        return self.threshold
    def set_threshold(self, *t):
        self.threshold = tuple(t)


class Clock:
    def __init__(self):
        self.t = 0.0
    def __call__(self):
        return self.t


def _collection(c, clock, generation, ms):
    """Drive one collection through the controller's callback: start, `ms` of pause, stop."""
    c.gc_callback("start", {"generation": generation})
    clock.t += ms / 1000.0
    c.gc_callback("stop", {"generation": generation, "collected": 0, "uncollectable": 0})


class Trigger(unittest.TestCase):
    def _c(self, **kw):
        fake, clock = FakeGc(), Clock()
        kw.setdefault("full_freeze_ms", 250.0)
        return gf.GcFreeze(enabled=True, gc=fake, clock=clock, **kw), fake, clock

    def test_an_expensive_organic_full_collection_freezes_at_its_stop(self):
        c, fake, clock = self._c()
        _collection(c, clock, 2, 300.0)
        self.assertEqual(fake.calls, ["freeze"], "one freeze, no collect: the collector just walked everything")
        self.assertEqual((c.full_freezes, c.frozen, round(c.last_full_ms)), (1, True, 300))

    def test_a_cheap_full_collection_and_the_young_generations_do_not_freeze(self):
        c, fake, clock = self._c()
        _collection(c, clock, 2, 249.0)
        _collection(c, clock, 1, 900.0)
        _collection(c, clock, 0, 900.0)
        self.assertEqual(fake.calls, [], "under the threshold, or not a full collection: no freeze")
        self.assertEqual(c.full_freezes, 0)

    def test_the_controllers_own_collections_never_trigger_it(self):
        c, fake, clock = self._c()
        def slow_collect(*a):
            fake.calls.append("collect")
            _collection(c, clock, 2, 5000.0)      # the reconcile's own collection, as slow as you like
            return 0
        fake.collect = slow_collect
        c.tick(1)                                 # the initial freeze: one collect, one freeze of its own
        self.assertEqual(fake.calls, ["collect", "freeze"])
        self.assertEqual(c.full_freezes, 0, "a collection _run issued re-freezes itself; the callback stands down")

    def test_the_backstop_is_owed_when_full_pauses_reach_the_ratio_of_the_largest_walk(self):
        c, fake, clock = self._c(full_backstop_ratio=3)
        c.tick(1)                                 # initial
        _collection(c, clock, 2, 1000.0)          # a whole-heap walk: freezes, and is the reference (1000 ms)
        _collection(c, clock, 2, 300.0)
        _collection(c, clock, 2, 300.0)           # 1600 ms of full pauses since: under 3 x 1000
        self.assertIsNone(c.tick(1), "under the ratio: nothing owed")
        for _ in range(5):
            _collection(c, clock, 2, 300.0)       # 3100 ms: at 3 x the largest walk
        fake.calls.clear()
        self.assertEqual(c.tick(1), "backstop", "the cheap collections' own spending has paid for a backstop's walk")
        self.assertEqual(fake.calls, ["unfreeze", "collect", "freeze"])
        p = c.perf()
        self.assertEqual((p["fullSinceReclaim"], p["fullMsSinceReclaim"]), (0, 0.0), "and the sums start over")
        self.assertIsNone(c.tick(1))

    def test_no_backstop_is_owed_while_no_full_freeze_has_run_since_the_last_reclaim(self):
        c, fake, clock = self._c(full_backstop_ratio=1)
        c.tick(1)
        for _ in range(10):
            _collection(c, clock, 2, 200.0)       # cheap full collections only: nothing frozen by the callback, nothing pinned
        self.assertIsNone(c.tick(1))
        self.assertEqual(c.full_freezes, 0)

    def test_off_means_no_callback_no_threshold_change_and_no_freeze(self):
        for kw in ({"full_freeze_ms": None}, {"enabled": False}):
            fake, clock = FakeGc(), Clock()
            c = gf.GcFreeze(gc=fake, clock=clock, full_t2=1000, **dict({"enabled": True, "full_freeze_ms": 250.0}, **kw))
            self.assertFalse(c.install_full_freeze(), kw)
            self.assertEqual(fake.callbacks, [], kw)
            self.assertEqual(fake.threshold, (2000, 10, 10), "today's thresholds, untouched: %r" % (kw,))
            _collection(c, clock, 2, 10000.0)
            self.assertEqual(fake.calls, [], kw)

    def test_install_hooks_once_raises_the_third_threshold_and_remove_restores_it(self):
        c, fake, clock = self._c(full_t2=1000)
        self.assertTrue(c.install_full_freeze())
        self.assertTrue(c.install_full_freeze())
        self.assertEqual(fake.callbacks, [c.gc_callback], "one entry however often it is installed")
        self.assertEqual(fake.threshold, (2000, 10, 1000), "the first two thresholds kept, the third raised")
        c.remove_full_freeze()
        self.assertEqual((fake.callbacks, fake.threshold), ([], (2000, 10, 10)))

    def test_a_failure_inside_the_callback_is_counted_never_raised(self):
        c, fake, clock = self._c()
        def boom():
            raise RuntimeError("synthetic")
        fake.freeze = boom
        _collection(c, clock, 2, 999.0)           # must not raise into the collector
        self.assertEqual(c.callback_errors, 1)

    def test_perf_reports_the_full_freeze(self):
        c, fake, clock = self._c(full_t2=1000)
        _collection(c, clock, 2, 300.0)
        p = c.perf()
        self.assertEqual((p["fullFreezeMs"], p["fullFreezes"], p["fullT2"], p["fullSinceReclaim"]), (250.0, 1, 1000, 1))


class Knobs(unittest.TestCase):
    def test_full_freeze_ms(self):
        f = gf.full_freeze_ms_from_env
        self.assertEqual(f({}), (gf.DEFAULT_FULL_FREEZE_MS, None))
        self.assertEqual(f({"ROMP_GC_FREEZE_FULL_MS": " "}), (gf.DEFAULT_FULL_FREEZE_MS, None))
        for off in ("off", "OFF", "0", "false"):
            self.assertEqual(f({"ROMP_GC_FREEZE_FULL_MS": off}), (None, None), off)
        self.assertEqual(f({"ROMP_GC_FREEZE_FULL_MS": "750"}), (750.0, None))
        for bad in ("lots", "-5", "nan", "inf"):
            self.assertEqual(f({"ROMP_GC_FREEZE_FULL_MS": bad}), (gf.DEFAULT_FULL_FREEZE_MS, bad), bad)

    def test_full_t2(self):
        f = gf.full_t2_from_env
        self.assertEqual(f({}), (gf.DEFAULT_FULL_T2, None))
        for off in ("off", "0", "false"):
            self.assertEqual(f({"ROMP_GC_FREEZE_FULL_T2": off}), (None, None), off)
        self.assertEqual(f({"ROMP_GC_FREEZE_FULL_T2": "250"}), (250, None))
        for bad in ("x", "-1", "2.5"):
            self.assertEqual(f({"ROMP_GC_FREEZE_FULL_T2": bad}), (gf.DEFAULT_FULL_T2, bad), bad)

    def test_the_defaults_are_on(self):
        self.assertGreater(gf.DEFAULT_FULL_FREEZE_MS, 0)
        self.assertGreater(gf.DEFAULT_FULL_T2, 10, "above CPython's default third threshold, or it changes nothing")


class Cyclic:
    pass


class RealCollector(unittest.TestCase):
    def setUp(self):
        self._thr = gc.get_threshold()

    def tearDown(self):
        gc.unfreeze()
        gc.set_threshold(*self._thr)
        gc.collect()

    def _controller(self, **kw):
        c = gf.GcFreeze(enabled=True, gc=gc, full_freeze_ms=1e-6, **kw)   # any full collection counts as expensive here
        self.assertTrue(c.install_full_freeze())
        self.addCleanup(c.remove_full_freeze)
        return c

    @staticmethod
    def _cycle():
        a = Cyclic(); b = Cyclic(); a.o = b; b.o = a
        return weakref.ref(a)

    def test_the_stop_freeze_pins_no_garbage_and_takes_the_survivors_out_of_the_walk(self):
        gc.unfreeze(); gc.collect()
        frozen0 = gc.get_freeze_count()
        c = self._controller()
        heap = [{"k": [i]} for i in range(20000)]             # long-lived, tracked
        dead = self._cycle()                                  # garbage when the collection runs
        gc.collect()                                          # an organic full collection: the callback freezes at its stop
        self.assertEqual(c.full_freezes, 1)
        self.assertIsNone(dead(), "the cycle that was garbage at the collection is freed, never pinned")
        self.assertGreaterEqual(gc.get_freeze_count() - frozen0, 40000, "the heap's dicts and lists are frozen")
        self.assertFalse(any(o is heap for o in gc.get_objects(2)), "and gone from the oldest generation's walk")

    def test_a_cycle_formed_later_among_frozen_objects_waits_for_the_backstop(self):
        gc.unfreeze(); gc.collect()
        c = self._controller(full_backstop_ratio=1)
        c.tick(1)                                             # the controller's initial freeze
        a = Cyclic(); b = Cyclic()
        gc.collect()                                          # organic: a and b frozen at its stop
        a.o = b; b.o = a
        w = weakref.ref(a)
        del a, b                                              # now cyclic garbage, but frozen
        c.remove_full_freeze()                                # so the check's own collection freezes nothing new
        gc.collect()
        self.assertIsNotNone(w(), "a full collection never walks the frozen set: the cycle is pinned")
        self.assertEqual(c.tick(1), "backstop", "the full freeze owed a backstop")
        self.assertIsNone(w(), "the backstop's unfreeze-and-collect frees it")

    def test_off_leaves_the_real_collector_alone(self):
        before = gc.get_threshold()
        c = gf.GcFreeze(enabled=True, gc=gc, full_freeze_ms=None, full_t2=1000)
        self.assertFalse(c.install_full_freeze())
        self.assertNotIn(c.gc_callback, gc.callbacks)
        self.assertEqual(gc.get_threshold(), before)


class KernelWiring(unittest.TestCase):
    NAMES = ("ROMP_GC_FREEZE", "ROMP_GC_FREEZE_FULL_MS", "ROMP_GC_FREEZE_FULL_T2")

    def _load(self, tag, **env):
        saved = {k: os.environ.get(k) for k in self.NAMES}
        try:
            for k in self.NAMES:
                os.environ.pop(k, None)
            os.environ.update(env)
            return load_source("romp_kernel_gcf_full_" + tag, os.path.join(BIN, "romp-kernel"))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_the_kernel_reads_the_knobs_and_off_restores_todays_collector(self):
        km = self._load("on", ROMP_GC_FREEZE="on", ROMP_GC_FREEZE_FULL_MS="300", ROMP_GC_FREEZE_FULL_T2="77")
        self.assertEqual((km._GC_FREEZE.full_freeze_ms, km._GC_FREEZE.full_t2), (300.0, 77))
        km = self._load("default", ROMP_GC_FREEZE="on")
        self.assertEqual((km._GC_FREEZE.full_freeze_ms, km._GC_FREEZE.full_t2), (gf.DEFAULT_FULL_FREEZE_MS, gf.DEFAULT_FULL_T2))
        km = self._load("off", ROMP_GC_FREEZE="on", ROMP_GC_FREEZE_FULL_MS="off")
        self.assertIsNone(km._GC_FREEZE.full_freeze_ms)
        before = gc.get_threshold()
        self.assertFalse(km._GC_FREEZE.install_full_freeze(), "off: nothing installed")
        self.assertEqual(gc.get_threshold(), before)
        self.assertNotIn(km._GC_FREEZE.gc_callback, gc.callbacks)

    def test_main_installs_it_after_the_perf_hook(self):
        src = Path(ROOT, "kernel", "kernel.py").read_text()
        body = src[src.index("\ndef main():"):]
        nxt = body.find("\ndef ", 1)               # main may be the file's last function
        body = body if nxt < 0 else body[:nxt]
        i_perf = body.index("_PERF_STATS.install_gc_hook()")
        i_full = body.index("_GC_FREEZE.install_full_freeze()")
        self.assertLess(i_perf, i_full, "the perf hook first, so its timing of a collection never includes the freeze")
        self.assertTrue(re.search(r"\n    _GC_FREEZE\.install_full_freeze\(\)", body), "called at main's top level")


if __name__ == "__main__":
    unittest.main()
