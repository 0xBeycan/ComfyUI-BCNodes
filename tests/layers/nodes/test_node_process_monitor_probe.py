"""The Process Monitor's ServerProbe (nodes/process_monitor.py) against a prompt queue with the locking
of ComfyUI's PromptQueue (execution.py: an RLock, a Condition on it, flags) and a prompt worker thread
shaped like main.py's: ComfyUI's free asked for and waited for, and the queue held."""

import heapq
import threading
import time
import types

import pytest


class Queue:
    """ComfyUI's PromptQueue as far as the probe meets it: the same lock, condition, put, get and flags."""

    def __init__(self):
        self.mutex = threading.RLock()
        self.not_empty = threading.Condition(self.mutex)
        self.queue, self.currently_running, self.flags = [], {}, {}

    def put(self, item):
        with self.mutex:
            heapq.heappush(self.queue, item)
            self.not_empty.notify()

    def get(self, timeout=None):
        with self.not_empty:
            while len(self.queue) == 0:
                self.not_empty.wait(timeout=timeout)
                if timeout is not None and len(self.queue) == 0:
                    return None
            item = heapq.heappop(self.queue)
            self.currently_running[0] = item
            return item

    def set_flag(self, name, data):
        with self.mutex:
            self.flags[name] = data
            self.not_empty.notify()

    def get_flags(self, reset=True):
        with self.mutex:
            if reset:
                ret, self.flags = self.flags, {}
                return ret
            return self.flags.copy()

    def get_current_queue_volatile(self):
        with self.mutex:
            return list(self.currently_running.values()), list(self.queue)


def prompt_worker(queue, stop, between, reached, freed, taken):
    """main.py's loop: q.get, then its flags; the first round stops between get_flags() and q.get(),
    where main.py runs gc.collect and empty_cache after a prompt."""
    first = True
    while not stop.is_set():
        flags = queue.get_flags()
        if flags:
            freed.append(sorted(flags))
        if first:
            reached.set()
            between.wait(5)
            first = False
        item = queue.get(timeout=30)
        if item is not None:
            taken.append(item)


@pytest.fixture
def worker(bcnodes):
    queue = Queue()
    stop, between, reached = threading.Event(), threading.Event(), threading.Event()
    freed, taken = [], []
    thread = threading.Thread(target=prompt_worker, args=(queue, stop, between, reached, freed, taken), daemon=True)
    thread.start()
    probe = bcnodes["nodes.process_monitor"].ServerProbe(types.SimpleNamespace(prompt_queue=queue))
    yield types.SimpleNamespace(queue=queue, probe=probe, between=between, reached=reached, freed=freed, taken=taken)
    stop.set()
    between.set()
    queue.set_flag("stop", True)  # wakes q.get
    thread.join(5)


def wait_for(cond, seconds=5.0):
    deadline = time.monotonic() + seconds
    while not cond():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def test_the_free_happens_when_the_wake_up_came_between_get_flags_and_q_get(worker):
    """set_flag's notify finds no waiter while the worker is between get_flags() and q.get(); the probe
    wakes it again while the flags wait, so the free runs at once, not after q.get's timeout."""
    assert worker.reached.wait(5)
    worker.probe.request_free()  # its notify is lost: nobody waits on the condition yet
    worker.between.set()  # the worker goes to sleep in q.get(timeout=30) with the flags set
    assert wait_for(worker.probe.free_done)
    assert worker.freed == [["free_memory", "unload_models"]]


def test_no_prompt_starts_while_the_queue_is_held(worker):
    worker.between.set()
    assert wait_for(lambda: type(worker.queue).get.__code__ in worker.probe._worker_stack())
    with worker.probe.hold():
        assert worker.probe.busy() is None  # re-entrant: the queue is read under the hold
        putter = threading.Thread(target=worker.queue.put, args=((0, "prompt"),), daemon=True)
        putter.start()  # a POST /prompt waits for the lock
        time.sleep(0.3)
        assert putter.is_alive() and worker.taken == []
    putter.join(5)
    assert wait_for(lambda: worker.taken == [(0, "prompt")])
