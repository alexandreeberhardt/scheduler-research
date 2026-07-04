from bisect import insort
from collections import deque
from typing import Dict, List, Optional, Tuple

from constants import (
    MAX_RT_PRIO,
    RR_TIMESLICE_NS,
    SCHED_LATENCY_NS,
    SCHED_MIN_GRAN_NS,
    SCHED_WAKEUP_GRAN_NS,
    WEIGHT_NICE0,
)
from task import Task


class CFSRunQueue:
    def __init__(self):
        self._tree: List[Tuple[int, int, Task]] = []
        self.min_vruntime: int = 0
        self.total_weight: int = 0
        self.curr: Optional[Task] = None

    @property
    def nr_running(self) -> int:
        return len(self._tree) + (1 if self.curr else 0)

    def _insert(self, task: Task, *, account_weight: bool = True):
        insort(self._tree, (task.vruntime, task.pid, task))
        if account_weight:
            self.total_weight += task.load_weight

    def _remove(self, task: Task, *, account_weight: bool = True):
        for index, (_, pid, _) in enumerate(self._tree):
            if pid == task.pid:
                self._tree.pop(index)
                if account_weight:
                    self.total_weight = max(0, self.total_weight - task.load_weight)
                return

    def leftmost(self) -> Optional[Task]:
        return self._tree[0][2] if self._tree else None

    def ordered_tasks(self) -> List[Task]:
        return [task for _, _, task in self._tree]

    def enqueue(self, task: Task, flags: int = 0, clock_ns: int = 0):
        if flags & ENQUEUE_WAKEUP:
            self._place_entity(task, initial=False)
        task.on_rq = True
        if task is not self.curr:
            self._insert(task)

    def dequeue(self, task: Task):
        if task is self.curr:
            self.curr = None
            self.total_weight = max(0, self.total_weight - task.load_weight)
        else:
            self._remove(task)
        task.on_rq = False
        self._update_min_vruntime()

    def pick_next(self, clock_ns: int) -> Optional[Task]:
        if not self._tree:
            return None
        task = self._tree.pop(0)[2]
        self.curr = task
        task.exec_start = clock_ns
        task.prev_sum_exec = task.sum_exec_runtime
        return task

    def pick_specific(self, task: Task, clock_ns: int) -> Optional[Task]:
        for index, (_, pid, queued_task) in enumerate(self._tree):
            if pid == task.pid:
                self._tree.pop(index)
                self.curr = queued_task
                queued_task.exec_start = clock_ns
                queued_task.prev_sum_exec = queued_task.sum_exec_runtime
                return queued_task
        return None

    def put_prev(self, task: Task):
        if self.curr is task:
            self.curr = None
        if task.on_rq:
            self._insert(task, account_weight=False)

    def update_curr(self, clock_ns: int):
        if not self.curr:
            return
        delta = clock_ns - self.curr.exec_start
        if delta <= 0:
            return
        self.curr.exec_start = clock_ns
        self.curr.sum_exec_runtime += delta
        self.curr.vruntime += self._calc_delta_fair(delta, self.curr)
        self._update_min_vruntime()

    def sched_slice(self, task: Task) -> int:
        if self.total_weight == 0 or self.nr_running == 0:
            return SCHED_LATENCY_NS
        slice_ns = int(SCHED_LATENCY_NS * task.load_weight / max(self.total_weight, 1))
        return max(slice_ns, SCHED_MIN_GRAN_NS)

    def check_preempt(self, clock_ns: int) -> bool:
        if not self.curr or not self._tree:
            return False
        ideal = self.sched_slice(self.curr)
        delta = self.curr.sum_exec_runtime - self.curr.prev_sum_exec
        if delta >= ideal:
            return True
        leftmost = self._tree[0][2]
        lag = self.curr.vruntime - leftmost.vruntime
        return lag >= SCHED_MIN_GRAN_NS

    def _calc_delta_fair(self, delta: int, task: Task) -> int:
        if task.load_weight == WEIGHT_NICE0:
            return delta
        return int(delta * WEIGHT_NICE0 / task.load_weight)

    def _update_min_vruntime(self):
        vruntime = self.curr.vruntime if self.curr else self.min_vruntime
        if self._tree:
            left_vruntime = self._tree[0][0]
            vruntime = min(vruntime, left_vruntime) if self.curr else left_vruntime
        self.min_vruntime = max(self.min_vruntime, vruntime)

    def _place_entity(self, task: Task, initial: bool):
        vruntime = self.min_vruntime
        if initial:
            slice_ns = int(
                SCHED_LATENCY_NS
                * task.load_weight
                / max(self.total_weight + task.load_weight, 1)
            )
            vruntime += self._calc_delta_fair(slice_ns, task)
        else:
            vruntime -= self._calc_delta_fair(SCHED_WAKEUP_GRAN_NS, task)
        task.vruntime = max(task.vruntime, vruntime)

    def place_initial(self, task: Task):
        self._place_entity(task, initial=True)

    def dump_tree(self) -> str:
        lines = [f"  CFS rbtree (min_vrt={self.min_vruntime:,}ns):"]
        for vruntime, pid, task in self._tree:
            lag = vruntime - self.min_vruntime
            lines.append(
                f"    pid={pid:<4} nice={task.nice:+3}  vrt={vruntime:>14,}  lag={lag:>+12,}"
            )
        if self.curr:
            lines.append(f"    [RUNNING] pid={self.curr.pid}")
        return "\n".join(lines)


ENQUEUE_WAKEUP = 0x01
ENQUEUE_RESTORE = 0x02
DEQUEUE_SLEEP = 0x01


class RTRunQueue:
    def __init__(self):
        self.queues: Dict[int, deque] = {}
        self.nr_running: int = 0

    def _slot(self, task: Task) -> int:
        return MAX_RT_PRIO - task.rt_priority

    def enqueue(self, task: Task):
        slot = self._slot(task)
        if slot not in self.queues:
            self.queues[slot] = deque()
        self.queues[slot].append(task)
        task.on_rq = True
        self.nr_running += 1
        if task.rt_timeslice_ns == 0:
            task.rt_timeslice_ns = RR_TIMESLICE_NS

    def dequeue(self, task: Task):
        slot = self._slot(task)
        queue = self.queues.get(slot)
        if queue and task in queue:
            queue.remove(task)
            self.nr_running -= 1
            task.on_rq = False

    def pick_next(self) -> Optional[Task]:
        for slot in sorted(self.queues):
            queue = self.queues[slot]
            if queue:
                return queue[0]
        return None

    def rotate_rr(self, task: Task):
        slot = self._slot(task)
        queue = self.queues.get(slot)
        if queue and queue[0] is task:
            queue.rotate(-1)
