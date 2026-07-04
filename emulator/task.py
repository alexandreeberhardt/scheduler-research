from dataclasses import dataclass
from constants import *


def nice_to_prio(nice: int) -> int:
    return MAX_RT_PRIO + nice + 20

def prio_to_nice(prio: int) -> int:
    return prio - MAX_RT_PRIO - 20

def prio_to_weight(prio: int) -> int:
    if prio == MAX_PRIO - 1:
        return WEIGHT_IDLE
    idx = prio - MAX_RT_PRIO
    idx = max(0, min(39, idx))
    return PRIO_TO_WEIGHT[idx]

def is_rt_prio(prio: int) -> bool:
    return prio < MAX_RT_PRIO


@dataclass
class Task:
    # identity
    pid: int
    name: str = "task"

    # scheduling policy & priority
    policy: int = SCHED_NORMAL
    static_prio: int = DEFAULT_PRIO
    prio: int = DEFAULT_PRIO
    rt_priority: int = 0
    nice: int = 0

    # workload description
    burst_total_ns: int = 0
    burst_remaining_ns: int = 0
    io_ratio: float = 0.0
    arrival_ns: int = 0
    deadline_ns: int = 0

    # state
    state: str = "new"

    # CFS entity
    vruntime: int = 0
    sum_exec_runtime: int = 0
    exec_start: int = 0
    prev_sum_exec: int = 0
    load_weight: int = WEIGHT_NICE0
    on_rq: bool = False

    # RT entity
    rt_timeslice_ns: int = RR_TIMESLICE_NS

    # accounting
    start_time_ns: int = 0
    finish_time_ns: int = 0
    total_wait_ns: int = 0
    wait_start_ns: int = 0
    response_time_ns: int = -1
    utime_ns: int = 0
    ctx_switches: int = 0
    io_sleep_ns: int = 0


    def set_policy(self, policy: int, nice: int = 0, rt_prio: int = 0):
        self.policy = policy
        if policy in (SCHED_FIFO, SCHED_RR):
            rt_prio = max(1, min(99, rt_prio))
            self.rt_priority = rt_prio
            self.prio = MAX_RT_PRIO - rt_prio
            self.static_prio = self.prio
            self.nice = 0
            self.load_weight = WEIGHT_NICE0
        elif policy == SCHED_IDLE:
            self.rt_priority = 0
            self.static_prio = MAX_PRIO - 1
            self.prio = MAX_PRIO - 1
            self.nice = MAX_NICE
            self.load_weight = WEIGHT_IDLE
        else:
            nice = max(MIN_NICE, min(MAX_NICE, nice))
            self.nice = nice
            self.rt_priority = 0
            self.static_prio = nice_to_prio(nice)
            self.prio = self.static_prio
            self.load_weight = prio_to_weight(self.static_prio)

    @property
    def turnaround_ns(self) -> int:
        if self.finish_time_ns > 0:
            return max(0, self.finish_time_ns - self.arrival_ns)
        return 0

    def __repr__(self):
        pol = {SCHED_NORMAL:"CFS", SCHED_BATCH:"BATCH", SCHED_IDLE:"IDLE",
               SCHED_FIFO:"FIFO", SCHED_RR:"RR"}.get(self.policy,"?")
        return (f"Task(pid={self.pid} name={self.name} policy={pol} "
                f"nice={self.nice} burst={self.burst_total_ns//NS_PER_MS}ms "
                f"io={self.io_ratio:.0%} state={self.state})")
