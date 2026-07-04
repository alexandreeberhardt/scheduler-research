import math
from typing import Iterable, Optional

from constants import (
    MAX_PRIO,
    NS_PER_MS,
    SCHED_BATCH,
    SCHED_FIFO,
    SCHED_IDLE,
    SCHED_NN,
    SCHED_NORMAL,
    SCHED_RR,
)
from task import Task, is_rt_prio


STATE_ORDER = ["new", "ready", "running", "sleeping", "dead"]
POLICY_ORDER = [
    SCHED_NORMAL,
    SCHED_BATCH,
    SCHED_IDLE,
    SCHED_FIFO,
    SCHED_RR,
    SCHED_NN,
]

TASK_FEATURE_NAMES = [
    "pid_log",
    "observed",
    "is_current",
    "on_rq",
    "is_rt",
    "nice_norm",
    "prio_norm",
    "static_prio_norm",
    "rt_priority_norm",
    "load_weight_log",
    "arrival_age_log_ms",
    "burst_total_log_ms",
    "burst_remaining_log_ms",
    "executed_log_ms",
    "wait_log_ms",
    "response_log_ms",
    "turnaround_log_ms",
    "remaining_ratio",
    "io_ratio",
    "has_deadline",
    "deadline_slack_signed_log_ms",
    "vruntime_delta_signed_log_ms",
    "ctx_switches_log",
    "state_new",
    "state_ready",
    "state_running",
    "state_sleeping",
    "state_dead",
    "policy_cfs",
    "policy_batch",
    "policy_idle",
    "policy_fifo",
    "policy_rr",
    "policy_nn",
]

GLOBAL_FEATURE_NAMES = [
    "clock_log_ms",
    "history_fill_ratio",
    "total_tasks_log",
    "observed_ratio",
    "completed_ratio",
    "ready_ratio",
    "sleeping_ratio",
    "rt_runnable_ratio",
    "cfs_runnable_ratio",
    "cpu_busy_ratio",
    "has_current",
    "current_is_rt",
    "current_runtime_log_ms",
]


def effective_turnaround_ns(task: Task, clock_ns: int) -> int:
    if task.finish_time_ns > 0:
        return max(0, task.finish_time_ns - task.arrival_ns)
    if task.arrival_ns <= clock_ns:
        return max(0, clock_ns - task.arrival_ns)
    return 0


def effective_wait_ns(task: Task, clock_ns: int) -> int:
    wait_ns = task.total_wait_ns
    if task.wait_start_ns > 0:
        wait_ns += clock_ns - task.wait_start_ns
    return max(0, wait_ns)


def effective_response_ns(task: Task, clock_ns: int) -> int:
    if task.response_time_ns >= 0:
        return task.response_time_ns
    if task.arrival_ns <= clock_ns:
        return max(0, clock_ns - task.arrival_ns)
    return 0


def task_feature_names() -> list[str]:
    return list(TASK_FEATURE_NAMES)


def global_feature_names() -> list[str]:
    return list(GLOBAL_FEATURE_NAMES)


def build_task_features(
    task: Task,
    clock_ns: int,
    *,
    current_pid: Optional[int],
    cfs_min_vruntime: int,
) -> list[float]:
    observed = 1.0 if task.arrival_ns <= clock_ns else 0.0
    arrival_age_ns = max(0, clock_ns - task.arrival_ns) if observed else 0
    wait_ns = effective_wait_ns(task, clock_ns)
    response_ns = effective_response_ns(task, clock_ns)
    turnaround_ns = effective_turnaround_ns(task, clock_ns)
    deadline_slack_ns = task.deadline_ns - clock_ns if task.deadline_ns else 0
    remaining_ratio = (
        task.burst_remaining_ns / task.burst_total_ns if task.burst_total_ns > 0 else 0.0
    )

    features = [
        math.log1p(max(task.pid, 0)),
        observed,
        1.0 if current_pid == task.pid else 0.0,
        1.0 if task.on_rq else 0.0,
        1.0 if is_rt_prio(task.prio) else 0.0,
        task.nice / 20.0,
        task.prio / MAX_PRIO,
        task.static_prio / MAX_PRIO,
        task.rt_priority / 99.0,
        math.log1p(max(task.load_weight, 0)),
        _log_ms(arrival_age_ns),
        _log_ms(task.burst_total_ns),
        _log_ms(task.burst_remaining_ns),
        _log_ms(task.utime_ns),
        _log_ms(wait_ns),
        _log_ms(response_ns),
        _log_ms(turnaround_ns),
        remaining_ratio,
        task.io_ratio,
        1.0 if task.deadline_ns else 0.0,
        _signed_log_ms(deadline_slack_ns),
        _signed_log_ms(task.vruntime - cfs_min_vruntime),
        math.log1p(max(task.ctx_switches, 0)),
    ]
    features.extend(_one_hot(task.state, STATE_ORDER))
    features.extend(_one_hot(task.policy, POLICY_ORDER))
    return features


def build_global_features(
    all_tasks: Iterable[Task],
    clock_ns: int,
    *,
    current_task: Optional[Task],
    cfs_runnable: int,
    rt_runnable: int,
    cpu_busy_ratio: float,
    history_length: int,
    max_history: int,
) -> list[float]:
    tasks = list(all_tasks)
    total = max(len(tasks), 1)
    observed = [task for task in tasks if task.arrival_ns <= clock_ns]
    completed = sum(task.state == "dead" for task in tasks)
    ready = sum(task.state == "ready" for task in tasks)
    sleeping = sum(task.state == "sleeping" for task in tasks)

    current_runtime_ns = 0
    if current_task is not None and current_task.state == "running":
        current_runtime_ns = max(0, clock_ns - current_task.exec_start)

    return [
        _log_ms(clock_ns),
        min(history_length, max_history) / max(max_history, 1),
        math.log1p(len(tasks)),
        len(observed) / total,
        completed / total,
        ready / total,
        sleeping / total,
        rt_runnable / total,
        cfs_runnable / total,
        cpu_busy_ratio,
        1.0 if current_task is not None else 0.0,
        1.0 if current_task is not None and is_rt_prio(current_task.prio) else 0.0,
        _log_ms(current_runtime_ns),
    ]


def _log_ms(value_ns: int) -> float:
    return math.log1p(max(value_ns, 0) / NS_PER_MS)


def _signed_log_ms(value_ns: int) -> float:
    if value_ns == 0:
        return 0.0
    sign = 1.0 if value_ns > 0 else -1.0
    return sign * math.log1p(abs(value_ns) / NS_PER_MS)


def _one_hot(value, ordered_values: list) -> list[float]:
    return [1.0 if value == expected else 0.0 for expected in ordered_values]
