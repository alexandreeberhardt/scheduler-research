from __future__ import annotations

from typing import TYPE_CHECKING

from constants import NS_PER_MS
from task import Task

if TYPE_CHECKING:
    from sched_em import GanttSlot, SchedStats


SHORT_POLICY_NAMES = {
    0: "CFS",
    1: "FIFO",
    2: "RR",
    3: "BATCH",
    5: "IDLE",
    99: "NN",
}

FULL_POLICY_NAMES = {
    0: "SCHED_NORMAL",
    1: "SCHED_FIFO",
    2: "SCHED_RR",
    3: "SCHED_BATCH",
    5: "SCHED_IDLE",
    99: "SCHED_NN",
}


def print_stats_header():
    print(
        f"  {'ALGORITHM':<22} {'TrnArd(ms)':>10} {'Wait(ms)':>9} {'Resp(ms)':>9} "
        f"{'CPU%':>8} {'Thruput':>8} {'CtxSw':>7} {'Fair':>7} {'Starv':>6}"
    )
    print("  " + "-" * 94)


def print_per_task(tasks: List[Task], clock_ns: int | None = None):
    print("\n  Per-task breakdown:")
    print(
        f"  {'PID':<6} {'NAME':<16} {'POL':<6} {'NICE':>5} {'BURST':>8} "
        f"{'TrnArd':>8} {'Wait':>8} {'Resp':>8} {'CtxSw':>6}"
    )
    print("  " + "-" * 76)
    for task in tasks:
        if task.turnaround_ns:
            turnaround_ms = task.turnaround_ns // NS_PER_MS
        elif clock_ns is not None and task.arrival_ns <= clock_ns:
            turnaround_ms = max(0, clock_ns - task.arrival_ns) // NS_PER_MS
        else:
            turnaround_ms = 0

        wait_ns = task.total_wait_ns
        if clock_ns is not None and task.wait_start_ns > 0:
            wait_ns += clock_ns - task.wait_start_ns
        wait_ms = wait_ns // NS_PER_MS

        if task.response_time_ns >= 0:
            response_ms = task.response_time_ns // NS_PER_MS
        elif clock_ns is not None and task.arrival_ns <= clock_ns:
            response_ms = max(0, clock_ns - task.arrival_ns) // NS_PER_MS
        else:
            response_ms = 0

        burst_ms = task.burst_total_ns // NS_PER_MS
        print(
            f"  {task.pid:<6} {task.name:<16} {SHORT_POLICY_NAMES.get(task.policy, '?'):<6} "
            f"{task.nice:>5} {burst_ms:>7}ms {turnaround_ms:>7}ms {wait_ms:>7}ms "
            f"{response_ms:>7}ms {task.ctx_switches:>6}"
        )


def print_gantt(tasks: List[Task], gantt: List["GanttSlot"], end_ns: int, width: int = 80):
    total_ms = max(end_ns // NS_PER_MS, 1)
    print(f"\n  ASCII Gantt Chart (0..{total_ms}ms, width={width})")
    print(f"  {'PID':<4} {'NAME':<16} |{'':^{width}}|  run / wait / turnaround")
    print("  " + "-" * (width + 26))
    for task in tasks:
        bar = list("." * width)
        arrival_ms = task.arrival_ns // NS_PER_MS
        done_ms = (task.arrival_ns + task.turnaround_ns) // NS_PER_MS if task.turnaround_ns else 0
        for column in range(int(arrival_ms / total_ms * width), min(width, int(done_ms / total_ms * width) + 1)):
            bar[column] = "-"
        for slot in gantt:
            if slot.pid != task.pid:
                continue
            start_col = int(slot.start_ns / end_ns * width)
            end_col = int(slot.end_ns / end_ns * width)
            for column in range(max(0, start_col), min(width, end_col + 1)):
                bar[column] = "#"

        run_ms = task.utime_ns // NS_PER_MS
        wait_ms = task.total_wait_ns // NS_PER_MS
        turnaround_ms = task.turnaround_ns // NS_PER_MS
        print(
            f"  {task.pid:<4} {task.name:<16} |{''.join(bar)}|  "
            f"run={run_ms}ms wait={wait_ms}ms TA={turnaround_ms}ms"
        )
    print("  " + "-" * (width + 26))
    print("  Legend: #=running  -=lifetime  .=not yet/done")


def print_task_table(tasks: List[Task]):
    header = (
        f"  {'PID':<6} {'NAME':<16} {'POLICY':<14} {'NICE':<6} {'BURST(ms)':>9}  "
        f"{'IO':>6}  {'RT_PRI':<7} {'DEADLINE'}"
    )
    print(header)
    print("  " + "-" * 78)
    for task in tasks:
        deadline = "yes" if task.deadline_ns else "none"
        print(
            f"  {task.pid:<6} {task.name:<16} {FULL_POLICY_NAMES.get(task.policy, '?'):<14} "
            f"{task.nice:<6} {task.burst_total_ns // NS_PER_MS:>9.1f}  {task.io_ratio:>6.0%}  "
            f"{task.rt_priority:<7} {deadline}"
        )
    print()


def print_stat_row(stats: "SchedStats", highlight: bool = False):
    prefix = "* " if highlight else "  "
    print(
        f"{prefix}{stats.algo_name:<22} {stats.avg_turnaround_ms:>10.1f} {stats.avg_wait_ms:>9.1f} "
        f"{stats.avg_response_ms:>9.1f} {stats.cpu_utilization * 100:>7.1f}% "
        f"{stats.throughput:>8.2f} {stats.ctx_switches:>7} "
        f"{stats.fairness_index:>7.3f} {stats.starvation_count:>6}"
    )
