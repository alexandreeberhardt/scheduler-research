import sys
from typing import List

from constants import NS_PER_MS, SCHED_NORMAL
from task import Task


POLICY_MAP = {
    "CFS": 0,
    "NORMAL": 0,
    "0": 0,
    "FIFO": 1,
    "1": 1,
    "RR": 2,
    "2": 2,
    "BATCH": 3,
    "3": 3,
    "IDLE": 5,
    "5": 5,
    "NN": 99,
    "99": 99,
}


def parse_taskfile(path: str) -> List[Task]:
    tasks = []
    with open(path) as file_handle:
        for line_number, raw_line in enumerate(file_handle, 1):
            line = raw_line.split("#")[0].strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 6:
                print(
                    f"Warning: line {line_number}: need >=6 fields, got {len(parts)}",
                    file=sys.stderr,
                )
                continue

            parsed_task = parse_task_line(parts, line_number)
            if parsed_task is not None:
                tasks.append(parsed_task)

    return tasks


def parse_task_line(parts: list[str], line_number: int) -> Task | None:
    try:
        pid = int(parts[0])
        arrival_ms = int(parts[1])
        burst_ms = int(parts[2])
        nice = int(parts[3])
        io_ratio = float(parts[4])
        policy_name = parts[5].upper()
        rt_priority = int(parts[6]) if len(parts) > 6 else 0
        deadline_ms = int(parts[7]) if len(parts) > 7 else 0
        name = parts[8][:15] if len(parts) > 8 else "task"
        policy = POLICY_MAP.get(policy_name, SCHED_NORMAL)
    except (ValueError, IndexError) as error:
        print(f"Warning: line {line_number}: {error}", file=sys.stderr)
        return None

    task = Task(pid=pid, name=name)
    task.burst_total_ns = burst_ms * NS_PER_MS
    task.burst_remaining_ns = task.burst_total_ns
    task.arrival_ns = arrival_ms * NS_PER_MS
    task.io_ratio = max(0.0, min(1.0, io_ratio))
    task.deadline_ns = (arrival_ms + deadline_ms) * NS_PER_MS if deadline_ms else 0
    task.response_time_ns = -1
    task.set_policy(policy, nice, rt_priority)
    return task
