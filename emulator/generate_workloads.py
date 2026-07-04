#!/usr/bin/env python3

import argparse
import random
from pathlib import Path


TASK_NAMES = [
    "bash",
    "clang",
    "gcc_build",
    "postgres",
    "redis",
    "firefox",
    "chrome",
    "node",
    "python",
    "pytest",
    "Xorg",
    "wayland",
    "pipewire",
    "ffmpeg",
    "gstreamer",
    "backup",
    "rsync",
    "nginx",
    "kworker",
    "watchdog",
    "dnsmasq",
    "java",
    "gradle",
    "cargo",
    "scanner",
    "indexer",
    "telemetry",
    "notebook",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Génère des workloads .tasks variés pour l'émulateur."
    )
    parser.add_argument(
        "--output-dir",
        default="emulator/tasks/generated",
        help="Dossier de sortie",
    )
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-tasks", type=int, default=10)
    parser.add_argument("--max-tasks", type=int, default=24)
    parser.add_argument("--max-arrival-ms", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(args.count):
        path = output_dir / f"workload_{index:03d}.tasks"
        write_workload(
            path,
            rng,
            min_tasks=args.min_tasks,
            max_tasks=args.max_tasks,
            max_arrival_ms=args.max_arrival_ms,
        )

    print(f"{args.count} workloads générés dans {output_dir}")


def write_workload(
    path: Path,
    rng: random.Random,
    *,
    min_tasks: int,
    max_tasks: int,
    max_arrival_ms: int,
) -> None:
    task_count = rng.randint(min_tasks, max_tasks)
    pid_pool = list(range(1000, 100000))
    rng.shuffle(pid_pool)

    lines = [
        "# pid arrival_ms burst_ms nice io_ratio policy rt_prio deadline_ms name"
    ]
    cfs_seen = 0
    rt_seen = 0

    for task_index in range(task_count):
        pid = pid_pool[task_index]
        arrival_ms = rng.randint(0, max_arrival_ms)
        burst_ms = draw_burst_ms(rng)
        nice = rng.choice([-15, -10, -5, 0, 0, 0, 5, 10, 15])
        io_ratio = draw_io_ratio(rng)
        policy = draw_policy(rng)
        rt_prio = 0
        if policy in {"FIFO", "RR"}:
            rt_prio = rng.randint(10, 90)
            rt_seen += 1
        else:
            cfs_seen += 1
        deadline_ms = draw_deadline_ms(rng, burst_ms, policy)
        name = f"{rng.choice(TASK_NAMES)}_{task_index}"
        lines.append(
            f"{pid} {arrival_ms} {burst_ms} {nice} {io_ratio:.2f} "
            f"{policy} {rt_prio} {deadline_ms} {name}"
        )

    if cfs_seen == 0:
        lines.append(
            f"{pid_pool[task_count]} 0 120 0 0.20 CFS 0 0 {rng.choice(TASK_NAMES)}_extra"
        )
    if rt_seen == 0:
        lines.append(
            f"{pid_pool[task_count + 1]} 0 40 0 0.00 RR 60 0 watchdog_extra"
        )

    path.write_text("\n".join(lines) + "\n")


def draw_burst_ms(rng: random.Random) -> int:
    band = rng.random()
    if band < 0.25:
        return rng.randint(15, 60)
    if band < 0.65:
        return rng.randint(60, 220)
    return rng.randint(220, 900)


def draw_io_ratio(rng: random.Random) -> float:
    band = rng.random()
    if band < 0.25:
        return rng.uniform(0.0, 0.05)
    if band < 0.75:
        return rng.uniform(0.05, 0.30)
    return rng.uniform(0.30, 0.70)


def draw_policy(rng: random.Random) -> str:
    band = rng.random()
    if band < 0.72:
        return "CFS"
    if band < 0.84:
        return "BATCH"
    if band < 0.90:
        return "IDLE"
    if band < 0.95:
        return "FIFO"
    return "RR"


def draw_deadline_ms(rng: random.Random, burst_ms: int, policy: str) -> int:
    if policy in {"FIFO", "RR"}:
        return 0
    if rng.random() < 0.7:
        return 0
    return rng.randint(max(20, burst_ms // 3), max(40, burst_ms * 2))


if __name__ == "__main__":
    main()
