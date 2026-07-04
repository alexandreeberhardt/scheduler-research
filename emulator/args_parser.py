import argparse


def generate_parser():
    parser = argparse.ArgumentParser(
        description="Linux Scheduler Emulator (CFS / RT / NN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("taskfile", help="Task definition file")
    parser.add_argument(
        "-p",
        "--policy",
        default="CFS",
        choices=["CFS", "BATCH", "IDLE", "FIFO", "RR", "NN"],
        help="Scheduling policy (default: CFS)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=5000,
        help="Simulation duration in ms (default: 5000)",
    )
    parser.add_argument(
        "-c",
        "--compare",
        action="store_true",
        help="Compare all algorithms on same workload",
    )
    parser.add_argument(
        "-g",
        "--gantt",
        action="store_true",
        help="Print ASCII Gantt chart",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose tick-by-tick output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible I/O sleeps",
    )
    parser.add_argument(
        "--cpu-id",
        type=int,
        default=0,
        help="Logical CPU id used in exported synthetic traces",
    )
    parser.add_argument(
        "--trace-out",
        default=None,
        help="Write a synthetic sched_switch trace to this file",
    )
    parser.add_argument(
        "--append-trace",
        action="store_true",
        help="Append to --trace-out instead of overwriting it",
    )
    parser.add_argument(
        "--stats-json",
        default=None,
        help="Write machine-readable benchmark results to JSON",
    )
    parser.add_argument(
        "--model-mode",
        default="none",
        choices=["none", "shadow", "closed-loop"],
        help="Run the learned scheduler in shadow mode or closed-loop mode",
    )
    parser.add_argument(
        "--model-run-dir",
        default=None,
        help="Model run directory containing best_model.pt and dataset_artifacts.json",
    )
    parser.add_argument(
        "--heuristic-policy",
        default="none",
        choices=["none", "metric"],
        help="Use a small metric-aware heuristic instead of a learned model",
    )
    return parser
