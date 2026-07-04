import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, List, Tuple
from task import Task, is_rt_prio
from constants import *
from run_queue import *
from args_parser import generate_parser
from decision_features import (
    build_global_features,
    build_task_features,
    effective_response_ns,
    effective_turnaround_ns,
    effective_wait_ns,
)
from heuristic_policy import MetricHeuristicPolicy
from model_policy import SchedulerModelPolicy
from task_parser import parse_taskfile, POLICY_MAP
from helpers import *

@dataclass
class GanttSlot:
    pid: int
    name: str
    start_ns: int
    end_ns: int


@dataclass
class SchedStats:
    algo_name: str = ""
    tasks_completed: int = 0
    avg_turnaround_ms: float = 0.0
    avg_wait_ms: float = 0.0
    avg_response_ms: float = 0.0
    cpu_utilization: float = 0.0
    throughput: float = 0.0
    ctx_switches: int = 0
    fairness_index: float = 0.0
    starvation_count: int = 0


@dataclass
class SchedTraceEvent:
    cpu_id: int
    timestamp_ns: int
    prev_pid: int
    prev_name: str
    prev_prio: int
    next_pid: int
    next_name: str
    next_prio: int


def policy_name(policy: int) -> str:
    return {
        SCHED_NORMAL: "CFS",
        SCHED_BATCH: "BATCH",
        SCHED_IDLE: "IDLE",
        SCHED_FIFO: "FIFO",
        SCHED_RR: "RR",
        SCHED_NN: "NN",
    }.get(policy, "?")


def run_label(policy: int, decision_mode: str, decision_source_name: str = "model") -> str:
    base = policy_name(policy)
    if decision_mode == "shadow":
        return f"{base} + {decision_source_name} shadow"
    if decision_mode == "closed_loop":
        return f"{base} + {decision_source_name} closed-loop"
    return base


def stats_delta(candidate: SchedStats, baseline: SchedStats) -> dict:
    return {
        "avg_turnaround_ms": candidate.avg_turnaround_ms - baseline.avg_turnaround_ms,
        "avg_wait_ms": candidate.avg_wait_ms - baseline.avg_wait_ms,
        "avg_response_ms": candidate.avg_response_ms - baseline.avg_response_ms,
        "fairness_index": candidate.fairness_index - baseline.fairness_index,
        "ctx_switches": candidate.ctx_switches - baseline.ctx_switches,
        "starvation_count": candidate.starvation_count - baseline.starvation_count,
    }


def metric_direction(metric: str) -> int:
    if metric == "fairness_index":
        return 1
    return -1


def summarize_model_events(events: List[dict]) -> dict:
    summary = {
        "opportunities": len(events),
        "selected_candidate": 0,
        "no_candidates": 0,
        "same_as_cfs": 0,
        "different_from_cfs": 0,
        "applied_predictions": 0,
        "fallback_to_cfs": 0,
    }
    for event in events:
        reason = event["reason"]
        if reason in summary:
            summary[reason] += 1
        if event["matched_cfs"]:
            summary["same_as_cfs"] += 1
        elif event["selected_task_pid"] is not None:
            summary["different_from_cfs"] += 1
        if event["applied"]:
            summary["applied_predictions"] += 1
        elif event["mode"] == "closed_loop":
            summary["fallback_to_cfs"] += 1
    return summary


def summarize_shadow_records(records: List[dict]) -> dict:
    metrics = [
        "avg_turnaround_ms",
        "avg_wait_ms",
        "avg_response_ms",
        "fairness_index",
        "ctx_switches",
    ]
    summary = {
        "divergences": len(records),
        "metrics": {},
    }
    for metric in metrics:
        better = 0
        worse = 0
        ties = 0
        deltas = [record["delta"][metric] for record in records]
        for delta in deltas:
            signed = delta * metric_direction(metric)
            if signed > 1e-9:
                better += 1
            elif signed < -1e-9:
                worse += 1
            else:
                ties += 1
        summary["metrics"][metric] = {
            "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
            "better": better,
            "worse": worse,
            "tied": ties,
        }
    return summary


class Scheduler:
    def __init__(self, policy: int = SCHED_NORMAL, verbose: bool = False,
                 sched_latency_ns: int = SCHED_LATENCY_NS,
                 sched_min_gran_ns: int = SCHED_MIN_GRAN_NS,
                 sched_wakeup_gran_ns: int = SCHED_WAKEUP_GRAN_NS,
                 cpu_id: int = 0,
                 model_policy: Optional[SchedulerModelPolicy] = None,
                 decision_mode: str = "baseline",
                 decision_source_name: str = "model",
                 record_decisions: bool = False,
                 decision_teacher: str = "cfs",
                 max_history: int = 256):
        self.policy = policy
        self.verbose = verbose
        self.clock = 0
        self.tick_ns = NS_PER_MS
        self.cpu_id = cpu_id
        self.idle_name = f"swapper/{cpu_id}"
        self.model_policy = model_policy
        self.decision_mode = decision_mode
        self.decision_source_name = decision_source_name
        self.record_decisions = record_decisions
        self.decision_teacher = decision_teacher
        self.teacher_policy = MetricHeuristicPolicy() if decision_teacher == "heuristic" else None
        self.max_history = max_history
        self.run_duration_ns = 0

        self.cfs = CFSRunQueue()
        self.rt = RTRunQueue()

        self.curr: Optional[Task] = None
        self.all_tasks: List[Task] = []
        self.completed: List[Task] = []
        self.gantt: List[GanttSlot] = []
        self.trace_events: List[SchedTraceEvent] = []
        self.model_events: List[dict] = []
        self.shadow_records: List[dict] = []
        self.decision_history: List[List[float]] = []
        self.decision_samples: List[dict] = []

        self.nr_ctx_switches: int = 0
        self.cpu_busy_ns: int = 0

    def add_task(self, task: Task):
        self.all_tasks.append(task)

    def _activate(self, task: Task, wakeup: bool = False):
        task.state = "ready"
        if task.start_time_ns == 0:
            task.start_time_ns = self.clock
        if task.wait_start_ns == 0:
            task.wait_start_ns = self.clock

        flags = ENQUEUE_WAKEUP if wakeup else 0

        if is_rt_prio(task.prio):
            self.rt.enqueue(task)
        else:
            if not wakeup:
                self.cfs.place_initial(task)
            self.cfs.enqueue(task, flags, self.clock)

    def _deactivate(self, task: Task):
        self._accum_wait(task)
        if is_rt_prio(task.prio):
            self.rt.dequeue(task)
        else:
            self.cfs.dequeue(task)

    def _accum_wait(self, task: Task):
        if task.wait_start_ns > 0:
            task.total_wait_ns += self.clock - task.wait_start_ns
            task.wait_start_ns = 0

    def _task_identity(self, task: Optional[Task]) -> Tuple[int, str, int]:
        if task is None:
            return 0, self.idle_name, DEFAULT_PRIO
        return task.pid, task.name, task.prio

    def _task_ref(self, task: Optional[Task]) -> dict:
        if task is None:
            return {"pid": 0, "name": self.idle_name}
        return {"pid": task.pid, "name": task.name}

    def _task_features(self, task: Task, *, current_task: Optional[Task]) -> List[float]:
        current_pid = current_task.pid if current_task is not None else None
        return build_task_features(
            task,
            self.clock,
            current_pid=current_pid,
            cfs_min_vruntime=self.cfs.min_vruntime,
        )

    def _history_features_for_task(self, task: Task) -> List[float]:
        task_snapshot = copy.copy(task)
        task_snapshot.state = "running"
        return build_task_features(
            task_snapshot,
            self.clock,
            current_pid=task.pid,
            cfs_min_vruntime=self.cfs.min_vruntime,
        )

    def _append_decision_history(self, task: Optional[Task]) -> None:
        if task is None:
            return
        self.decision_history.append(self._history_features_for_task(task))

    def _history_window(self) -> List[List[float]]:
        return self.decision_history[-self.max_history :]

    def _find_task_by_pid(self, pid: int) -> Optional[Task]:
        for task in self.all_tasks:
            if task.pid == pid:
                return task
        return None

    def _cfs_candidates(self, prev: Optional[Task]) -> List[Task]:
        candidates: List[Task] = []
        if (
            prev is not None
            and prev.state == "running"
            and not is_rt_prio(prev.prio)
            and prev.burst_remaining_ns > 0
        ):
            candidates.append(prev)
        for task in self.cfs.ordered_tasks():
            if all(existing.pid != task.pid for existing in candidates):
                candidates.append(task)
        return candidates

    def _global_features(self) -> List[float]:
        return build_global_features(
            self.all_tasks,
            self.clock,
            current_task=self.curr,
            cfs_runnable=self.cfs.nr_running,
            rt_runnable=self.rt.nr_running,
            cpu_busy_ratio=(self.cpu_busy_ns / self.clock if self.clock > 0 else 0.0),
            history_length=len(self.decision_history),
            max_history=self.max_history,
        )

    def _build_decision_state(self, candidates: List[Task]) -> dict:
        return {
            "timestamp_ns": self.clock,
            "history_features": [list(features) for features in self._history_window()],
            "candidate_features": [
                self._task_features(task, current_task=self.curr) for task in candidates
            ],
            "candidate_tasks": [self._task_ref(task) for task in candidates],
            "all_task_features": [
                self._task_features(task, current_task=self.curr) for task in self.all_tasks
            ],
            "all_tasks": [self._task_ref(task) for task in self.all_tasks],
            "global_features": self._global_features(),
        }

    def _counterfactual_key(self, stats: SchedStats) -> tuple:
        return (
            stats.starvation_count,
            stats.avg_turnaround_ms,
            stats.avg_wait_ms,
            stats.avg_response_ms,
            -stats.fairness_index,
            stats.ctx_switches,
        )

    def _cfs_target_index(
        self,
        candidates: List[Task],
        cfs_next: Optional[Task],
    ) -> int:
        if cfs_next is None:
            return 0
        for index, candidate in enumerate(candidates):
            if candidate.pid == cfs_next.pid:
                return index
        return 0

    def _teacher_scores(
        self,
        candidates: List[Task],
        cfs_next: Optional[Task],
        decision_state: dict,
    ) -> tuple[int, list[float]]:
        if self.decision_teacher == "heuristic" and self.teacher_policy is not None:
            scores, best_index = self.teacher_policy.score_candidates(decision_state)
            return best_index, scores

        target_index = self._teacher_target_index(candidates, cfs_next)
        scores = [0.0] * len(candidates)
        scores[target_index] = 1.0
        return target_index, scores

    def _teacher_target_index(
        self,
        candidates: List[Task],
        cfs_next: Optional[Task],
    ) -> int:
        if not candidates:
            raise ValueError("Cannot build a target without candidates")
        if self.decision_teacher == "oracle":
            rng_state = random.getstate()
            best_index = 0
            best_key = None
            for index, candidate in enumerate(candidates):
                stats = self._simulate_counterfactual(candidate.pid, rng_state)
                key = self._counterfactual_key(stats)
                if best_key is None or key < best_key:
                    best_key = key
                    best_index = index
            random.setstate(rng_state)
            return best_index

        return self._cfs_target_index(candidates, cfs_next)

    def _peek_next(self) -> Tuple[Optional[Task], str]:
        rt_next = self.rt.pick_next()
        if rt_next:
            return rt_next, "rt"
        return self.cfs.leftmost(), "cfs"

    def _apply_schedule_choice(self, prev: Optional[Task], next_task: Optional[Task]) -> None:
        if prev is not None and next_task is prev:
            self._append_decision_history(next_task)
            return
        if prev:
            self.cfs.update_curr(self.clock)
            if prev.state == "running":
                prev.state = "ready"
                prev.wait_start_ns = self.clock
                if not is_rt_prio(prev.prio):
                    self.cfs.put_prev(prev)

        if next_task and not is_rt_prio(next_task.prio):
            picked = self.cfs.pick_specific(next_task, self.clock)
            if picked is not None:
                next_task = picked

        self._context_switch(prev, next_task)

    def _select_model_task(
        self,
        cfs_next: Optional[Task],
        candidates: List[Task],
        decision_state: dict,
    ) -> Tuple[Optional[Task], dict]:
        selection = self.model_policy.predict(decision_state)
        selected_task = None
        if selection.selected_task_pid is not None:
            selected_task = self._find_task_by_pid(selection.selected_task_pid)

        event = {
            "timestamp_ns": self.clock,
            "mode": self.decision_mode,
            "history_length": len(self.decision_history),
            "candidate_count": len(candidates),
            "cfs_task": self._task_ref(cfs_next),
            "selected_task_name": selection.selected_task_name,
            "selected_task_pid": selection.selected_task_pid,
            "confidence": selection.confidence,
            "reason": selection.reason,
            "matched_cfs": (
                selected_task is not None
                and cfs_next is not None
                and selected_task.pid == cfs_next.pid
            ),
            "applied": False,
        }
        return selected_task, event

    def _fork_for_counterfactual(self) -> "Scheduler":
        branch = copy.deepcopy(self)
        branch.model_policy = None
        branch.decision_mode = "baseline"
        branch.record_decisions = False
        branch.model_events = []
        branch.shadow_records = []
        branch.decision_samples = []
        return branch

    def _run_until(self, duration_ns: int) -> SchedStats:
        while self.clock < duration_ns:
            self.tick()
            alive = any(task.state != "dead" for task in self.all_tasks)
            if not alive:
                break
        return self._collect_stats()

    def _run_until_complete(self) -> SchedStats:
        while True:
            alive = any(task.state != "dead" for task in self.all_tasks)
            if not alive:
                break
            self.tick()
        return self._collect_stats()

    def _effective_turnaround_ns(self, task: Task) -> int:
        return effective_turnaround_ns(task, self.clock)

    def _effective_wait_ns(self, task: Task) -> int:
        return effective_wait_ns(task, self.clock)

    def _effective_response_ns(self, task: Task) -> int:
        return effective_response_ns(task, self.clock)

    def _simulate_counterfactual(self, next_pid: int, rng_state) -> SchedStats:
        branch = self._fork_for_counterfactual()
        random.setstate(rng_state)
        branch_next = branch._find_task_by_pid(next_pid) if next_pid else None
        branch._apply_schedule_choice(branch.curr, branch_next)
        return branch._run_until_complete()

    def _evaluate_shadow_choice(
        self,
        cfs_next: Task,
        model_next: Task,
        model_event: dict,
    ) -> None:
        rng_state = random.getstate()
        cfs_stats = self._simulate_counterfactual(cfs_next.pid, rng_state)
        model_stats = self._simulate_counterfactual(model_next.pid, rng_state)
        random.setstate(rng_state)

        self.shadow_records.append(
            {
                "timestamp_ns": self.clock,
                "history_length": len(self.decision_history),
                "cfs_task": self._task_ref(cfs_next),
                "model_task": self._task_ref(model_next),
                "confidence": model_event["confidence"],
                "cfs_remaining": asdict(cfs_stats),
                "model_remaining": asdict(model_stats),
                "delta": stats_delta(model_stats, cfs_stats),
            }
        )

    def schedule(self):
        prev = self.curr
        next_task, source = self._peek_next()

        if next_task is prev:
            return

        candidates: List[Task] = []
        decision_state = None
        if source == "cfs":
            candidates = self._cfs_candidates(prev)
            if candidates:
                decision_state = self._build_decision_state(candidates)
                if self.record_decisions and next_task is not None:
                    executed_index, teacher_scores = self._teacher_scores(
                        candidates,
                        next_task,
                        decision_state,
                    )
                    cfs_index = self._cfs_target_index(candidates, next_task)
                    self.decision_samples.append(
                        {
                            "timestamp_ns": self.clock,
                            "history_length": len(self.decision_history),
                            "candidate_features": decision_state["candidate_features"],
                            "candidate_tasks": decision_state["candidate_tasks"],
                            "all_task_features": decision_state["all_task_features"],
                            "all_tasks": decision_state["all_tasks"],
                            "global_features": decision_state["global_features"],
                            "target_index": executed_index,
                            "teacher_scores": teacher_scores,
                            "cfs_index": cfs_index,
                            "executed_task_features": self._history_features_for_task(next_task),
                        }
                    )

        if (
            self.model_policy
            and source == "cfs"
            and self.decision_mode in {"shadow", "closed_loop"}
            and candidates
        ):
            if decision_state is None:
                decision_state = self._build_decision_state(candidates)
            model_next, model_event = self._select_model_task(next_task, candidates, decision_state)
            if self.decision_mode == "shadow":
                if (
                    model_next is not None
                    and next_task is not None
                    and model_next.pid != next_task.pid
                ):
                    self._evaluate_shadow_choice(next_task, model_next, model_event)
            elif self.decision_mode == "closed_loop" and model_next is not None:
                next_task = model_next
                model_event["applied"] = True
            self.model_events.append(model_event)

        self._apply_schedule_choice(prev, next_task)

    def _context_switch(self, prev: Optional[Task], next_task: Optional[Task]):
        if self.verbose and next_task and prev and prev is not next_task:
            print(f"  [{self.clock:>12,}ns] CTX: {prev.name}(pid={prev.pid}) "
                  f"=> {next_task.name}(pid={next_task.pid})  "
                  f"vrt={getattr(next_task,'vruntime',0):,}")

        if prev and prev is not next_task:
            prev.ctx_switches += 1
        self.nr_ctx_switches += 1
        prev_pid, prev_name, prev_prio = self._task_identity(prev)
        next_pid, next_name, next_prio = self._task_identity(next_task)
        self.trace_events.append(
            SchedTraceEvent(
                cpu_id=self.cpu_id,
                timestamp_ns=self.clock,
                prev_pid=prev_pid,
                prev_name=prev_name,
                prev_prio=prev_prio,
                next_pid=next_pid,
                next_name=next_name,
                next_prio=next_prio,
            )
        )

        self.curr = next_task
        self.cfs.curr = None
        if next_task:
            next_task.state = "running"
            next_task.exec_start = self.clock
            if not is_rt_prio(next_task.prio):
                self.cfs.curr = next_task
                next_task.prev_sum_exec = next_task.sum_exec_runtime

            if next_task.response_time_ns < 0:
                next_task.response_time_ns = self.clock - next_task.start_time_ns
            if next_task.wait_start_ns > 0:
                self._accum_wait(next_task)
            self._append_decision_history(next_task)

    def tick(self):
        self.clock += self.tick_ns

        self._wake_sleeping()

        for t in self.all_tasks:
            if t.state == "new" and t.arrival_ns <= self.clock:
                self._activate(t, wakeup=False)
                if self.verbose:
                    print(f"  [{self.clock:>12,}ns] ARRIVE pid={t.pid} {t.name}")

        if self.curr:
            self.cpu_busy_ns += self.tick_ns

            self.curr.utime_ns += self.tick_ns
            self.curr.burst_remaining_ns = max(0, self.curr.burst_remaining_ns - self.tick_ns)

            if not is_rt_prio(self.curr.prio):
                self.cfs.update_curr(self.clock)

            if self.curr.policy == SCHED_RR:
                self.curr.rt_timeslice_ns = max(0, self.curr.rt_timeslice_ns - self.tick_ns)
                if self.curr.rt_timeslice_ns == 0:
                    self.curr.rt_timeslice_ns = RR_TIMESLICE_NS
                    self.rt.rotate_rr(self.curr)

            io_prob = self.curr.io_ratio
            if self.curr.burst_remaining_ns > 0 and random.random() < io_prob:
                sleep_ms = random.randint(1, 5)
                self.curr.io_sleep_ns = sleep_ms * NS_PER_MS
                self.curr.state = "sleeping"
                if self.verbose:
                    print(f"  [{self.clock:>12,}ns] I/O   pid={self.curr.pid} "
                          f"sleep={sleep_ms}ms")
                self._deactivate(self.curr)
                self.curr = None
                self.cfs.curr = None
                self.schedule()
                return

            if self.curr.burst_remaining_ns == 0:
                self._finish_task(self.curr)
                return

            if self.gantt and self.gantt[-1].pid == self.curr.pid:
                self.gantt[-1].end_ns = self.clock
            else:
                self.gantt.append(GanttSlot(
                    pid=self.curr.pid, name=self.curr.name,
                    start_ns=self.clock - self.tick_ns, end_ns=self.clock))

            if not is_rt_prio(self.curr.prio) and self.policy != SCHED_NN:
                if self.cfs.check_preempt(self.clock):
                    self.schedule()
                    return

            if self.rt.nr_running > 0:
                rt_top = self.rt.pick_next()
                if rt_top and rt_top.prio < self.curr.prio:
                    self.schedule()
                    return

        else:
            # CPU idle: try to schedule
            self.schedule()

    def _finish_task(self, task: Task):
        task.state = "dead"
        task.finish_time_ns = self.clock
        self._deactivate(task)
        self.curr = None
        self.cfs.curr = None
        self.completed.append(task)

        if self.verbose:
            print(f"  [{self.clock:>12,}ns] DONE  pid={task.pid} {task.name}  "
                  f"ta={task.turnaround_ns//NS_PER_MS}ms "
                  f"wait={task.total_wait_ns//NS_PER_MS}ms")

        self.schedule()

    def _wake_sleeping(self):
        for t in self.all_tasks:
            if t.state == "sleeping" and t.io_sleep_ns > 0:
                t.io_sleep_ns = max(0, t.io_sleep_ns - self.tick_ns)
                if t.io_sleep_ns == 0:
                    t.state = "ready"
                    t.wait_start_ns = self.clock
                    if is_rt_prio(t.prio):
                        self.rt.enqueue(t)
                    else:
                        self.cfs.enqueue(t, ENQUEUE_WAKEUP, self.clock)
                    if self.verbose:
                        print(f"  [{self.clock:>12,}ns] WAKE  pid={t.pid} {t.name}")

    def run(self, tasks: List[Task], duration_ns: int) -> SchedStats:
        self.run_duration_ns = duration_ns
        for t in tasks:
            self.add_task(t)

        return self._run_until(duration_ns)

    def _collect_stats(self) -> SchedStats:
        done = self.completed
        observed = [task for task in self.all_tasks if task.arrival_ns <= self.clock]
        s = SchedStats()
        s.algo_name = run_label(self.policy, self.decision_mode, self.decision_source_name)
        s.tasks_completed = len(done)
        if observed:
            turnaround_values = [self._effective_turnaround_ns(task) for task in observed]
            wait_values = [self._effective_wait_ns(task) for task in observed]
            response_values = [self._effective_response_ns(task) for task in observed]
            s.avg_turnaround_ms = sum(turnaround_values) / len(observed) / NS_PER_MS
            s.avg_wait_ms = sum(wait_values) / len(observed) / NS_PER_MS
            s.avg_response_ms = sum(response_values) / len(observed) / NS_PER_MS
            ta_vals = [value for value in turnaround_values if value > 0]
            if len(ta_vals) > 1:
                sx = sum(ta_vals)
                sx2 = sum(x * x for x in ta_vals)
                s.fairness_index = sx*sx / (len(ta_vals) * sx2) if sx2 > 0 else 1.0
            else:
                s.fairness_index = 1.0
            starvation_thresh = 500 * NS_PER_MS
            s.starvation_count = sum(1 for value in wait_values if value > starvation_thresh)
        s.cpu_utilization = self.cpu_busy_ns / self.clock if self.clock > 0 else 0
        s.throughput = len(done) * 1e11 / self.clock if self.clock > 0 else 0
        s.ctx_switches = self.nr_ctx_switches
        return s


def prepare_tasks_for_run(tasks: List[Task], policy: int) -> List[Task]:
    clones = [copy.deepcopy(task) for task in tasks]
    for task in clones:
        if policy == SCHED_NN:
            if not is_rt_prio(task.prio):
                task.set_policy(SCHED_NN, task.nice, task.rt_priority)
        elif policy != SCHED_NORMAL and task.policy in (
            SCHED_NORMAL,
            SCHED_BATCH,
            SCHED_IDLE,
            SCHED_NN,
        ):
            task.set_policy(policy, task.nice, task.rt_priority)
    return clones


def run_one(
    tasks: List[Task],
    policy: int,
    duration_ns: int,
    verbose: bool,
    cpu_id: int = 0,
    seed: Optional[int] = None,
    model_policy: Optional[SchedulerModelPolicy] = None,
    decision_mode: str = "baseline",
    decision_source_name: str = "model",
    record_decisions: bool = False,
    decision_teacher: str = "cfs",
    max_history: int = 256,
) -> Tuple[SchedStats, Scheduler]:
    if seed is not None:
        random.seed(seed)

    clones = prepare_tasks_for_run(tasks, policy)
    sched = Scheduler(
        policy=policy,
        verbose=verbose,
        cpu_id=cpu_id,
        model_policy=model_policy,
        decision_mode=decision_mode,
        decision_source_name=decision_source_name,
        record_decisions=record_decisions,
        decision_teacher=decision_teacher,
        max_history=max_history,
    )
    stats = sched.run(clones, duration_ns)
    stats.algo_name = run_label(policy, decision_mode, decision_source_name)
    return stats, sched


def run_one_to_completion(
    tasks: List[Task],
    policy: int,
    verbose: bool,
    cpu_id: int = 0,
    seed: Optional[int] = None,
    model_policy: Optional[SchedulerModelPolicy] = None,
    decision_mode: str = "baseline",
    decision_source_name: str = "model",
    record_decisions: bool = False,
    decision_teacher: str = "cfs",
    max_history: int = 256,
) -> Tuple[SchedStats, Scheduler]:
    if seed is not None:
        random.seed(seed)

    clones = prepare_tasks_for_run(tasks, policy)
    sched = Scheduler(
        policy=policy,
        verbose=verbose,
        cpu_id=cpu_id,
        model_policy=model_policy,
        decision_mode=decision_mode,
        decision_source_name=decision_source_name,
        record_decisions=record_decisions,
        decision_teacher=decision_teacher,
        max_history=max_history,
    )
    for task in clones:
        sched.add_task(task)
    stats = sched._run_until_complete()
    stats.algo_name = run_label(policy, decision_mode, decision_source_name)
    return stats, sched


def task_to_dict(task: Task, clock_ns: int) -> dict:
    deadline_ms = None
    if task.deadline_ns:
        deadline_ms = max(0, task.deadline_ns - task.arrival_ns) // NS_PER_MS

    turnaround_ns = effective_turnaround_ns(task, clock_ns)
    wait_ns = effective_wait_ns(task, clock_ns)
    response_ns = effective_response_ns(task, clock_ns)

    return {
        "pid": task.pid,
        "name": task.name,
        "policy": policy_name(task.policy),
        "nice": task.nice,
        "rt_priority": task.rt_priority,
        "arrival_ms": task.arrival_ns // NS_PER_MS,
        "burst_total_ms": task.burst_total_ns // NS_PER_MS,
        "io_ratio": task.io_ratio,
        "deadline_ms": deadline_ms,
        "state": task.state,
        "completed": task.state == "dead",
        "turnaround_ms": turnaround_ns // NS_PER_MS,
        "wait_ms": wait_ns // NS_PER_MS,
        "response_ms": response_ns // NS_PER_MS,
        "cpu_time_ms": task.utime_ns // NS_PER_MS,
        "ctx_switches": task.ctx_switches,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def write_synthetic_trace(
    path: Path,
    sched: Scheduler,
    *,
    taskfile: str,
    policy: str,
    duration_ms: int,
    seed: Optional[int],
    append: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_has_content = path.exists() and path.stat().st_size > 0
    mode = "a" if append else "w"
    with path.open(mode) as fh:
        if append and file_has_content:
            fh.write("\n")
        fh.write("# synthetic sched_switch trace generated by emulator/sched_em.py\n")
        fh.write(
            f"# taskfile={taskfile} policy={policy} duration_ms={duration_ms} "
            f"seed={seed} cpu_id={sched.cpu_id} events={len(sched.trace_events)}\n"
        )
        for event in sched.trace_events:
            timestamp_s = event.timestamp_ns / 1_000_000_000
            fh.write(
                f"emulator-{event.cpu_id:03d} [{event.cpu_id:03d}] {timestamp_s:12.6f}: "
                "sched:sched_switch: "
                f"prev_comm={event.prev_name} prev_pid={event.prev_pid} "
                f"prev_prio={event.prev_prio} ==> "
                f"next_comm={event.next_name} next_pid={event.next_pid} "
                f"next_prio={event.next_prio}\n"
            )


def build_single_run_report(
    *,
    args: argparse.Namespace,
    stats: SchedStats,
    sched: Scheduler,
    wall_clock_ms: float,
    taskfile: str,
    baseline_stats: Optional[SchedStats] = None,
    full_workload_stats: Optional[SchedStats] = None,
    full_workload_baseline_stats: Optional[SchedStats] = None,
) -> dict:
    report = {
        "compare": False,
        "taskfile": taskfile,
        "policy": stats.algo_name,
        "duration_ms": args.duration,
        "seed": args.seed,
        "cpu_id": args.cpu_id,
        "wall_clock_ms": wall_clock_ms,
        "trace_event_count": len(sched.trace_events),
        "model_mode": args.model_mode,
        "model_run_dir": args.model_run_dir,
        "heuristic_policy": args.heuristic_policy,
        "stats": asdict(stats),
        "model_summary": summarize_model_events(sched.model_events),
        "tasks": [task_to_dict(task, sched.clock) for task in sched.all_tasks],
    }
    if sched.shadow_records:
        report["shadow_summary"] = summarize_shadow_records(sched.shadow_records)
        report["shadow_records"] = sched.shadow_records
    if baseline_stats is not None:
        report["baseline_cfs_stats"] = asdict(baseline_stats)
        report["delta_vs_cfs"] = stats_delta(stats, baseline_stats)
    if full_workload_stats is not None:
        report["full_workload_stats"] = asdict(full_workload_stats)
    if full_workload_baseline_stats is not None:
        report["full_workload_baseline_cfs_stats"] = asdict(full_workload_baseline_stats)
        report["full_workload_delta_vs_cfs"] = stats_delta(
            full_workload_stats,
            full_workload_baseline_stats,
        )
    return report


def build_compare_report(
    *,
    args: argparse.Namespace,
    taskfile: str,
    benchmark_rows: List[dict],
) -> dict:
    return {
        "compare": True,
        "taskfile": taskfile,
        "duration_ms": args.duration,
        "seed": args.seed,
        "results": benchmark_rows,
    }


def print_delta_summary(label: str, delta: dict) -> None:
    print()
    print(f"  -- {label} --")
    print(f"  turnaround delta: {delta['avg_turnaround_ms']:+.3f} ms")
    print(f"  wait delta: {delta['avg_wait_ms']:+.3f} ms")
    print(f"  response delta: {delta['avg_response_ms']:+.3f} ms")
    print(f"  fairness delta: {delta['fairness_index']:+.6f}")
    print(f"  ctx_switch delta: {delta['ctx_switches']:+d}")
    print(f"  starvation delta: {delta['starvation_count']:+d}")


def print_shadow_summary(shadow_summary: dict) -> None:
    print()
    print("  -- Shadow summary --")
    print(f"  divergences evaluated: {shadow_summary['divergences']}")
    for metric, values in shadow_summary["metrics"].items():
        print(
            f"  {metric}: mean_delta={values['mean_delta']:+.6f} "
            f"better={values['better']} worse={values['worse']} tied={values['tied']}"
        )

def main():
    parser = generate_parser()
    args = parser.parse_args()
    if args.append_trace and not args.trace_out:
        parser.error("--append-trace requires --trace-out")
    if args.compare and args.trace_out:
        parser.error("--trace-out requires a single-policy run; remove --compare")
    if args.compare and args.model_mode != "none":
        parser.error("--model-mode is only supported in single-policy runs")
    if args.heuristic_policy != "none" and args.model_mode == "none":
        parser.error("--heuristic-policy requires --model-mode shadow or closed-loop")
    if args.cpu_id < 0:
        parser.error("--cpu-id must be >= 0")
    if args.cpu_id > 999:
        parser.error("--cpu-id must be <= 999 to stay compatible with the trace parser")

    if args.seed is not None:
        random.seed(args.seed)

    print("Kernel tunables (defaults):")
    print(f"  sched_latency_ns = {SCHED_LATENCY_NS:,} ({SCHED_LATENCY_NS/NS_PER_MS:.1f} ms)")
    print(f"  sched_min_granularity_ns = {SCHED_MIN_GRAN_NS:,} ({SCHED_MIN_GRAN_NS/NS_PER_MS:.3f} ms)")
    print(f"  sched_wakeup_granularity_ns = {SCHED_WAKEUP_GRAN_NS:,} ({SCHED_WAKEUP_GRAN_NS/NS_PER_MS:.3f} ms)")
    print(f"  Simulation duration = {args.duration} ms")
    if args.seed is not None:
        print(f"  Random seed = {args.seed}")
    print()

    if not os.path.exists(args.taskfile):
        print(f"Error: {args.taskfile} not found", file=sys.stderr)
        sys.exit(1)
    tasks = parse_taskfile(args.taskfile)
    print(f"  Loaded {len(tasks)} tasks from {args.taskfile}")
    print()
    print_task_table(tasks)

    duration_ns = args.duration * NS_PER_MS
    policy = POLICY_MAP.get(args.policy.upper(), SCHED_NORMAL)
    decision_mode = "baseline" if args.model_mode == "none" else args.model_mode.replace("-", "_")
    if policy == SCHED_NN:
        print("  Warning: SCHED_NN is still an experimental placeholder in this emulator.")
        print("           Use it as a benchmark label or synthetic-trace source, not as a trained policy.")
        print()
    if args.model_mode != "none" and policy != SCHED_NORMAL:
        parser.error("--model-mode currently supports only the CFS base policy")

    model_policy = None
    decision_source_name = "model"
    if args.model_mode != "none":
        if args.heuristic_policy != "none":
            if args.heuristic_policy != "metric":
                parser.error(f"Unsupported heuristic policy: {args.heuristic_policy}")
            model_policy = MetricHeuristicPolicy()
            decision_source_name = "heuristic"
        else:
            if not args.model_run_dir:
                parser.error("--model-run-dir is required when --model-mode is enabled")
            try:
                model_policy = SchedulerModelPolicy(args.model_run_dir)
            except Exception as exc:
                parser.error(str(exc))
        print(f"  Model mode = {args.model_mode}")
        if args.heuristic_policy != "none":
            print(f"  Heuristic policy = {args.heuristic_policy}")
        else:
            print(f"  Model run dir = {args.model_run_dir}")
        print()

    if not args.compare:
        t0 = time.perf_counter()
        stats, sched = run_one(
            tasks,
            policy,
            duration_ns,
            args.verbose,
            cpu_id=args.cpu_id,
            seed=args.seed,
            model_policy=model_policy,
            decision_mode=decision_mode,
            decision_source_name=decision_source_name,
        )
        elapsed = time.perf_counter() - t0
        baseline_stats = None
        delta_vs_cfs = None
        full_workload_stats = None
        full_workload_baseline_stats = None
        full_workload_delta_vs_cfs = None
        if args.model_mode == "closed-loop":
            baseline_stats, _ = run_one(
                tasks,
                SCHED_NORMAL,
                duration_ns,
                False,
                cpu_id=args.cpu_id,
                seed=args.seed,
            )
            delta_vs_cfs = stats_delta(stats, baseline_stats)
            full_workload_stats, _ = run_one_to_completion(
                tasks,
                SCHED_NORMAL,
                False,
                cpu_id=args.cpu_id,
                seed=args.seed,
                model_policy=model_policy,
                decision_mode=decision_mode,
                decision_source_name=decision_source_name,
            )
            full_workload_baseline_stats, _ = run_one_to_completion(
                tasks,
                SCHED_NORMAL,
                False,
                cpu_id=args.cpu_id,
                seed=args.seed,
            )
            full_workload_delta_vs_cfs = stats_delta(
                full_workload_stats,
                full_workload_baseline_stats,
            )

        print(f"   --- Results: {stats.algo_name} ---  (sim={elapsed*1000:.1f}ms wall-clock)")
        print_stats_header()
        print_stat_row(stats, highlight=True)
        print()
        print_per_task(sched.all_tasks, sched.clock)

        if args.model_mode != "none":
            model_summary = summarize_model_events(sched.model_events)
            print()
            print("  -- Decision policy --")
            print(f"  opportunities: {model_summary['opportunities']}")
            print(f"  same as CFS: {model_summary['same_as_cfs']}")
            print(f"  different from CFS: {model_summary['different_from_cfs']}")
            print(f"  selected candidate: {model_summary['selected_candidate']}")
            print(f"  applied: {model_summary['applied_predictions']}")
            print(f"  fallback to CFS: {model_summary['fallback_to_cfs']}")
            print(f"  no candidates: {model_summary['no_candidates']}")

        if sched.shadow_records:
            print_shadow_summary(summarize_shadow_records(sched.shadow_records))

        if delta_vs_cfs is not None:
            print_delta_summary("Closed-loop vs CFS (same duration)", delta_vs_cfs)

        if full_workload_delta_vs_cfs is not None:
            print_delta_summary(
                "Closed-loop vs CFS (to completion)",
                full_workload_delta_vs_cfs,
            )

        if args.gantt:
            print_gantt(sched.all_tasks, sched.gantt, sched.clock)

        if args.trace_out:
            trace_path = Path(args.trace_out)
            write_synthetic_trace(
                trace_path,
                sched,
                taskfile=args.taskfile,
                policy=stats.algo_name,
                duration_ms=args.duration,
                seed=args.seed,
                append=args.append_trace,
            )
            print(f"\n  Synthetic trace exported to {trace_path}")

        if args.stats_json:
            stats_path = Path(args.stats_json)
            report = build_single_run_report(
                args=args,
                stats=stats,
                sched=sched,
                wall_clock_ms=elapsed * 1000,
                taskfile=args.taskfile,
                baseline_stats=baseline_stats,
                full_workload_stats=full_workload_stats,
                full_workload_baseline_stats=full_workload_baseline_stats,
            )
            write_json(stats_path, report)
            print(f"  Benchmark report exported to {stats_path}")


    else:
        policies = [
            (SCHED_NORMAL, "CFS (SCHED_NORMAL)"),
            (SCHED_BATCH, "SCHED_BATCH"),
            (SCHED_FIFO, "SCHED_FIFO (RT)"),
            (SCHED_RR, "SCHED_RR (RT)"),
            (SCHED_IDLE, "SCHED_IDLE")
        ]

        print(" --- Comparative Analysis: all algorithms on same workload ---")
        print()
        results = []
        scheds = []
        benchmark_rows = []
        for pol, label in policies:
            t0 = time.perf_counter()
            s, sched = run_one(
                tasks,
                pol,
                duration_ns,
                False,
                cpu_id=args.cpu_id,
                seed=args.seed,
                decision_mode="baseline",
            )
            elapsed = time.perf_counter() - t0
            s.algo_name = label
            results.append(s)
            scheds.append(sched)
            benchmark_rows.append(
                {
                    "policy": label,
                    "wall_clock_ms": elapsed * 1000,
                    "stats": asdict(s),
                }
            )
            print(f"  [{len(results)}/{len(policies)}] {label:<28}  "
                  f"done ({s.tasks_completed} tasks)  [{elapsed*1000:.0f}ms wall]")

        print()
        print_stats_header()

        best_ta = min(r.avg_turnaround_ms for r in results)
        best_wait = min(r.avg_wait_ms for r in results)
        best_cpu = max(r.cpu_utilization for r in results)
        best_fair = max(r.fairness_index for r in results)
        best_ctxsw = min(r.ctx_switches for r in results)

        for r in results:
            wins = (abs(r.avg_turnaround_ms - best_ta) < 0.5 or
                    abs(r.avg_wait_ms - best_wait) < 0.5 or
                    abs(r.cpu_utilization - best_cpu) < 0.001 or
                    abs(r.fairness_index - best_fair) == 0 or
                    abs(r.ctx_switches - best_ctxsw) < 0.001)
            print_stat_row(r, highlight=wins)

        print()
        print("  -- Best performers --")
        for r in results:
            medals = []
            if abs(r.avg_turnaround_ms - best_ta) < 0.5:
                medals.append("[turnaround]")
            if abs(r.avg_wait_ms - best_wait) < 0.5:
                medals.append("[wait_time]")
            if abs(r.cpu_utilization - best_cpu) < 0.001:
                medals.append("[cpu_util]")
            if abs(r.fairness_index - best_fair) < 0.001:
                medals.append("[fairness]")
            if abs(r.ctx_switches - best_ctxsw) == 0:
                medals.append("[ctx_switch]")
            if medals:
                print(f" * {r.algo_name:<28} wins: {' '.join(medals)}")

        if args.gantt:
            print_gantt(scheds[0].all_tasks, scheds[0].gantt, scheds[0].clock)

        if args.stats_json:
            stats_path = Path(args.stats_json)
            report = build_compare_report(
                args=args,
                taskfile=args.taskfile,
                benchmark_rows=benchmark_rows,
            )
            write_json(stats_path, report)
            print(f"\n  Benchmark report exported to {stats_path}")

    print()
    print("  ------------------------")
    print("  Simulation complete.")
    print()


if __name__ == "__main__":
    main()
