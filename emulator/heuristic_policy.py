import math

from decision_features import TASK_FEATURE_NAMES
from model_policy import ModelSelection


_IDX = {name: index for index, name in enumerate(TASK_FEATURE_NAMES)}


def _exp_ms(value: float) -> float:
    return math.expm1(value)


def _signed_exp_ms(value: float) -> float:
    if value == 0.0:
        return 0.0
    return math.copysign(math.expm1(abs(value)), value)


class MetricHeuristicPolicy:
    """Small metric-aware reranker for CFS candidates."""

    def __init__(
        self,
        *,
        response_threshold_ms: float = 12.0,
        wait_weight: float = 1.5,
        new_wait_weight: float = 2.0,
        remaining_penalty: float = 0.004,
        vruntime_penalty: float = 0.25,
        deadline_bonus: float = 10.0,
        switch_margin: float = 0.6,
        keep_current_ms: float = 10.0,
    ) -> None:
        self.response_threshold_ms = response_threshold_ms
        self.wait_weight = wait_weight
        self.new_wait_weight = new_wait_weight
        self.remaining_penalty = remaining_penalty
        self.vruntime_penalty = vruntime_penalty
        self.deadline_bonus = deadline_bonus
        self.switch_margin = switch_margin
        self.keep_current_ms = keep_current_ms

    def _score_task(self, features: list, remaining_ms: float) -> float:
        wait_ms = _exp_ms(features[_IDX["wait_log_ms"]])
        executed_ms = _exp_ms(features[_IDX["executed_log_ms"]])
        vruntime_delta_ms = _signed_exp_ms(features[_IDX["vruntime_delta_signed_log_ms"]])
        deadline_slack_ms = _signed_exp_ms(features[_IDX["deadline_slack_signed_log_ms"]])
        has_deadline = features[_IDX["has_deadline"]] > 0.5
        is_new = executed_ms < 0.5

        deadline_score = 0.0
        if has_deadline:
            if deadline_slack_ms >= 0.0:
                deadline_score = self.deadline_bonus / max(deadline_slack_ms + 25.0, 5.0)
            else:
                deadline_score = 6.0

        return (
            self.wait_weight * wait_ms / max(remaining_ms + 10.0, 1.0)
            + self.new_wait_weight * float(is_new) * wait_ms / max(remaining_ms + 20.0, 1.0)
            + deadline_score
            - self.remaining_penalty * remaining_ms
            - self.vruntime_penalty * max(vruntime_delta_ms, 0.0)
        )

    def score_candidates(self, decision_state: dict) -> tuple[list[float], int]:
        candidate_features = decision_state["candidate_features"]
        if not candidate_features:
            return [], 0

        scores: list[float] = []
        remaining_times: list[float] = []
        response_times: list[float] = []
        current_index = None
        urgent_index = None
        urgent_score = None
        best_index = 0
        best_score = None

        for index, features in enumerate(candidate_features):
            remaining_ms = _exp_ms(features[_IDX["burst_remaining_log_ms"]])
            response_ms = _exp_ms(features[_IDX["response_log_ms"]])
            executed_ms = _exp_ms(features[_IDX["executed_log_ms"]])
            is_current = features[_IDX["is_current"]] > 0.5
            is_new = executed_ms < 0.5

            if is_current:
                current_index = index

            if is_new and response_ms >= self.response_threshold_ms:
                response_ratio = response_ms / max(remaining_ms + 8.0, 1.0)
                if urgent_score is None or response_ratio > urgent_score:
                    urgent_score = response_ratio
                    urgent_index = index

            score = self._score_task(features, remaining_ms)
            scores.append(score)
            remaining_times.append(remaining_ms)
            response_times.append(response_ms)

            if best_score is None or score > best_score:
                best_score = score
                best_index = index

        if urgent_index is not None and urgent_score is not None:
            current_has_low_response = (
                current_index is None
                or response_times[current_index] < self.response_threshold_ms
            )
            if current_has_low_response:
                scores[urgent_index] = max(scores[urgent_index], best_score + max(urgent_score, 1.0))
                best_index = urgent_index

        if current_index is not None and best_index != current_index:
            if remaining_times[current_index] <= self.keep_current_ms:
                scores[current_index] = max(scores[current_index], scores[best_index] + self.switch_margin)
                best_index = current_index
            elif scores[best_index] <= scores[current_index] + self.switch_margin:
                best_index = current_index

        return scores, best_index

    def predict(self, decision_state: dict) -> ModelSelection:
        candidate_tasks = decision_state["candidate_tasks"]
        if not candidate_tasks:
            return ModelSelection(None, None, None, "no_candidates")

        _, best_index = self.score_candidates(decision_state)
        chosen_task = candidate_tasks[best_index]
        return ModelSelection(
            selected_task_name=chosen_task["name"],
            selected_task_pid=chosen_task["pid"],
            confidence=None,
            reason="metric_heuristic",
        )
