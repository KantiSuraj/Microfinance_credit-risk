"""
episode_logger.py — Structured episode sampling and pattern detection.

Strategy 6: Logging + Sampling
═════════════════════════════
Logs every Nth episode and provides pattern detection to catch:
  - Repeated action sequences across episodes
  - Suspiciously stable reward distributions
  - Always-same strategy detection
  - Degenerate Phase 2 behaviour
"""

from __future__ import annotations
import json
import collections
from typing import List, Optional, Dict


class EpisodeLogger:
    """
    Samples and logs episode trajectories for manual and automated inspection.

    Usage:
        logger = EpisodeLogger(sample_rate=10)

        # After each episode:
        logger.log_episode({
            "actions": [...],
            "final_decision": "APPROVE",
            "reward": 0.7,
            "num_steps": 4,
            "docs_collected": ["credit_history"],
            "phase2_actions": ["DO_NOTHING", "SEND_REMINDER", ...],
        })

        # Periodically check for exploit patterns:
        alerts = logger.detect_patterns()
    """

    def __init__(self, sample_rate: int = 10, max_stored: int = 500):
        """
        Args:
            sample_rate: Log every Nth episode. Set to 1 for full logging.
            max_stored: Maximum number of episodes to keep in memory.
        """
        self.sample_rate = max(1, sample_rate)
        self.max_stored = max_stored
        self.episode_count = 0
        self.sampled_episodes: List[dict] = []
        self._action_sequence_counter: Dict[str, int] = collections.defaultdict(int)
        self._decision_counter: Dict[str, int] = collections.defaultdict(int)
        self._reward_history: List[float] = []

    def log_episode(self, episode_data: dict) -> Optional[dict]:
        """
        Log an episode. Returns the logged data if sampled, None otherwise.

        episode_data expected keys:
            actions           : list of action strings
            final_decision    : "APPROVE" | "REJECT" | "TIMEOUT"
            reward            : float
            num_steps         : int
            docs_collected    : list of str
            phase2_actions    : list of str (Phase 2 action sequence)
            terminal_outcome  : "REPAID" | "DEFAULT" | "ESCALATED" | "TIMEOUT"
        """
        self.episode_count += 1
        reward = episode_data.get("reward", 0.0)
        self._reward_history.append(reward)

        # Track action sequences for pattern detection
        action_seq = tuple(episode_data.get("actions", []))
        self._action_sequence_counter[str(action_seq)] += 1

        # Track decisions
        decision = episode_data.get("final_decision", "UNKNOWN")
        self._decision_counter[decision] += 1

        # Sample
        if self.episode_count % self.sample_rate == 0:
            log_entry = {
                "episode_number": self.episode_count,
                **episode_data,
            }
            self.sampled_episodes.append(log_entry)
            # Trim if over max
            if len(self.sampled_episodes) > self.max_stored:
                self.sampled_episodes = self.sampled_episodes[-self.max_stored:]
            return log_entry

        return None

    def detect_patterns(self) -> dict:
        """
        Analyze logged episodes for suspicious patterns.

        Returns a dict of alerts:
            constant_action_sequence : bool — same action sequence > 60% of time
            constant_decision        : bool — same decision > 80% of time
            low_reward_variance      : bool — reward variance suspiciously low
            always_reject            : bool — rejects > 85% of time
            always_approve           : bool — approves > 85% of time
            description              : str  — human-readable summary
        """
        alerts = {}
        total = max(self.episode_count, 1)

        # Pattern 1: Constant action sequence (same sequence > 60% of time)
        if self._action_sequence_counter:
            most_common_seq, most_common_count = max(
                self._action_sequence_counter.items(), key=lambda x: x[1]
            )
            alerts["constant_action_sequence"] = (most_common_count / total) > 0.60
            alerts["most_common_sequence"] = most_common_seq
            alerts["sequence_dominance"] = round(most_common_count / total, 3)
        else:
            alerts["constant_action_sequence"] = False

        # Pattern 2: Constant decision (same decision > 80% of time)
        if self._decision_counter:
            most_common_dec, most_common_count = max(
                self._decision_counter.items(), key=lambda x: x[1]
            )
            alerts["constant_decision"] = (most_common_count / total) > 0.80
            alerts["decision_dominance"] = round(most_common_count / total, 3)
        else:
            alerts["constant_decision"] = False

        # Pattern 3: Low reward variance (suspiciously stable rewards)
        if len(self._reward_history) >= 10:
            mean_r = sum(self._reward_history) / len(self._reward_history)
            variance = sum((r - mean_r) ** 2 for r in self._reward_history) / len(self._reward_history)
            alerts["low_reward_variance"] = variance < 0.01
            alerts["reward_variance"] = round(variance, 4)
        else:
            alerts["low_reward_variance"] = False

        # Pattern 4: Always reject
        reject_rate = self._decision_counter.get("REJECT", 0) / total
        alerts["always_reject"] = reject_rate > 0.85

        # Pattern 5: Always approve
        approve_rate = self._decision_counter.get("APPROVE", 0) / total
        alerts["always_approve"] = approve_rate > 0.85

        # Summary
        flagged = [k for k, v in alerts.items()
                   if isinstance(v, bool) and v]
        if flagged:
            alerts["description"] = f"⚠ Suspicious patterns detected: {', '.join(flagged)}"
        else:
            alerts["description"] = "✓ No suspicious patterns detected."

        alerts["any_alert"] = len(flagged) > 0
        return alerts

    def get_summary(self) -> dict:
        """Return a summary of all logged data."""
        total = max(self.episode_count, 1)
        return {
            "total_episodes": self.episode_count,
            "sampled_episodes": len(self.sampled_episodes),
            "decision_distribution": dict(self._decision_counter),
            "unique_action_sequences": len(self._action_sequence_counter),
            "reward_mean": round(sum(self._reward_history) / max(len(self._reward_history), 1), 4),
            "reward_min": round(min(self._reward_history), 4) if self._reward_history else 0.0,
            "reward_max": round(max(self._reward_history), 4) if self._reward_history else 0.0,
        }

    def export_sampled(self) -> str:
        """Export sampled episodes as JSON string for external analysis."""
        return json.dumps(self.sampled_episodes, indent=2, default=str)
