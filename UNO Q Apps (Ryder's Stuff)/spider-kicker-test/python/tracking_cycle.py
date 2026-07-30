"""Pure state machine for the autonomous ball-fetch cycle - no camera/motor
I/O, unit-testable without the physical rig. main.py drives the hardware
(move_to()/kick()) in response to the Action this class returns from step().

Phases: SAMPLING -> LIFTING -> TRANSLATING -> DROPPING -> VERIFYING -> SETTLING -> KICKING
                                     ^                          |
                                     +---- (failed verify, retries left) ----+
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Action:
    kind: str  # "none" | "move" | "kick" | "reset"
    target: Optional[Tuple[float, float, float]] = None  # (x, y, z) mm, for kind == "move"
    reason: Optional[str] = None  # human-readable, for logging


class TrackingCycle:
    def __init__(self, sample_count, travel_height_mm, drop_height_mm,
                 verify_tolerance_mm, max_verify_retries, stale_timeout_s,
                 kicker_offset_mm=(0.0, 0.0), correction_gain=1.0, settle_time_s=0.0):
        self.sample_count = sample_count
        self.travel_height_mm = travel_height_mm
        self.drop_height_mm = drop_height_mm
        self.verify_tolerance_mm = verify_tolerance_mm
        self.max_verify_retries = max_verify_retries
        self.stale_timeout_s = stale_timeout_s
        self.kicker_offset_mm = kicker_offset_mm
        self.correction_gain = correction_gain
        self.settle_time_s = settle_time_s
        self.reset()

    def reset(self):
        self.phase = "SAMPLING"
        self.samples = []
        self.target = None  # (x, y) mm - the commanded reference-point target for this cycle
        self.verify_retry_count = 0
        self.last_verify_error_mm = None
        self.phase_entered_at = None

    def _enter(self, phase, now):
        self.phase = phase
        self.phase_entered_at = now

    def step(self, current_xy, ball_world, platform_world, now):
        """Call once per tracking_loop() iteration. Returns exactly one
        Action - never performs I/O itself; caller must actually perform it
        (move_to()/kick()) before the next step()."""
        if self.phase_entered_at is None:
            self._enter(self.phase, now)

        if self.phase == "SAMPLING":
            return self._step_sampling(ball_world, now)
        if self.phase == "LIFTING":
            self._enter("TRANSLATING", now)
            return Action("move", (current_xy[0], current_xy[1], self.travel_height_mm), "lift")
        if self.phase == "TRANSLATING":
            self._enter("DROPPING", now)
            tx, ty = self.target
            return Action("move", (tx, ty, self.travel_height_mm), "translate")
        if self.phase == "DROPPING":
            self._enter("VERIFYING", now)
            tx, ty = self.target
            return Action("move", (tx, ty, self.drop_height_mm), "drop")
        if self.phase == "VERIFYING":
            return self._step_verifying(platform_world, now)
        if self.phase == "SETTLING":
            return self._step_settling(now)
        if self.phase == "KICKING":
            self.reset()
            return Action("kick")
        raise AssertionError(f"unreachable phase {self.phase}")

    def _step_sampling(self, ball_world, now):
        if ball_world is None:
            return Action("none")
        self.samples.append(ball_world)
        if len(self.samples) < self.sample_count:
            return Action("none", reason=f"sample {len(self.samples)}/{self.sample_count} collected")
        xs = [p[0] for p in self.samples]
        ys = [p[1] for p in self.samples]
        avg_x, avg_y = sum(xs) / len(xs), sum(ys) / len(ys)
        self.target = (avg_x - self.kicker_offset_mm[0], avg_y - self.kicker_offset_mm[1])
        self.samples.clear()
        self._enter("LIFTING", now)
        return Action("none", reason=f"averaged {self.sample_count} samples -> target {self.target}")

    def _step_verifying(self, platform_world, now):
        if platform_world is None:
            if (now - self.phase_entered_at) > self.stale_timeout_s:
                return self._fail_verify("timed out waiting for a platform detection", now)
            return Action("none")
        tx, ty = self.target
        error_mm = math.hypot(platform_world[0] - tx, platform_world[1] - ty)
        self.last_verify_error_mm = round(error_mm, 1)
        if error_mm <= self.verify_tolerance_mm:
            self._enter("SETTLING", now)
            return Action("none", reason=f"verified, error {error_mm:.1f}mm - settling {self.settle_time_s}s before kick")
        corrected = (
            tx + (tx - platform_world[0]) * self.correction_gain,
            ty + (ty - platform_world[1]) * self.correction_gain,
        )
        return self._fail_verify(f"error {error_mm:.1f}mm > tolerance {self.verify_tolerance_mm}mm", now, corrected)

    def _step_settling(self, now):
        """Wait settle_time_s after landing before kicking - lets the
        cable-suspended platform's post-move swing damp out, so the kicker
        fires from a still platform rather than mid-oscillation."""
        if (now - self.phase_entered_at) < self.settle_time_s:
            return Action("none")
        self._enter("KICKING", now)
        return Action("none", reason="settled - ready to kick")

    def _fail_verify(self, reason, now, corrected_target=None):
        if self.verify_retry_count >= self.max_verify_retries:
            self.reset()
            return Action("reset", reason=f"verify retries exhausted ({reason})")
        self.verify_retry_count += 1
        if corrected_target is not None:
            self.target = corrected_target
        self._enter("LIFTING", now)  # re-lift before retrying, not a direct translate from DROP_HEIGHT_MM
        return Action("none", reason=f"verify retry {self.verify_retry_count}/{self.max_verify_retries}: {reason}")
