"""Opt-in parked single-fruit bridge to the unchanged HybridHarvestPolicy.

No simulator imports on load, environment steps, object writes, resets, training
or walking. A concrete mounted task supplies fresh physical telemetry. PPO sees
the measured basket-floor-surface centre, never the old stationary basket. The
legacy IK still uses its original world-axis routes/clearances; therefore this
adapter refuses moving/tilted baskets and does not claim gentle placement.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from .harvest_cycle import Phase, PhysicalEvidence
from .mounted_basket_verifier import PoseTwist, rotate, to_physical_evidence

TASK_NAME = "parked_mounted_single_fruit_harvest_v1"
HYBRID_FLAGS = dict(bounded_transfer=True, branch_aware_transfer=True,
                    upright_transfer=True, learned_grasp_governor=True)


@dataclass(frozen=True)
class ParkedTaskSignals:
    sample: int
    time_s: float
    target_id: str
    attached: bool
    sustained_grasp: bool
    opposing_grasp: bool
    carried: bool
    released: bool
    failure_code: int
    extra_apple_assistance: bool = False
    fallen: bool = False
    unsafe_collision: bool = False
    base_linear_speed_m_s: float = 0.
    base_angular_speed_rad_s: float = 0.
    time_limit_reached: bool = False


@dataclass(frozen=True)
class ParkedLimits:
    max_basket_speed_m_s: float = .005
    max_basket_angular_speed_rad_s: float = .01
    max_basket_tilt_rad: float = .02
    max_base_speed_m_s: float = .005
    max_base_angular_speed_rad_s: float = .01

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in vars(self).values()):
            raise ValueError("Invalid proposed parked-controller limit")


@dataclass(frozen=True)
class TaskDecision:
    evidence: PhysicalEvidence
    stop_reason: str | None
    task_name: str = TASK_NAME


def motion_at_frame_origin(*, time_s, position_w_m, orientation_xyzw,
                           com_linear_velocity_w_m_s, angular_velocity_w_rad_s,
                           center_of_mass_b_m):
    """Native rigid COM velocity -> basket floor-frame origin point velocity.

    Input pose is the body/link origin, exactly the authored basket B frame.
    Offset is authored/measured COM in B. No position or velocity is written.
    """
    offset = rotate(orientation_xyzw, center_of_mass_b_m)
    w = angular_velocity_w_rad_s
    correction = (w[1]*offset[2]-w[2]*offset[1], w[2]*offset[0]-w[0]*offset[2],
                  w[0]*offset[1]-w[1]*offset[0])
    velocity = tuple(v-d for v, d in zip(com_linear_velocity_w_m_s, correction))
    if len(velocity) != 3 or any(not math.isfinite(x) for x in (*velocity, *position_w_m, *w, time_s)):
        raise ValueError("Invalid native basket pose/twist")
    return PoseTwist(time_s, tuple(position_w_m), tuple(orientation_xyzw), velocity, tuple(w))


class MountedSingleFruitTask:
    """New named termination, separate from the stationary task's success flag.

    Requires measured sustained opposing acquisition, physical detachment with
    grip, native carried/released evidence, independent settled floor support,
    and no failure/assistance. This is a task predicate, not the verifier adapter
    inventing success. Runtime correctness remains unvalidated.
    """
    def __init__(self, target_id, *, limits=None):
        self.target_id, self.limits = target_id, limits or ParkedLimits()
        self.last_sample = -1
        self.grasp_seen = self.detached_seen = self.carried_seen = False
        self.done = False

    def observe(self, signals, frame, verification):
        if self.done:
            raise ValueError("Completed/stopped task cannot resume or reset itself")
        if (type(signals.sample) is not int or signals.sample <= self.last_sample
                or signals.target_id != self.target_id or signals.sample != frame.sample
                or signals.sample != verification.sample or signals.time_s != frame.time_s
                or verification.time_s != frame.time_s):
            raise ValueError("Wrong-target/stale/asynchronous task signals")
        self.last_sample = signals.sample
        flags = (signals.attached, signals.sustained_grasp, signals.opposing_grasp,
                 signals.carried, signals.released, signals.extra_apple_assistance,
                 signals.fallen, signals.unsafe_collision, signals.time_limit_reached)
        if any(type(x) is not bool for x in flags) or type(signals.failure_code) is not int:
            raise ValueError("Physical task flags must have explicit bool/int types")
        evidence = PhysicalEvidence(signals.sample, self.target_id,
            calibrated_contacts=verification.valid,
            opposing_grasp=signals.sustained_grasp and signals.opposing_grasp,
            physical_detachment_verified=not signals.attached,
            extra_apple_assistance=signals.extra_apple_assistance,
            released=signals.released, fallen=signals.fallen, unsafe_collision=signals.unsafe_collision)
        evidence = to_physical_evidence(evidence, verification)
        reason = None
        l = self.limits
        speeds = (signals.base_linear_speed_m_s, signals.base_angular_speed_rad_s)
        if any(not math.isfinite(v) or v < 0 for v in speeds):
            reason = "invalid_base_velocity"
        elif not verification.valid:
            reason = "invalid_mounted_contact_or_pose_evidence"
        elif signals.failure_code != 0 or signals.fallen or signals.unsafe_collision or signals.extra_apple_assistance:
            reason = "native_failure_or_unsafe_assistance"
        elif signals.time_limit_reached:
            reason = "mounted_task_timeout"
        elif any(f.spilled or "unobserved_retention_interval" in f.reasons for f in verification.fruits):
            reason = "payload_spill_or_unverified_retention"
        else:
            up = rotate(frame.basket.orientation_xyzw, (0., 0., 1.))
            tilt = math.acos(max(-1., min(1., up[2])))
            norm = lambda v: math.sqrt(sum(x*x for x in v))
            if (norm(frame.basket.linear_velocity_w_m_s) > l.max_basket_speed_m_s
                    or norm(frame.basket.angular_velocity_w_rad_s) > l.max_basket_angular_speed_rad_s
                    or tilt > l.max_basket_tilt_rad
                    or speeds[0] > l.max_base_speed_m_s or speeds[1] > l.max_base_angular_speed_rad_s):
                reason = "outside_parked_level_controller_contract"
        if not reason:
            self.grasp_seen |= signals.sustained_grasp and signals.opposing_grasp
            if signals.attached and self.detached_seen:
                reason = "fruit_reattached_or_world_reset"
            elif not signals.attached and not self.detached_seen:
                if not self.grasp_seen or not (signals.sustained_grasp and signals.opposing_grasp):
                    reason = "detached_without_verified_grasp"
                else:
                    self.detached_seen = True
            if not reason and signals.carried and not self.detached_seen:
                reason = "carry_without_verified_detachment"
            self.carried_seen |= signals.carried and self.detached_seen
        fruit = next((f for f in verification.fruits if f.target_id == self.target_id), None)
        success = (not reason and self.detached_seen and self.carried_seen and evidence.released
                   and fruit is not None and fruit.physical_deposit_verified
                   and evidence.inside_basket and evidence.support_chain_verified
                   and evidence.settled_relative_samples >= 30)
        evidence = replace(evidence, named_task_success=bool(success))
        self.done = bool(reason or success)
        return TaskDecision(evidence, reason)


def _list(value):
    return value.detach().cpu().tolist() if hasattr(value, "detach") else (
        value.tolist() if hasattr(value, "tolist") else list(value))


class MeasuredBasketEnvView:
    """Read-only Hybrid env facade. Writes/steps/reset remain with the real env.

    Native observation construction is called *on this facade*, so the actor and
    critic use the same fresh physical basket point as IK. The original env's
    stale _basket_pos and stationary success/support flags are never patched.
    """
    def __init__(self, env):
        self._env, self._frame, self._verification, self._decision = env, None, None, None
        counter = getattr(env, 'episode_length_buf', None)
        self.skill_start_sample = int(counter[0]) if counter is not None else 0

    def __getattr__(self, name):
        return getattr(self._env, name)

    def refresh(self, frame, verification, decision):
        self._frame, self._verification, self._decision = frame, verification, decision

    @property
    def episode_length_buf(self):
        # Hybrid's startup governor uses skill-relative step0. Physical samples,
        # task history and actor's executed-action history keep their real clocks.
        counter = self._env.episode_length_buf
        return counter-counter.new_tensor([self.skill_start_sample])

    @property
    def basket_pos(self):
        if self._frame is None:
            raise ValueError("No fresh measured basket pose")
        origin = _list(self._env.scene.env_origins[0])
        position = [x-y for x, y in zip(self._frame.basket.position_w_m, origin)]
        return self._env._apple_center.new_tensor([position])

    def get_state(self):
        if self._decision is None:
            raise ValueError("Mounted task has not observed this step")
        state = dict(self._env.get_state())
        f = next((f for f in self._verification.fruits if f.target_id == self._decision.evidence.target_id), None)
        values = dict(basket_supported=bool(f and f.support_chain_verified),
            apple_settled_in_basket=bool(f and f.physical_deposit_verified),
            placed_steps=self._decision.evidence.settled_relative_samples,
            released=self._decision.evidence.released, success=self._decision.evidence.named_task_success)
        for key, value in values.items():
            state[key] = state[key].new_tensor([value])
        return state


def legacy_hybrid_factory(env, learned_policy):
    # Call only after the explicitly approved concrete scene already exists.
    from .hybrid_policy import HybridHarvestPolicy
    return HybridHarvestPolicy(env, learned_policy, **HYBRID_FLAGS)


class HarvestCycleAdapter:
    """One parked fruit, one existing hybrid skill, no autonomous runtime loop.

    reader(env)->(BasketFrame, ParkedTaskSignals). observation_builder(view)
    uses the original actor observation function against the measured facade.
    A concrete task must prevent automatic world resets; rollback of episode
    counters stops this adapter. Caller stops stepping when action() returnsNone.
    """
    def __init__(self, env, learned_policy, cycle, verifier, *, reader,
                 observation_builder, hybrid_factory=None):
        if (env.num_envs != 1 or env.num_controlled_joints != 14 or env.cfg.action_mode != "delta"
                or not env._robot.is_fixed_base or learned_policy.observation_dim != 72
                or learned_policy.action_dim != 14):
            raise ValueError("Require parked original72/14 fixed-base delta-action contract")
        if getattr(env, "mounted_task_contract", None) != TASK_NAME:
            raise ValueError("Use the explicit no-auto-reset mounted task, not the legacy environment")
        if cycle.phase not in (Phase.MANIPULATE, Phase.VERIFY) or len(cycle.targets) != 1:
            raise ValueError("Require one admitted active fruit; stopped groups cannot run")
        if set(verifier.fruits) != {cycle.active}:
            raise ValueError("Cycle/verifier target inventory mismatch")
        self.env, self.cycle, self.verifier = env, cycle, verifier
        self.reader, self.observation_builder = reader, observation_builder
        self.view = MeasuredBasketEnvView(env)
        self.skill = (hybrid_factory or legacy_hybrid_factory)(self.view, learned_policy)
        self.task = MountedSingleFruitTask(cycle.active)
        self.last_sample = -1
        self.last_result = self.last_decision = None
        self.stopped = False

    def action(self):
        if self.stopped:
            raise ValueError("Adapter stopped; no automatic retry, world reset or new target")
        try:
            frame, signals = self.reader(self.env)
            if frame.sample <= self.last_sample:
                raise ValueError("Repeated/rolled-back episode step; no world reset allowed")
            self.last_sample = frame.sample
            result = self.verifier.observe(frame, now_s=signals.time_s)
            decision = self.task.observe(signals, frame, result)
            self.last_result, self.last_decision = result, decision
            self.view.refresh(frame, result, decision)
            if decision.stop_reason:
                self.cycle.fail(decision.stop_reason)
                self.stopped = True
                return None
            self.cycle.observe(decision.evidence)
            if self.cycle.phase in (Phase.STOW, Phase.REVIEW):
                self.stopped = True
                return None
            # No scripted reach, action remapping, extra filtering or body writes.
            return self.skill(self.observation_builder(self.view))
        except Exception as exc:
            if self.cycle.active is not None and self.cycle.phase in (Phase.MANIPULATE, Phase.VERIFY):
                self.cycle.fail(f"mounted_adapter_error: {type(exc).__name__}: {exc}")
            self.stopped = True
            raise
