"""Opt-in continuous dense-tree skill orchestration; no simulator on import.

The same environment and mounted verifier survive every target. Stow commands
are ordinary bounded delta joint targets, not reset/pose writes. Physical task
success belongs to MountedSingleFruitTask, never a proximity/reward proxy.
"""
from dataclasses import replace
import math

from .harvest_cycle import Phase, PhysicalEvidence
from .harvest_cycle_adapter import (MeasuredBasketEnvView, MountedSingleFruitTask,
                                    legacy_hybrid_factory)
from .mounted_basket_verifier import rotate, BasketVerification


def bounded_stow_delta(actual, targets, home, *, scale, max_delta=.02):
    """Command toward real home; completion checks measured joints, not commands."""
    if len(actual) != 14 or len(targets) != 14 or len(home) != 14 or scale <= 0:
        raise ValueError('Require exact14 delta-joint state')
    if any(not math.isfinite(x) for x in (*actual, *targets, *home)):
        raise ValueError('Non-finite stow state')
    return tuple(max(-max_delta, min(max_delta, h-t))/scale for h,t in zip(home, targets))


class SlotBasketView(MeasuredBasketEnvView):
    """IK receives a genuine unoccupied floor point in measured basket frame.

    PPO observations are built from a separate centre-frame facade so its
    original apple-to-basket semantics are unchanged.
    """
    slot_b_m = (0., 0., 0.)

    @property
    def basket_pos(self):
        center = super().basket_pos
        offset = rotate(self._frame.basket.orientation_xyzw, self.slot_b_m)
        return center+center.new_tensor([offset])


class DenseHarvestCycleAdapter:
    def __init__(self, env, policy, cycle, verifier, *, target_order, reader,
                 observation_builder, slot_selector, hybrid_factory=None,
                 stow_reader=None, max_pick_steps=900, max_stow_steps=300):
        order = tuple(target_order)
        if (not order or set(order) != set(cycle.targets) or len(order) != len(set(order))
                or env.num_envs != 1 or env.num_controlled_joints != 14
                or env.cfg.action_mode != 'delta' or not env._robot.is_fixed_base
                or policy.observation_dim != (82 if getattr(policy,'geometry10',False) else 72) or policy.action_dim != 14
                or not set(order) <= set(verifier.fruits)):
            raise ValueError('Exact frozen target order/inventory and declared72or82/14 contract required')
        self.env, self.policy, self.cycle, self.verifier = env, policy, cycle, verifier
        self.order, self.reader, self.observation_builder = order, reader, observation_builder
        self.slot_selector = slot_selector
        self.factory = hybrid_factory or legacy_hybrid_factory
        self.stow_reader = stow_reader or self._native_stow
        self.max_pick_steps, self.max_stow_steps = max_pick_steps, max_stow_steps
        self.last_sample = -1
        self.last_result = self.last_decision = None
        self.stopped = self.named_tree_success = False
        self.two_pick_checkpoint = None
        self.pick_start = self.stow_start = None
        self.initial_actions = None
        self._select_next()

    def _select_next(self):
        target = next((t for t in self.order if t not in self.cycle.deposited), None)
        if target is None:
            self.named_tree_success = (self.cycle.tree_complete(self.cycle.tree)
                                       and self.cycle.deposited <= self.last_result.retained_ids)
            self.stopped = True
            return
        self.env.select_target(target)
        # Admission is explicit in the frozen runtime packet. Diagnostic targets
        # may remain unresolved, in which case the runner cannot claim eligible
        # or all-reachable success from the result.
        self.cycle.select(target, admitted=True, stance_stable=True)
        self.task = MountedSingleFruitTask(target)
        self.slot = None
        self.view, self.observation_view = SlotBasketView(self.env), MeasuredBasketEnvView(self.env)
        self.skill = self.factory(self.view, self.policy)
        self.pick_start = int(self.env.episode_length_buf[0])

    def _stop(self, reason):
        if self.cycle.active is not None:
            self.cycle.fail(reason)
        else:
            self.cycle._payload_failure(self.last_result.retained_ids if self.last_result else (), reason)
        self.stopped = True

    def action(self):
        if self.stopped:
            raise ValueError('No action after stop; no automatic retry')
        try:
            frame, signals = self.reader(self.env)
            if frame.sample <= self.last_sample:
                raise ValueError('Repeated/rolled-back physical clock; no reset allowed')
            self.last_sample = frame.sample
            if not frame.calibrated_contacts:
                # First integrated scene must establish raw-report sign/identity
                # against a positive calibrated finger sensor response. Ordinary
                # PPO acquisition may approach; no named grasp/deposit credit,
                # detach, prior payload or verification may pass this gate.
                if (self.cycle.deposited or not signals.attached or signals.failure_code
                        or signals.unsafe_collision or signals.fallen
                        or signals.extra_apple_assistance or signals.time_limit_reached
                        or frame.sample-self.pick_start >= self.max_pick_steps):
                    self._stop('unsafe_collision_before_contact_mapping' if signals.unsafe_collision
                               else 'contact_mapping_unvalidated_before_detachment_or_failure')
                    return None
                result = BasketVerification(frame.sample, frame.time_s, frame.basket_id,
                    False, ('awaiting_positive_native_contact_mapping',), (), frame.calibration_reference)
                decision = type('PendingContactDecision', (), {'evidence': PhysicalEvidence(
                    frame.sample, self.cycle.active, calibrated_contacts=False)})()
                self.last_result = result
                self.view.refresh(frame, result, decision)
                self.observation_view.refresh(frame, result, decision)
                return self.skill(self.observation_builder(self.observation_view))
            frame = replace(frame, deposited_target_ids=tuple(sorted(self.cycle.deposited)))
            result = self.verifier.observe(frame, now_s=signals.time_s)
            self.last_result = result
            if (not result.valid or not self.cycle.deposited <= result.retained_ids
                    or signals.unsafe_collision or signals.fallen or signals.extra_apple_assistance
                    or signals.time_limit_reached):
                self._stop('dense_safety_retention_or_timeout')
                return None
            if self.cycle.phase == Phase.STOW:
                command, measured_stowed, no_held = self.stow_reader(self.env, frame, result)
                if frame.sample-self.stow_start > self.max_stow_steps:
                    self._stop('physical_stow_timeout')
                    return None
                if measured_stowed and no_held:
                    self.cycle.stow_complete(arm_stowed=True, no_held_apple=True,
                                             retained_ids=result.retained_ids & self.cycle.deposited)
                    self._select_next()
                    # Target switched at the SAME physical sample; consume no
                    # stale target telemetry. This hold is an ordinary action,
                    # preserving executed history before the next measured read.
                    return None if self.stopped else command*0
                return command
            if frame.sample-self.pick_start >= self.max_pick_steps:
                self._stop('per_pick_timeout')
                return None
            decision = self.task.observe(signals, frame, result)
            self.last_decision = decision
            if decision.stop_reason:
                self._stop(decision.stop_reason)
                return None
            self.view.refresh(frame, result, decision)
            self.observation_view.refresh(frame, result, decision)
            if self.slot is None:
                slot = self.slot_selector(frame, result, self.cycle.active)
                if slot is None:
                    self._stop('mounted_basket_full_no_clear_deposit_slot')
                    self.cycle.phase = Phase.FULL
                    return None
                self.slot = tuple(slot)
            self.view.slot_b_m = self.slot
            self.cycle.observe(decision.evidence)
            if self.cycle.phase == Phase.REVIEW:
                self.stopped = True
                return None
            if self.cycle.phase == Phase.STOW:
                self.stow_start = frame.sample
                if len(self.cycle.deposited) == 2 and self.two_pick_checkpoint is None:
                    self.two_pick_checkpoint = dict(sample=frame.sample, retained_ids=sorted(self.cycle.deposited))
                return self.stow_reader(self.env, frame, result)[0]
            return self.skill(self.observation_builder(self.observation_view))
        except Exception as exc:
            if not self.stopped:
                self._stop(f'dense_adapter_error:{type(exc).__name__}:{exc}')
            raise

    @staticmethod
    def _native_stow(env, frame, result):
        from .compat import as_torch
        robot = env._robot.data
        measured = as_torch(robot.joint_pos)[:, env.ctrl_joint_ids]
        velocities = as_torch(robot.joint_vel)[:, env.ctrl_joint_ids]
        home = as_torch(robot.default_joint_pos)[:, env.ctrl_joint_ids]
        # Command history, not a fabricated home/reset target.
        action = (home-env._joint_targets).clamp(-.02, .02)/env.cfg.action_scale
        action = action.clamp(-env.cfg.clip_actions, env.cfg.clip_actions)
        stowed = bool((measured-home).abs().max() < .04 and velocities.abs().max() < .10)
        held = any(r.held for r in frame.releases)
        # Attached neighbouring fruit are intentionally held by their OWN stem;
        # reader.held includes attached. Only detached fruit can remain in hand.
        by_id = {s['target_id']: s for s in env.dense_fruit_inventory()}
        held = any(r.held and not by_id[r.target_id]['attached'] for r in frame.releases)
        return action, stowed, not held
