"""Parity test: harvest_rl.task_logic vs harvest_isaac's original orchard logic.

Loads HarvestEvidence, opposing_contact and settled_in_basket straight from a harvest_isaac
checkout, drives them with randomized harvest attempts (successes, slips, drops, early
releases...) one env at a time, and checks the batched tracker gives identical results at
every step. No simulator needed:

    C:\\isaac\\env_isaaclab3\\Scripts\\python.exe C:\\isaac\\harvest_rl\\tests\\test_task_logic.py
"""

import importlib.util
import math
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harvest_rl.task_logic import FAILURE_NAMES, HarvestParams, HarvestTracker  # noqa: E402

HARVEST_ISAAC = Path(os.environ.get("HARVEST_ISAAC_DIR", r"C:\isaac\harvest_isaac"))
ORCHARD = HARVEST_ISAAC / "src/vla_isaaclab/envs/orchard_pick"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"orig_{name}", ORCHARD / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


interaction, layout_mod, contact_mod = _load("interaction"), _load("layout"), _load("contact")
FINGER_LINKS = contact_mod.FINGER_LINKS

# Values of the OrchardBench seed-42 task.
RADIUS, MASS, TARGET_Z = 0.03, 0.16, 1.0597111973013031
BASKET = (0.4258749684943297, -1.4960502731017878, 0.78)


class Original:
    """Scalar transcription of update_metrics (measured task) around the original helpers."""

    def __init__(self):
        self.evidence = interaction.HarvestEvidence()
        self.contact_steps = 0
        self.hold = 0
        self.failure = None
        self.released = False

    def step(self, detached, pos, speed, palm, finger_forces, floor_z, hand_joints, finite):
        hand_distance = math.dist(palm, pos)
        lifted = detached and pos[2] > TARGET_Z + 0.06 and hand_distance < 0.18
        contact = contact_mod.opposing_contact(finger_forces)
        self.contact_steps = self.contact_steps + 1 if contact else 0
        grasp_contact = self.contact_steps >= 5
        self.evidence.update(detached=detached, sustained_contact=grasp_contact, lift_candidate=lifted)
        if self.evidence.failure:
            self.failure = self.evidence.failure
        supported = floor_z > MASS * 9.81 * 0.4
        opened = max(abs(v) for v in hand_joints) < 0.30
        self.released = self.evidence.released(opened=opened, contact=contact, hand_distance=hand_distance)
        placed = (finite and detached and self.evidence.carried and self.released and opened
                  and hand_distance > 0.18 and supported
                  and layout_mod.settled_in_basket(pos, speed, BASKET, RADIUS))
        self.hold = self.hold + 1 if placed else 0
        if not finite:
            self.failure = "non_finite_state"
        elif pos[2] < 0.40:
            self.failure = "apple_dropped"
        success = self.hold >= 30 and self.failure is None
        return dict(carried=self.evidence.carried, released=self.released, hold=self.hold,
                    failure=self.failure or "none", success=success)


def episode(rng, length=160):
    """A noisy scripted harvest attempt; random flaws make many of them fail."""
    on_tree = (0.7758749684943297, -1.0960502731017878, TARGET_Z)
    in_basket = (BASKET[0] + rng.uniform(-0.05, 0.05), BASKET[1] + rng.uniform(-0.05, 0.05), BASKET[2] + 0.01 + RADIUS)
    t_grip, t_detach = rng.randint(5, 25), rng.randint(20, 45)
    t_lift_end, t_place = t_detach + rng.randint(5, 30), rng.randint(70, 100)
    t_open = t_place + rng.randint(-5, 10)
    t_away = t_open + rng.randint(1, 8)
    slip = rng.random() < 0.25  # lose grip briefly while carrying
    weak = rng.random() < 0.15  # thumb never presses hard enough
    drop = rng.random() < 0.1  # apple falls to the ground
    nan_step = rng.randint(0, length) if rng.random() < 0.05 else -1
    lift = rng.choice([0.03, 0.08, 0.12])
    open_value = rng.choice([0.0, 0.1, 0.35])  # 0.35 rad never counts as an open hand
    for t in range(length):
        detached = t >= t_detach
        gripping = t_grip <= t < t_open and not (slip and t_detach + 3 <= t < t_detach + 6)
        if not detached:
            pos = on_tree
        elif t < t_lift_end:
            pos = (on_tree[0], on_tree[1], on_tree[2] + lift * (t - t_detach) / max(1, t_lift_end - t_detach))
        elif t < t_place:
            pos = (on_tree[0], on_tree[1], on_tree[2] + lift)
        else:
            pos = in_basket
        if drop and t > t_lift_end:
            pos = (pos[0], pos[1], 0.03)
        speed = 0.0 if t >= t_place + 2 or not detached else rng.choice([0.01, 0.2])
        palm = (pos[0] + 0.03, pos[1], pos[2] + 0.05) if t < t_away else (pos[0] + 0.25, pos[1], pos[2] + 0.2)
        thumb = 0.0 if weak else (0.4 if gripping else 0.0)
        finger_forces = {name: (thumb if i < 3 else (0.5 if gripping else 0.0)) for i, name in enumerate(FINGER_LINKS)}
        floor_z = MASS * 9.81 if pos == in_basket else 0.0
        hand = [0.8] * 7 if gripping else [open_value] * 7
        finite = t != nan_step
        yield detached, pos, speed, palm, finger_forces, floor_z, hand, finite


def test_parity(num_episodes=400, seed=0):
    rng = random.Random(seed)
    params = HarvestParams(apple_radius=RADIUS, apple_mass=MASS, apple_rest_height=TARGET_Z, basket_center=BASKET)
    episodes = [list(episode(rng)) for _ in range(num_episodes)]
    tracker = HarvestTracker(num_episodes, params, "cpu")
    originals = [Original() for _ in range(num_episodes)]
    outcomes = {}
    for t in range(len(episodes[0])):
        rows = [ep[t] for ep in episodes]
        tensor = lambda values, **kw: torch.tensor(values, **kw)  # noqa: E731
        forces = [[r[4][name] for name in FINGER_LINKS] for r in rows]
        tracker.update(
            detached=tensor([r[0] for r in rows]),
            apple_pos=tensor([r[1] for r in rows]),
            apple_speed=tensor([r[2] for r in rows]),
            palm_pos=tensor([r[3] for r in rows]),
            thumb_forces=tensor([f[:3] for f in forces]),
            finger_forces=tensor([f[3:] for f in forces]),
            basket_normal_force_z=tensor([r[5] for r in rows]),
            hand_joint_pos=tensor([r[6] for r in rows]),
            finite=tensor([r[7] for r in rows]),
        )
        for i, (orig, row) in enumerate(zip(originals, rows)):
            want = orig.step(*row)
            got = dict(carried=bool(tracker.carried[i]), released=bool(tracker.released[i]),
                       hold=int(tracker.hold_steps[i]), failure=FAILURE_NAMES[int(tracker.failure[i])],
                       success=bool(tracker.success[i]))
            assert got == want, f"episode {i} step {t}: batched {got} != original {want}"
            outcomes.setdefault(i, set()).add(want["failure"] if want["failure"] != "none" else
                                              ("success" if want["success"] else "none"))
    summary = {}
    for kinds in outcomes.values():
        for k in kinds:
            summary[k] = summary.get(k, 0) + 1
    return summary


if __name__ == "__main__":
    summary = test_parity()
    print("parity OK over 400 randomized episodes x 160 steps; outcomes seen:", summary)
