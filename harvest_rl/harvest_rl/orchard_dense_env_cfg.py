"""Opt-in full OrchardBench fruit configuration; no simulator imports or launches.

The supplied inventory/export is frozen by the caller. This does not infer
reachability or grant runtime permission. Legacy scenes are never mutated.
"""
from __future__ import annotations

import copy
import math


def configure_dense_inventory(cfg, inventory_rows, tree_usd_path, active_fruit_id=None, *,
                              setup_callback=None, basket_position_reader=None,
                              before_step=None, after_step=None):
    """Private derivative containing every source apple and its own contact sensor.

    Call after configure_orchard_case so the parked robot stance is already fixed.
    USD and texture closure/hash checks belong to the review packet. No fruit is
    omitted on the basis of feasibility or actor performance.
    """
    rows = tuple(copy.deepcopy(inventory_rows))
    if not rows or cfg.scene.num_envs != 1 or cfg.action_mode != 'delta':
        raise ValueError('Require full nonempty inventory, one world and existing delta actions')
    ids = [int(row['fruit_id']) for row in rows]
    if len(ids) != len(set(ids)) or any(i < 0 for i in ids):
        raise ValueError('Fruit IDs must be distinct nonnegative integers')
    for row in rows:
        fruit = row['fruit']
        if str(row['fruit_id']) != str(row['target_id']) or fruit['id'] != row['fruit_id']:
            raise ValueError('Exact source fruit identity required')
        if (len(fruit['center']) != 3 or len(fruit['anchor']) != 3 or
                any(not math.isfinite(x) for x in (*fruit['center'], *fruit['anchor'])) or
                math.dist(fruit['center'], fruit['anchor']) <= 0 or
                any(not math.isfinite(fruit[k]) or fruit[k] <= 0 for k in
                    ('mass', 'radius', 'stem_length', 'stem_break_force'))):
            raise ValueError('Require finite authentic fruit/stem geometry and positive dynamics')
    selected = ids[0] if active_fruit_id is None else int(active_fruit_id)
    if selected not in ids:
        raise ValueError('Initial target absent from full inventory')
    result = copy.deepcopy(cfg)
    apple_template, contact_template, stem_template = (copy.deepcopy(getattr(result.scene, key))
                                                       for key in ('apple', 'apple_contact', 'stem'))
    result.scene.tree.spawn.usd_path = str(tree_usd_path)
    for index, row in enumerate(rows):
        fid, f = row['fruit_id'], row['fruit']
        apple = copy.deepcopy(apple_template)
        apple.prim_path = f'{{ENV_REGEX_NS}}/Apple_{fid}'
        apple.init_state.pos = tuple(f['center'])
        apple.spawn.mass_props.mass = f['mass']
        apple.spawn.collision_props.collision_enabled = True
        apple.spawn.activate_contact_sensors = True
        # Every object has its own rigid body. No kinematic fruit/parenting.
        apple.spawn.rigid_props.kinematic_enabled = False
        sensor = copy.deepcopy(contact_template)
        sensor.prim_path = apple.prim_path
        stem = copy.deepcopy(stem_template)
        stem.prim_path = f'{{ENV_REGEX_NS}}/Stem_{fid}'
        direction = tuple((c-a)/math.dist(f['center'], f['anchor'])
                          for c, a in zip(f['center'], f['anchor']))
        stem.spawn.height = f['stem_length']
        stem.init_state.pos = tuple(a+d*f['stem_length']/2 for a,d in zip(f['anchor'], direction))
        # Cylinder's local Z points down the physical attachment direction.
        # Isaac config uses backend quaternion convention; derive it from the
        # template convention (existing compat handles actual runtime separately).
        setattr(result.scene, 'apple' if index == 0 else f'dense_apple_{fid}', apple)
        setattr(result.scene, 'apple_contact' if index == 0 else f'dense_contact_{fid}', sensor)
        setattr(result.scene, 'stem' if index == 0 else f'dense_stem_{fid}', stem)
    result.dense_inventory = rows
    result.dense_stem_directions = {str(r['fruit_id']): tuple((c-a)/math.dist(r['fruit']['center'],r['fruit']['anchor']) for c,a in zip(r['fruit']['center'],r['fruit']['anchor'])) for r in rows}
    result.dense_initial_target_id = str(selected)
    result.dense_setup_callback = setup_callback
    result.dense_basket_position_reader = basket_position_reader
    result.dense_before_step, result.dense_after_step = before_step, after_step
    result.terminate_on_success = result.terminate_on_failure = False
    result.stem_break_force_range = None
    for key in ('apple_offset_xy', 'apple_offset_z', 'basket_offset_xy', 'reset_joint_noise', 'arm_reset_noise'):
        setattr(result, key, 0.)
    result.dense_physics_validated = False
    return result
