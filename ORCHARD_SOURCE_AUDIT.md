# Orchard source audit (2026-10-01)

This note distinguishes tree geometry, physics runtimes, downloadable assets, and
their licenses. It does not claim a successful Isaac harvest or approve an asset
whose files and license were not inspected.

## Inspected baseline

`harvest_isaac` HEAD is `c38fa20ce22cd0bab6e1b3467f32eb929a2c188a`.
At inspection, `README.md`, `ROADMAP.md`, and `scripts/arena_apple_pick.py`
were modified; planning notes, the OrchardBench exporter/importer, and several
other files were untracked. These are preserved. The local OrchardBench review
checkout at `/private/tmp/harvest-orchardbench-review-20261001` is
`6313313db8b1a7d23fb2cc3afd67cac46f29399a`; its checked `LICENSE` is
Apache-2.0. The existing G1 task is Isaac Lab 2.0.2 / Isaac Sim 4.5 with a
normalized 43-joint G1 + Dex3 action contract, three RGB cameras, contact and
PhysX joint-break logic, and LeRobot export. The separate Arena trial targets
Isaac Sim 6.1. See local `AGENTS.md` and `README.md` for that distinction.

## Candidate evidence

| Source | Verified contribution | Runtime / license boundary |
|---|---|---|
| [OrchardBench](https://github.com/humphreymunn/orchardbench) | At the pinned local revision, `treesim/skeleton.py` defines `Segment` with parent index, endpoints, taper radii, order, depth, and frame, with `TreeSkeleton` independent of physics/rendering. `treesim/config.py` has a distinct `apple` central-leader preset; `lsystem.py` grows it; `fruit.py` places apples on thin outer/mid wood and records parent segment and attachment point. | Apache-2.0 source. The full runtime uses [Newton/Warp](https://github.com/humphreymunn/orchardbench/blob/main/README.md) and a RidgebackFranka with wrist depth camera, rather than this G1 action/camera stack. `fruit.py` imports Warp and defines a runtime tether and palm-assist path; geometry-only export must isolate pure placement. |
| [OSU `lpy_treesim`](https://github.com/OSUrobotics/lpy_treesim) | README describes L-Py growth with pruning/tying, modern architectures including upright fruiting offshoot and V-trellis, and mesh generation with semantic, instance, or per-cylinder labels. Its README gives a `make_n_trees.py` export command and lists OpenAlea/L-Py, `usd-core`, and Shapely dependencies. | The public root file listing inspected here has no `LICENSE` file; a metadata claim of MIT is insufficient to establish distributed code or generated-asset terms. The README does not demonstrate that the exported mesh/labels preserve parent-child branch topology and apple anchor records. Verify those files and terms before reuse. |
| [RoboOrchardSim](https://github.com/HorizonRobotics/RoboOrchardSim) | Isaac Sim/Lab task assembly, launches, synthesis, evaluation, and policy tooling; source README describes [separate Hugging Face assets](https://huggingface.co/datasets/HorizonRobotics/robo_orchard_sim_assets) at revision `instructmove_v1`, with `OBJECTS` and `NVIDIA/Assets/Isaac/4.1` directories. | Code repository says Apache-2.0. Its **current** README specifies Python 3.11 and Isaac Sim 5.1.0. The asset dataset's exact files, tree morphology/topology, per-file licensing, and compatibility with Sim 4.5 were **not verified**: the HF revision inventory endpoint was inaccessible and the public dataset page has no card. Do not infer the code license covers the asset package. |
| [O3DE ROSConDemo](https://github.com/o3de/ROSConDemo) | README documents an apple-orchard scene, many apples, ROS 2 picking components, and an Apple Kraken robot. | The [project LICENSE](https://github.com/o3de/ROSConDemo/blob/main/LICENSE.txt) separates code/project data (Apache-2.0 OR MIT), `Project/Assets` 3D models/textures/game assets ([CC BY-NC 4.0](https://github.com/o3de/ROSConDemo/blob/main/LICENSE-CC-BY-NC-4.0.txt)), and documentation (CC BY 4.0). The [O3DE engine license](https://github.com/o3de/o3de/blob/development/LICENSE.txt) does not relicense the demo orchard assets. The demo is also an O3DE/ROS 2 runtime, not an Isaac scene asset with verified branch/fruit topology. |
| [Virtual Orchard Simulation, IEEE 11667843](https://ieeexplore.ieee.org/abstract/document/11667843) | Publisher abstract identifies a Gazebo/ROS 2 virtual orchard, dynamic rain/headlight plugins, compliant fruit-detachment joints, perception/localization benchmarks, and a tested soft-gripper prototype. | The abstract says integration of the full physical robot is future work. It establishes a research approach, not a verified downloadable tree asset, source license, or drop-in G1/Isaac runtime. |

## Adapter decision and physical limits

The narrowest integration is to run the pinned OrchardBench `apple` generator
offline and export its exact branch graph, radii, fruit parent IDs/anchors,
and foliage poses into a versioned metre/Z-up artifact. The current
`scripts/export_orchardbench_tree.py` and `external_tree.py` are untracked
integration work, not validated simulator output. Their source pin and topology
checks are useful; they still need source-faithful appearance, collider/anchor
alignment, and an Isaac run. Keeping geometry generation outside Isaac avoids
loading Newton/Warp into the Sim 4.5 Python process. Generated geometry may be
used under OrchardBench's Apache-2.0 terms, with the source revision and license
recorded; do not carry over its robot or unrelated downloaded assets by inference.

For one fruit, ensure its anchor projects onto the stated parent branch, the
fruit center remains at the native generated position, its stem has positive
length, the fixed-joint local frames coincide in world space, and the branch
collider does not initially penetrate the fruit. The importer must not shift or
shrink the tree to satisfy the existing G1 reach. Fruit gravity hold, measured
finger contact, native joint break, and final basket settlement are separate
runtime gates. A static wood/visual foliage adapter omits OrchardBench's branch
compliance and branch-break dynamics; it must be described as such.

A Newton migration would gain OrchardBench's native compliant tree and batched
physics, but it would require porting the G1 articulation and 43-joint mapping,
Dex3 contact and break-event criteria, locomotion/control semantics, the three
RGB views and calibration metadata, reset/termination behavior, and dataset
recording. OrchardBench's reference robot and depth camera do not supply those
equivalents. The geometry adapter therefore has a smaller surface for the
current single-tree task; it does not prove G1 can reach or harvest the chosen
source fruit. Scaling to rows or many environments remains a separate test.
