# Single-tree architecture decision

## Scope and inspected baseline

Decision date: 2026-10-01. Local and remote
`feature/isaac-orchard-preview` HEAD were both
`c38fa20ce22cd0bab6e1b3467f32eb929a2c188a`. Existing uncommitted user work is
preserved. The current commercial generator is not OrchardBench. Neither its
ladder-like morphology nor the older sphere canopy is accepted as a research
asset. Target runtime remains Songkhla, Isaac Sim 4.5 / Isaac Lab 2.0.2.

## Decision

Use **OrchardBench's offline stochastic apple generator**, export its branch
graph, fruit anchors and leaf transforms, and import those into Isaac. Keep the
generator outside the Isaac Python environment. Choose its **central-leader
apple** architecture, not the previous hand-authored tall-spindle arrangement.
Preserve its apple preset's physical scale; do not shrink it to G1's reach.

The morphology source is useful independently of the simulator. `skeleton.py`
stores segment parents, endpoints, taper, order and XYZW frames. Generation and
placement use NumPy. The fruit module also imports Warp for runtime kernels;
the offline adapter must isolate only placement, not initialize that runtime.

**Do not port OrchardBench's grip assist.** Its fruit runtime includes a spring
to a hand-frame position. That is not admissible evidence of the unassisted
contact grasp required here. Use native PhysX joint breaks and independently
measured finger contacts in the existing Isaac task.

## Alternatives

| Approach | What it provides | Work and risks | Decision |
|---|---|---|---|
| Isaac + OrchardBench offline | Apple-specific stochastic 3-D morphology, branch graph/taper, branch-associated fruit and foliage | Canonical export, faithful mesh conversion, aligned colliders and independent PhysX fruit required; upstream visual quality must still be inspected | Selected |
| Isaac + L-Py/TreeSim offline | Pruning/tying rules and modern orchard growth models; PLY/USD export tooling | L-Py/PlantGL environment, verify preservation of parent topology beyond segmentation labels; fruit/foliage integration and license-file ambiguity | Viable fallback, not shortest complete path |
| Isaac + static third-party trees | Potentially strong visual appearance and reusable meshes | Static transform variation is not stochastic morphology; topology and fruit anchors often unavailable; asset licenses separate from code | Reference/background use only unless topology inspected |
| New original generator | Complete schema and dependency control | Highest botanical validation burden; current iterations visibly fail; must independently implement all growth/fruit/leaf models | Reject for this iteration |
| Newton + OrchardBench | Native tree simulation, compliance and demo tooling | G1/contact control, locomotion, three cameras, action ordering, dataset export and policy execution all require integration; demo grip assist still must be removed | Not justified by geometry alone |

All Isaac choices preserve the existing normalized 43-D G1 action/state contract,
three cameras and recording pipeline. None creates a locomotion controller or
a successful grasp policy automatically. Newton is not intrinsically excluded
from those capabilities, but the current repository does not provide a verified
drop-in equivalent. Its migration cost is broader than a tree importer.

For many environments, rigid wood and visual-only foliage are an explicit first
approximation. Mesh batching/instancing and simplified wood colliders are useful;
flexible physics on every twig is not required now. Scaling performance has not
been benchmarked. No rows or locomotion expansion before the single-tree gates.

## Source and license distinctions

- [OrchardBench](https://github.com/humphreymunn/orchardbench): Apache-2.0 source;
  use a pinned revision and record source hashes in the generated artifact.
  Do not import its unrelated robot assets or claim their licenses from the
  generator's license. Our output contains generated geometry, not a copied
  simulator implementation.
- [TreeSim](https://github.com/OSUrobotics/lpy_treesim): metadata identifies MIT
  and references a license file. A metadata declaration alone is not sufficient
  to resolve absent/inconsistent distribution files or dependency licenses.
  Confirm terms with authors before redistribution if unresolved.
- [RoboOrchardSim](https://github.com/HorizonRobotics/RoboOrchardSim): Apache-2.0
  code is manipulation infrastructure, not proof of botanical orchard assets.
  Its current README targets Python 3.11 / Sim 5.1, so it is not a drop-in
  replacement for this Sim 4.5 environment. Its Hugging Face asset package needs
  a separate content/license audit.
- [O3DE ROSConDemo](https://github.com/o3de/ROSConDemo): inspect asset terms
  independently from the Apache/MIT code terms. Its `Project/Assets` are
  CC BY-NC 4.0; they are not incorporated into this asset.
- [Virtual Orchard Simulation](https://ieeexplore.ieee.org/abstract/document/11667843):
  reference, not an implementation dependency. The follow-up source audit
  records its abstract's scope; no downloadable asset or reusable implementation
  was established.

## Validation boundaries

Local Blender renders test geometry and scale, **not Isaac rendering or PhysX**.
The actual G1 USD is required for robot-for-scale views; a proxy is not equivalent.
Passing a JSON/topology test is not visual acceptance. Passive fruit stability
is not harvesting success. Only the named task success with measured contact,
detachment, carry, release and basket settlement counts as a completed pick.
Songkhla execution remains a user-run lab gate; SSH execution was not requested.
