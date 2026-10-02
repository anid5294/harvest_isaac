# OrchardBench seed 42 — single-fruit review asset

Generated from https://github.com/humphreymunn/orchardbench at
`6313313db8b1a7d23fb2cc3afd67cac46f29399a`. Apache-2.0 source license is
included as `LICENSE`; source-file SHA-256 values and all relevant generation
parameters are embedded in `tree.json`. No external robot assets or Newton
runtime are redistributed here. The exporter evaluates source geometry only.

Native apple/central-leader preset, seed 42, foliage density 0.6, maximum 40
apples. No morphology changes for robot reach. Branch endpoints span
2.5368 × 2.7822 × 2.4463 metres after the upstream ground-clearance operation;
these are skeleton dimensions, not the rendered leaf/fruit surface bounds.
168 segments, 344 leaves, 40 fruit. The source nominal target height is 2.6 m;
the final measured skeleton height is 2.4463 m, not assumed to equal that input.

`isaac/tree.usda` contains visual wood/foliage, 168 static tapered convex wood
colliders, and 39 visual-only fruit. It intentionally omits fruit 28 and its
stem: the Isaac task creates that sphere as a gravity-enabled rigid body with
a native breakable fixed joint. The selected source fruit radius is 0.030 m,
mass 0.160 kg, exposed stem length 0.025 m, break force 17.3411 N. The existing
Isaac fixture's 0.35 Nm break torque is an explicit simulation approximation,
not a measured cultivar parameter. Joint parameters still need lab validation.

Fruit 28 center: `(0.7758749685, -1.0960502731, 1.0597111973)` m.
Parent branch: 119. Anchor: same X/Y, Z `1.1147111973` m. Conservative minimum
clearance from all wood: 0.0071096 m. All source fruit anchors reconstruct on
their parent segments to numerical precision. Other source fruit can overlap
wood; they are deliberately nonphysical in this first gate, not presented as
collision-safe harvest targets. Leaves are visual-only; wood is rigid.

`isaac/collision_review.usda` exposes those exact collision hulls for inspection.
It is a diagnostic scene, not the task asset. `isaac/manifest.json` hashes the
canonical JSON and USD and identifies the one dynamic target. The loader checks
both hashes and reconstructs the expected layout from canonical topology.

Local Blender inspection passed the supplied OrchardBench-level morphology
bar, not a photorealistic mature-canopy standard. Isaac import, passive hold,
external-load detachment and G1 harvesting remain lab gates. Do not infer a
successful physical grasp from the asset or screenshots.
