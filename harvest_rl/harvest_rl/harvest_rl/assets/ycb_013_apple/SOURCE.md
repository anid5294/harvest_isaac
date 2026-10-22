# YCB 013_apple

Source: YCB Object and Model Set, `013_apple`, Google 16k scan
(`http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/013_apple_google_16k.tgz`,
5,950,199 bytes). Project page and license terms: https://www.ycbbenchmarks.com
(B. Calli et al., "The YCB Object and Model Set").

Converted by `scripts/build_ycb_apple.py`:

- `apple_visual.usda`: `textured.obj` (8,194 vertices, 16,384 faces) with `texture_map.png`,
  recentred on its bounding box (half extents 0.0377 x 0.0374 x 0.0359 m), Z up (stem up),
  real size, no physics.
- `apple_physics.usda`: references the visual apple and adds a rigid body, mass API (mass is
  set by the environment from tree.json: 0.16 kg), a convex-hull collider and harvest_isaac's
  apple friction (static 1.4, dynamic 1.1, restitution 0).
- `../tree_orchardbench_seed42/isaac/tree_ycb.usda`: copy of the OrchardBench `tree.usda` with its
  39 visual fruit spheres replaced by `apple_visual.usda`, hanging from each tree.json stem anchor.
