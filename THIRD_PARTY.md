# Third-party components

The minimal algorithm implementation in `src/dogleg_rl/_vendor/rsl_rl` comes
from the RSL-RL 2.0.2 fork included by
<https://github.com/limxdynamics/tron1-rl-isaaclab>, revision
`307145edfe95f49c45fd9ccd090ab950e8884b33`.

Each source file retains its original BSD-3-Clause copyright, redistribution
conditions and disclaimer. Absolute import prefixes were relocated under
`dogleg_rl._vendor`; line endings and trailing whitespace were normalized.
No installed environment or unrelated upstream robot model
is redistributed. Original and packaged hashes are in `docs/source_manifest.json`.

Isaac Sim and Isaac Lab are installed separately through their official sources
and remain subject to their own licenses. This repository does not redistribute
the Isaac Sim installation. No repository-wide license is inferred for the
project owner's robot assets or code; retained third-party notices apply to the
corresponding third-party files.
