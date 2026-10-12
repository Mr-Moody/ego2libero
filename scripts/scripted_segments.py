"""Write hand-designed bowl-onto-plate `RobotSegments` files, to test replay and everything after
it without perception. Demo 000 is the nominal demo; 001.. are jittered the way human demos vary
(see e2l_sim.scripted).

    uv run python scripts/scripted_segments.py                # 20261003_scripted_000 only
    uv run python scripts/scripted_segments.py --count 25     # rehearsal set (make rehearsal)
    uv run e2l sim replay 20261003_scripted_000 --video
"""

from pathlib import Path

import typer

from e2l_common.paths import DataPaths
from e2l_sim.scripted import write_scripted


def main(
    count: int = typer.Option(1, help="Demos to write: 000 nominal, the rest jittered."),
    seed: int = typer.Option(0, help="Jitter seed; demo i uses rng([seed, i])."),
    root: Path | None = typer.Option(None, help="Data root (default: data/ or $E2L_DATA)."),
) -> None:
    paths = DataPaths(root) if root else DataPaths.default()
    for npz in write_scripted(paths, count, seed):
        print(f"wrote {npz}")


if __name__ == "__main__":
    typer.run(main)
