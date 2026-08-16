"""Install/update hook, run by ComfyUI-Manager after it installs or updates this node pack.

`requirements.txt` pins the diffusers fork to an exact commit, but the fork's version string never changes —
it is `0.40.0.dev0` at every commit. So when the pin moves, pip fetches the new commit, sees that
`diffusers==0.40.0.dev0` is already installed, and installs nothing while reporting success. Updating would
silently leave the old engine in place next to new node code.

This forces the reinstall, and only when the installed commit actually differs, so a routine update that did
not move the pin costs nothing. `--no-deps` is not an optimisation: a full reinstall would resolve diffusers'
own dependency tree and can pull a different torch into ComfyUI's environment.
"""

import json
import subprocess
import sys
from pathlib import Path


def pinned_requirement() -> str | None:
    """The `git+...@<commit>` line for the diffusers fork, as written in requirements.txt."""
    requirements = Path(__file__).parent / "requirements.txt"
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("git+") and "diffusers" in line:
            return line
    return None


def installed_commit() -> str | None:
    """The commit pip recorded for the installed diffusers, or None if it did not come from a repository."""
    import importlib.metadata as metadata

    try:
        recorded = metadata.distribution("diffusers").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not recorded:
        return None
    return json.loads(recorded).get("vcs_info", {}).get("commit_id")


def main() -> int:
    requirement = pinned_requirement()
    if requirement is None:
        print("MiniMax Music 3: no diffusers fork pinned in requirements.txt, nothing to do")
        return 0

    wanted = requirement.rsplit("@", 1)[-1]
    current = installed_commit()
    if current == wanted:
        print(f"MiniMax Music 3: diffusers fork already at {wanted[:10]}")
        return 0

    print(f"MiniMax Music 3: updating the diffusers fork {str(current)[:10]} -> {wanted[:10]}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", requirement],
        check=False,
    )
    if result.returncode != 0:
        print(
            "MiniMax Music 3: that install failed. Run it by hand from ComfyUI's Python:\n"
            f"    python -m pip install --no-deps --force-reinstall {requirement}"
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
