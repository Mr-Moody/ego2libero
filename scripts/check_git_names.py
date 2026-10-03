"""Enforce CONTRIBUTING.md naming: `<type>(<scope>)?: <subject>` commits, `<type>/<slug>` branches,
and no agent / wave / orchestration names in either. Stdlib only; run by pre-commit.

    python scripts/check_git_names.py commit-msg .git/COMMIT_EDITMSG
    python scripts/check_git_names.py branch [name]   # default: current branch
"""

import re
import subprocess
import sys

TYPES = ("feat", "fix", "refactor", "perf", "test", "docs", "build", "ci", "chore")
SCOPES = (
    "common", "perception", "segment", "retarget", "sim", "generate", "train", "eval",
    "cli", "configs", "scripts", "deps", "repo",
)  # fmt: skip
LONG_LIVED = {"main"}
# Names describe the change, never who or what batch produced it.
BANNED = re.compile(
    r"^(agents?|subagents?|waves?|claude|codex|gpt|copilot|cursor|bot|llm|workers?|swarm"
    r"|phases?|rounds?|tasks?|lanes?|wt|worktree|tmp|wip|temp)"
    r"\d*$"
)
BRANCH = re.compile(rf"^({'|'.join(TYPES)})/([a-z0-9]+(?:-[a-z0-9]+)*)$")
HEADER = re.compile(rf"^({'|'.join(TYPES)})(?:\(([a-z0-9-]+)\))?(!)?: (\S.*)$")
PASSTHROUGH = re.compile(r"^(Merge |Revert \"|fixup! |squash! |amend! )")


def banned_tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[/_.-]", name.lower()) if BANNED.match(t) or t.isdigit()]


def check_branch(name: str) -> list[str]:
    if name in LONG_LIVED or not name or name == "HEAD":  # detached HEAD (rebase, bisect)
        return []
    errors = []
    if not BRANCH.match(name):
        errors.append(f"branch '{name}' must be <type>/<kebab-slug>, type in {', '.join(TYPES)}")
    if bad := banned_tokens(name.split("/", 1)[-1]):
        errors.append(f"branch '{name}' contains agent/wave/numbering tokens: {', '.join(bad)}")
    return errors


def check_commit_msg(text: str) -> list[str]:
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    header = lines[0].rstrip() if lines else ""
    if PASSTHROUGH.match(header):
        return []
    m = HEADER.match(header)
    if not m:
        return [f"'{header}' must be <type>(<scope>)?: <subject>, type in {', '.join(TYPES)}"]
    _, scope, _, subject = m.groups()
    errors = []
    if scope and scope not in SCOPES:
        errors.append(f"scope '{scope}' not in {', '.join(SCOPES)}")
    if len(header) > 72:
        errors.append(f"header is {len(header)} chars (max 72)")
    if subject[0].isupper() or subject.endswith("."):
        errors.append("subject must start lower-case and have no trailing full stop")
    if len(lines) > 1 and lines[1].strip():
        errors.append("leave a blank line between header and body")
    return errors


def current_branch() -> str:
    out = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
    return out.stdout.strip()


def main(argv: list[str]) -> int:
    match argv:
        case ["commit-msg", path]:
            with open(path, encoding="utf-8") as f:
                errors = check_commit_msg(f.read()) + check_branch(current_branch())
        case ["branch", *name]:
            errors = check_branch(name[0] if name else current_branch())
        case _:
            print(__doc__, file=sys.stderr)
            return 2
    for e in errors:
        print(f"✗ {e}  (see CONTRIBUTING.md)", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
