# Contributing

Rules for humans and agents alike. Hooks enforce the naming rules; install them once per clone
with `make setup` (or `uv run pre-commit install`), which wires `pre-commit`, `commit-msg` and
`pre-push`. Linked worktrees share the clone's hooks.

## Branches

`main` is the only long-lived branch. Everything else is `<type>/<kebab-slug>`:

```
feat/aruco-board-pose      fix/segment-grip-threshold      refactor/geometry-se3-helpers
```

- The slug says **what changes**, in 2–5 lowercase words joined by `-`.
- **Never** name a branch after who or what produced it, or after its place in a batch:
  no `agent`, `subagent`, `wave`, `phase`, `round`, `task`, `lane`, `worker`, `swarm`,
  `claude`, `codex`, `worktree`, `wip`, `tmp`, and no bare numbers (`feat/perception-2`).
  Parallel work gets parallel *descriptive* names: `feat/perception-board-pose` and
  `feat/segment-grasp-events`, not `wave-1/agent-a` and `wave-1/agent-b`.
- One branch, one change. Rebase on `main` before merging; delete it after.

## Commits

[Conventional Commits](https://www.conventionalcommits.org): `<type>(<scope>)?: <subject>`

| type | use for |
|---|---|
| `feat` | new behaviour (a stage step, CLI command, config option) |
| `fix` | wrong behaviour made right |
| `refactor` | same behaviour, better structure; no `feat`/`fix` mixed in |
| `perf` | same behaviour, faster or smaller |
| `test` | tests only |
| `docs` | README, CONTRIBUTING, docstrings, spec |
| `build` / `ci` | uv, Makefile, pre-commit, CI |
| `chore` | anything else that ships nothing (`chore(deps): bump lerobot to 0.6.2`) |

- Scope (optional): `common perception segment retarget sim generate train eval cli configs
  scripts deps repo`.
- Subject: imperative, lower-case start, no full stop, header ≤ 72 chars.
  `feat(segment): split demos at grasp and release events`.
- Breaking change: `feat(sim)!: ...` plus a `BREAKING CHANGE:` footer.
- Body (after a blank line) says *why*. Agent attribution trailers (`Co-Authored-By:`,
  `Claude-Session:`) go in the footer, never in the header or branch name.
- `Merge …`, `Revert "…"`, `fixup!` / `squash!` headers pass the hook untouched.
- History before this file used plain imperative headers; don't rewrite it.

Check a name by hand: `python3 scripts/check_git_names.py branch feat/my-change`.

## Multi-agent development (superpowers)

Agent work in this repo goes through the [superpowers](https://github.com/obra/superpowers)
skills, in this order. Invoke each skill; don't paraphrase it from memory.

| step | skill | output |
|---|---|---|
| 1. Shape the change | `superpowers:brainstorming` | `docs/superpowers/specs/YYYY-MM-DD-<slug>-design.md` |
| 2. Plan | `superpowers:writing-plans` | `docs/superpowers/plans/YYYY-MM-DD-<slug>.md` |
| 3. Isolate | `superpowers:using-git-worktrees` | one worktree + branch per plan |
| 4. Build | `superpowers:subagent-driven-development` (tasks share files) or `superpowers:dispatching-parallel-agents` (independent tasks) | commits |
| 5. Each task | `superpowers:test-driven-development` | failing test → code → `make test` |
| 6. Review | `superpowers:requesting-code-review`, then `superpowers:receiving-code-review` | fixes |
| 7. Prove it | `superpowers:verification-before-completion` | `make lint test` output |
| 8. Integrate | `superpowers:finishing-a-development-branch` | merge / PR, worktree removed |

Bugs start at `superpowers:systematic-debugging` instead of step 1.

**Worktrees.** One worktree per branch, one agent writing in a worktree at a time.

- Prefer the native tool (`EnterWorktree`, or `isolation: "worktree"` on a subagent). It names
  branches `worktree-<name>`, which the hooks reject: rename before the first commit with
  `git branch -m <type>/<slug>`. When dispatching, put the branch name in the subagent's prompt.
- Fallback: `git worktree add .worktrees/<slug> -b <type>/<slug>` (`.worktrees/` is ignored).
- Then, inside the worktree, `make worktree`. It gives the worktree its own venv under
  `/mnt/data/venvs/ego2libero-wt/` (workspace packages are editable installs, so the main
  `.venv` would import the main checkout's code, and `/` has no room for a local one),
  symlinks the main checkout's `data/` and `models/`, and runs the test baseline.
- Clean up merged worktrees with the `clean-worktrees` skill or `git worktree remove`.

**Parallel agents.** Split by stage package (`packages/e2l_*`); two agents never edit the same
package or `e2l_common` at once. Changes to `e2l_common` (geometry, schemas, config) land on
`main` first, then dependent branches rebase. The orchestrator names every branch from the
change it carries, before dispatching, using the branch rules above.
