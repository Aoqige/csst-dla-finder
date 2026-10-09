# Branch ledger

Last updated: 2026-10-09.

The repository is a model zoo: each network lives on its own branch, while shared
conventions stay on `main`. On 2026-10-09 the two current SOTA systems were
consolidated into **`sota/hrh_final`** and the remaining exploration branches were
archived.

## Active branches (fork `hrhgit/csst-dla-finder`)

| Branch | SHA | Role |
|---|---|---|
| `main` | `c1d8bb0` | Repository overview, contribution rules, shared conventions. |
| `sota/hrh_final` | `1988474` | **Consolidated SOTA branch** — CNN dual-tower + Transformer, cut from `main`. |
| `network/dilated-resnet-5head` | `a6df06d` | Source branch of the CNN family (merged upstream as PR #1/#2). Kept for history. |
| `network/Transformer` | `5a61ef5` | Source branch of the Transformer family (merged upstream as PR #3). Kept for history. |

## Archived and closed (2026-10-09)

Both were tagged first, then the fork-side branch was deleted. The tags pin the exact
SHAs, so nothing is lost.

| Branch | SHA | Archive tag | Reason |
|---|---|---|---|
| `network/hybrid` | `a4eb467` | `archive/network-hybrid` | Superseded dead candidate; identical to `upstream/network/hybrid`, 3 commits, never opened as a PR. |
| `docs/hrh-update-train-evaluate` | `b4c3ec4` | `archive/docs-hrh-update-train-evaluate` | Merged as PR #4 on 2026-10-09; content preserved on `upstream/main`. |

Restore a closed branch at any time:

```bash
git fetch origin --tags
git branch <name> archive/<tag>
```

## Branches not touched

| Branch | Where | Why |
|---|---|---|
| `network/wzx` | local + `upstream` | Contributed by another author. No write access upstream. |
| `codex/hrh-update-readme-20261005` | local | Has an active worktree at `~/codex-worktrees/hrh-update-readme-20261005`. |
| `upstream/network/{hrh,hrh_update,hybrid,wzx,Template_fitting}` | upstream (read-only) | Upstream history; this fork has no write access. |

## Working copies on the server (not branches)

These are separate clones / worktrees used for parallel experiments. They were **left
untouched** — several hold uncommitted exploration code, so removing them would be
destructive.

| Path | HEAD | Uncommitted entries |
|---|---|---|
| `~/csst-dla-finder` | `network/dilated-resnet-5head` | 59 |
| `~/csst_dla_sota` | `sota/hrh_final` | 0 (new worktree) |
| `~/csst_dla_wt_tf` | `network/Transformer` | — |
| `~/csst-dla-finder-cnnA` … `-cnnF`, `-distill` | `727ece3` | 43–47 each |
| `~/csst-dla-hybrid` | `a4eb467` (detached) | 0 |
| `~/csst-dla-wzx` | `87be853` (detached) | 0 |

## What `sota/hrh_final` does not carry

The branch is deliberately narrow. The following lines exist in the archived history
but are **not** carried on `sota/hrh_final`:

- CNN+Transformer (`ct_*`) feature fusion — closed by the experiment ledger.
- The standalone rescue verifier line (`train_verifier.py`, `build_verifier_val.py`).
- Backup snapshots, `__pycache__`, archived scratch directories and ad-hoc shell
  wrappers that lived in the old working tree.
