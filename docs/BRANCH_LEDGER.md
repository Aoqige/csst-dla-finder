# Branch ledger

Last updated: 2026-10-09.

The repository is a model zoo: each network lives on its own branch, while shared
conventions stay on `main`. On 2026-10-09 the two current SOTA systems were
consolidated into **`network/hrh_final`** and the remaining exploration branches were
archived.

## Active branches (fork `hrhgit/csst-dla-finder`)

| Branch | SHA | Role |
|---|---|---|
| `main` | `c1d8bb0` | Repository overview, contribution rules, shared conventions. |
| `network/hrh_final` | `cef21eb` | **Consolidated SOTA branch** — CNN dual-tower + Transformer, cut from `main`. |
| `network/dilated-resnet-5head` | `a6df06d` | Source branch of the CNN family (merged upstream as PR #1/#2). Kept for history. |
| `network/Transformer` | `5a61ef5` | Source branch of the Transformer family (merged upstream as PR #3). Kept for history. |

## Upstream (`Aoqige/csst-dla-finder`)

The upstream convention is `network/<name>`, and pull requests are merged into
`network/*` integration branches — **never into `main`**, which is still the untouched
2026-08-07 baseline.

| Upstream branch | SHA | Note |
|---|---|---|
| `main` | `c1d8bb0` | 2026-08-07 baseline. |
| `network/hrh_final` | `c1d8bb0` | **Reserved slot — empty**, identical to `main`. |
| `network/Transformer` | `81c0d86` | PR #3 merge commit. |
| `network/hrh_update` | `692520a` | PR #4 (docs) merge commit. |
| `network/hrh` | `4ca4abd` | PR #1 merge commit. |
| `network/hybrid` | `a4eb467` | Untouched. |
| `network/wzx` | `87be853` | Untouched. |

| PR | head | base | merged |
|---|---|---|---|
| #1 | `hrhgit:network/dilated-resnet-5head` | `network/hrh` | 2026-09-04 |
| #2 | `hrhgit:network/dilated-resnet-5head` | `network/hrh_update` | 2026-09-04 |
| #3 | `hrhgit:network/Transformer` | `network/Transformer` | 2026-09-17 |
| #4 | `hrhgit:docs/hrh-update-train-evaluate` | `network/hrh_update` | 2026-10-09 |
| #5 | `hrhgit:network/hrh_final` | `network/hrh_final` | open |

Note: the fork's `network/Transformer` (`5a61ef5`) is the pre-merge PR head; upstream
carries the merge commit `81c0d86`.

## Archived and closed (2026-10-09)

Both were tagged first, then the fork-side branch was deleted. The tags pin the exact
SHAs, so nothing is lost.

| Branch | SHA | Archive tag | Reason |
|---|---|---|---|
| `network/hybrid` | `a4eb467` | `archive/network-hybrid` | Superseded dead candidate; identical to `upstream/network/hybrid`, 3 commits, never opened as a PR. |
| `docs/hrh-update-train-evaluate` | `b4c3ec4` | `archive/docs-hrh-update-train-evaluate` | Merged as PR #4 on 2026-10-09; content preserved on `upstream/network/hrh_update`. |

Restore a closed branch at any time:

```bash
git fetch origin --tags
git branch <name> archive/<tag>
```

## Branches not touched

| Branch | Where | Why |
|---|---|---|
| `network/wzx` | upstream | Contributed by another author. No write access upstream. |
| `codex/hrh-update-readme-20261005` | local | Has an active worktree at `~/codex-worktrees/hrh-update-readme-20261005`. |

## Working copies on the server (not branches)

| Path | HEAD | Note |
|---|---|---|
| `~/csst-dla-finder` | `network/dilated-resnet-5head` | Main working tree. |
| `~/csst_dla_sota` | `network/hrh_final` | This branch's worktree. |
| `~/csst_dla_wt_tf` | `network/Transformer` | TF worktree. |
| `~/codex-worktrees/hrh-update-readme-20261005` | `codex/hrh-update-readme-20261005` | Docs worktree. |

The eight exploration clones (`cnnA`–`cnnF`, `distill`, `hybrid`, `wzx`) were removed on
2026-10-09; their uncommitted working trees are archived in
`~/csst_dla_runs_archive/exploration_clones_20261009/`.

## What `network/hrh_final` does not carry

The branch is deliberately narrow. The following lines exist in the archived history but
are **not** carried here:

- CNN+Transformer (`ct_*`) feature fusion — closed by the experiment ledger.
- The standalone rescue verifier line (`train_verifier.py`, `build_verifier_val.py`).
- Backup snapshots, `__pycache__`, archived scratch directories and ad-hoc shell wrappers.
