# Audit of the 2026-09-12 documentation cleanup

Date: 2026-09-12.
Auditor: Claude, at the user's instruction ("好的你去处理吧", closing the outstanding
item "the section-by-section comparison has not been done").

## 1. Why this audit exists

On 2026-09-12 the index page `icrl_softmax/ACTIVE_WORKSPACE.md` was rewritten by a
concurrent session from a `1042`-line English per-task evidence ledger into a
`102`-line Chinese current-state page: `+99 / −996`. The rewrite was landing work,
and my own review of it was **heading-level only**. I said at the time that whether
any *unique* content had been dropped could not be settled that way.

The deleted text is not lost — it is at commit `c1e03dd` — but "it is in git
history" is not the same as "it is in the research record". An index that has to be
read out of `git show` is not an index. This audit settles which parts are genuinely
unique and puts them back.

## 2. Method

Mechanical, then read:

1. Outline the deleted ledger by heading with line ranges (`37` headings).
2. For every deleted section, extract the machine-checkable tokens — commit hashes
   and three-or-more-decimal numeric constants — and search **every tracked file**
   except the deleted ledger itself.
3. For tokens not found as literal strings, search the sealed result bundles
   **numerically with a tolerance of `5e-4`**, because a prose value may simply have
   been rounded (`22.474` in the ledger is `22.473988581316465` in the bundle).
4. Read by hand the sections the mechanical pass flagged.

Prose that migrated into the new *Chinese* index page cannot be matched by string
search against the old *English* page, so migration into the new page is assessed
by reading, not by the token search. That limitation is why the per-task sections
are classified against their own result reports rather than against the new page.

## 3. Verdict

**No scientific evidence was destroyed by the cleanup.** Every numeric value the
ledger recorded is present in the tracked tree under a longer literal, or in the
sealed result bundles, or both. Every commit hash in the three "execution evidence"
sections survives in a task sheet or result report.

**One thing was lost: a synthesis paragraph.** The section
`Measured scale of the current bottleneck (2026-09-11)` tied the numbers together
into a conclusion that no other document states. It is restored in §6.

## 4. Section-by-section classification

| deleted section | lines | classification | where it lives now |
|---|---:|---|---|
| `Current objective` | `5` | migrated | new index page, "研究问题" |
| `FP-ITER5-001` | `46` | migrated | `research_branches/FP-ITER5-001/claude/first_result.md` |
| `FP-ATTN-ITER4-001` | `51` | migrated | `research_branches/FP-ATTN-ITER4-001/claude/first_result.md` |
| `FP-ITER4-001` | `51` | migrated | `research_branches/FP-ITER4-001/claude/first_result.md` |
| `FP-ATTN-ITER-001` | `63` | migrated | `research_branches/FP-ATTN-ITER-001/claude/first_result.md` |
| `FP-ITER3-001` | `43` | migrated | `research_branches/FP-ITER3-001/claude/first_result.md` |
| `FP-ITER2-001` | `50` | migrated | `research_branches/FP-ITER2-001/claude/first_result.md` |
| `FP-ATTN-001` | `61` | migrated | `research_branches/FP-ATTN-001/claude/first_result.md` |
| `FP-SCALE-002` | `49` | migrated | `research_branches/FP-SCALE-002/claude/first_result.md` |
| `The mixing effect is explained` | `25` | migrated | `research_branches/FP-SCALE-002/claude/mixing_mechanism.md` |
| `FP-SCALE-001` | `28` | migrated | `research_branches/FP-SCALE-001/claude/first_result.md` |
| `FP-ESARSA-001 corpus partially reconstructed` | `29` | migrated | `research_branches/FP-ESARSA-001/claude/reconstruction_posthoc.md` |
| `Verified precedents and evidence` | `35` | **partly unique** | prose migrated to the FP-ESARSA records; two commit pointers existed only here (§5) |
| `Measured scale of the current bottleneck` | `20` | **unique** | nowhere; restored in §6 |
| `FP-KERN-001`, `FP-KERN-002`, kernel intro | `61` | migrated | `research_branches/FP-KERN-002/codex/final_synthesis.md` and the task sheets |
| `FP-KERN evidence location correction` | `24` | migrated | **carried into the new index page** as "FP-KERN 历史结果的实际位置" |
| `FP-EXPL-001` | `49` | migrated | task sheet and result reports |
| `FP-ITER-001` | `29` | migrated | `research_branches/FP-ITER-001/codex/report.md` |
| `CTRL-PREFLIGHT-001` | `19` | migrated | `research_branches/CTRL-PREFLIGHT-001/codex/report.md` |
| `FP-ADV-001` | `22` | migrated | `research_branches/FP-ADV-001/codex/` |
| `Research governance` | `11` | migrated | `AGENTS.md`; `research_tasks/GOV-001.md` |
| `Verified action-gap execution evidence` | `46` | compressed | all `10` commit hashes survive in the FP-ADV-001 sheets/reports |
| `FP-KERN-001 execution evidence` | `34` | compressed | all `11` hashes survive in the FP-KERN-001 sheets/reports |
| `FP-KERN-002 execution evidence` | `56` | compressed | all `11` hashes survive in the FP-KERN-002 sheets/reports |
| `Active implementation` | `43` | **stale, correctly removed** | see §7 |
| `Active evidence` | `69` | redundant | an inventory of files that all still exist |
| `Historical archive` | `7` | migrated | new index page keeps the archive path |

`+99` lines of the new page therefore carry the migrated pointers; the `−996` are
either duplicated elsewhere or listed above.

## 5. The two commit pointers that existed only in the ledger

`Verified precedents and evidence` named two commits that appeared in **no** tracked
document:

- `1cb59f0` — Claude's sealed main route for FP-ESARSA-001;
- `0ac8eb6` — `main` and `origin/main` after that route was fast-forwarded by
  explicit user authorization on 2026-09-11.

Both **resolve in this repository** (`git cat-file -t` returns `commit` for each),
so they were never unrecoverable, but the ledger was the only place that said what
they were. Recorded here so that the pointer is not lost a second time.

## 6. Restored: measured scale of the bottleneck (2026-09-11)

This is the deleted text, with its numbers re-verified against the sealed outputs
and their exact source keys given. It is a historical motivation record for the
certificate-scale change that followed it, not a new claim.

> The two policy-improvement tasks both emitted zero updates. Reading their numbers
> together localizes the obstruction to certificate scale rather than to the
> construction:
>
> - among emitted certificates `E_Q` has minimum `22.474` and mean `57.686`; the
>   minimum oracle bound slack is `22.302`; the largest realized oracle `Q`
>   sup-error is `3.771`;
> - non-emission reasons are exactly `heldout_pair_support_missing` (`0.410`) and
>   `improvement_lcb_nonpositive` (`0.590`); no other reason fires;
> - `FP-ADV-001` independently emitted `0/480` on all six routes.
>
> So the certificate is roughly one order of magnitude looser than the realized
> error, and the relative-softmax one-step improvement is far smaller than that
> slack. This is the same obstruction `CTRL-PREFLIGHT-001` predicted analytically
> (analytic floor `6.343 > B = 5`). Any next task must choose a reachable guarantee
> or a tighter certificate scale deliberately, and record the choice before
> execution; no frozen parameter of a closed task may be retuned.

Exact sources, so the recovery is checkable rather than asserted:

| ledger value | source | exact value |
|---|---|---|
| `E_Q` min among emitted | `results/FP-ESARSA-001/claude/summary.json` → `/routes/expected_exact/min_e_q_among_emitted` | `22.473988581316465` |
| `E_Q` mean among emitted | same → `/routes/expected_exact/mean_e_q_among_emitted` | `57.685979417763974` |
| min oracle bound slack | same → `/routes/expected_exact/oracle_bound_slack_min` | `22.302395781789524` |
| largest realized oracle `Q` sup-error | same → `/routes/expected_finite/oracle_max_actual_q_sup_error` | `3.771177529405443` |
| `heldout_pair_support_missing` share | same → `/routes/*/failure_reason_rates/heldout_pair_support_missing` | `0.4104166…` |
| `improvement_lcb_nonpositive` share | same → `/routes/expected_exact/failure_reason_rates/improvement_lcb_nonpositive` | `0.5895833…` |
| analytic floor at `n = 16384` | `research_branches/CTRL-PREFLIGHT-001/codex/report.md` line `150` | `6.3429654099` |

Note that `results/` is git-ignored. The values are therefore recoverable **on this
machine** but are not in version control; that is the reason this section is written
down rather than left to a JSON file that a cleanup of `results/` would remove.

## 7. A stale claim whose removal is a cleanup win

`Active implementation` opened with:

> No FP-SCALE-001 code exists yet; the task is in `DRAFT` and nothing may be
> implemented before it becomes `ACTIVE`.

By the time of the cleanup that statement had been false for some days —
`FP-SCALE-001` is closed, `FP-SCALE-002` recorded the first certified improvement,
and the line has since reached six certified steps on two implementations. The
ledger was carrying a stale "active work" block alongside current results, which is
exactly the failure mode a per-task ledger accumulates. Its removal, and the new
page's explicit note that it drops such statements, are improvements; this audit
records the removal as intended rather than as loss.

## 8. Recommendation

The rewritten index page is sound and this audit found no evidence loss. Two small
follow-ups, both done rather than proposed:

1. this document, as the durable home for the one unique section; and
2. a pointer from the new index page's milestone table to this audit, so a reader
   who wants the bottleneck diagnostic can find it without reading git history.

The general lesson for this repository: an index that also serves as the evidence
ledger will drift, because results accumulate faster than summaries are revised.
Keeping the per-task record in `docs/research_branches/` and the index as a pointer
page is the right split — provided unique *synthesis* is written to a document of
its own at the moment it is made, which is what failed here.
