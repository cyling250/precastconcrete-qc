# Project Handover Document

> This document is maintained by AI for task handover. Completed tasks go into "Task history"; unfinished work is listed under "Current task".
> Languages: **English** | [中文](handover.zh-CN.md)
> Last updated: 2026-10-09 22:10 · Agent_20261009213000

## Project snapshot

> Maintaining agent: created by Agent_20260928163100 · last modified by Agent_20261009213000

- **Project**: precastconcrete-qc
- **One-line description**: companion code for the JBE manuscript on material-level point cloud segmentation of precast concrete components (two-stage cascade PC-CascadeNet). The submission package (cover letter included) is complete under `E:\OneDrive\预制构件分割\JBE\`, and the repository has finished untracking result data plus the experiment-code hygiene cleanup (T-011).
- **Tech stack**: Python 3.10 (managed by uv, `.venv` at the repo root) + torch 2.12.1+cu130 + numpy / open3d / scipy / rich / shapely / matplotlib; GPU: RTX 4060 Ti 16 GB.
- **Architecture overview**:
  - `lib/pointcloud_generator`: parametric point-cloud dataset generator (7 component types × 10 configurations × 2 reinforcement variants × 3 density levels × 3 noise levels = 1,260 samples, written to `mock/inuse/`);
  - `lib/cascade` + `lib/backbones`: two-stage cascade segmentation (S1 three-class noise/concrete/metal → S2 fine segmentation) with the PointMLP/PointNet/PointNet++/PointNeXt backbones;
  - `lib/baselines`: geometric baseline (PCA descriptors + RANSAC) and diagnostic tools (`_dbg_*`/`diag_features` are diagnostics documented in the baselines README, not dead code); `lib/measurement`: segmentation → downstream measurement (stirrup/longitudinal-bar spacing, sleeve offset);
  - `experiments/`: all experiment entry scripts (main baselines phase1/2, ablation A-series, sensitivity S-series, geometric baseline B-series, mechanism and closed-loop E-series).
  - Data flow: generator → `mock/inuse/{train,val,test}` → training (weights into `out/checkpoints`) → evaluation / sensitivity / measurement loop (results into `out/E*`, `out/sensitivity`).
- **Directory layout** (after the 2026-10-08 data/code/doc hygiene pass):
  - `src/lib/` — all library code (backbones / baselines / cascade / measurement / pointcloud_generator / utils)
  - `src/experiments/` — experiment entry scripts, uniformly named `experiment_<id>_<name>.py`; the `_` prefix marks helper modules (`_s1_trainer.py`/`_s2_trainer.py`/`_a2_trainer.py`/`_e5_deep_predict.py`/`_config.py`). Relative `--out` paths in the E/B series are anchored to the project root via `lib.utils.anchor_project_path`
  - `scripts/` — reusable scripts kept in the repo (queue runner queue_runner, plotting make_figures, data cross-checks summarize_out_jsons/key_numbers/e4e5_breakdown/e5_nested_p90, CSV export make_plot_csvs; see `scripts/README.md`)
  - `out/` — all result data (checkpoints / E1|E3|E4|E5|B1 / sensitivity / plot_csvs / train_logs / dataset); local only, gitignored as a whole, never committed
  - `out/checkpoints/` — model weights, history, summaries (one directory per experiment id)
  - `out/train_logs/` — training/run logs (numbered `00_…`–`21_…`), `progress.md` progress history, `_runner_status.json` queue final state
  - `out/E1|E3|E4|E5|B1/` — result JSONs of the mechanism/comparison experiments (final versions; the source of the revised manuscript numbers)
  - `out/sensitivity/` — S1–S4 sensitivity outputs (S3/S4 in both protocols; the `_allkeys` suffix = legacy protocol leakage control)
  - `out/plot_csvs/` — plotting CSVs exported by `scripts/make_plot_csvs.py` (new home of the plot data from the former JBE package)
  - `out/dataset/` — `split_test_keys.json` (14 test split keys)
  - `out/figures/` — deleted (draft figures and the retired Arial-style set; the authoritative figure set lives in `E:\OneDrive\预制构件分割\JBE\figure\`)
  - `docs/` — this handover document only; the manuscript-revision guidance files have all been moved out (2026-10-09; recoverable from git history and `F:\pcqc_backup_20260928\docs_deleted_20261008|09\`)
  - `mock/inuse/` — the generated dataset (6.9 GB, gitignored; `.gpu_cache_*.pt` under `train/` are training acceleration caches, do not delete); `mock/origin` (clean samples) and `mock/add_noise` (noised intermediate layer) were deleted (regenerable by the generator; at runtime only inuse is read)
  - `temp_scripts/` — AI scratch scripts (gitignored; currently holds 3 small scripts for docx extraction and cover-letter updates)
  - `temp/` — session-level staging directory (gitignored, emptied after use)
  - `E:\OneDrive\预制构件分割\JBE\` — submission directory (Manuscript + 5 attachments + the authoritative `figure\` set of 18 TIFFs + Cover letter new.docx); **the JBE package was deleted at the user's request**
  - `F:\pcqc_backup_20260928\` — external full backup taken before the reorganization (contains the retired `deprecated_s2_cache/`, the migration manifest `migration_report.json`, and `docs_deleted_20261008/`)
- **Common commands**:
  - Install dependencies: `uv sync` (at the repo root; torch comes from the cu130 index)
  - Run a single experiment: `cd src`, then `<root>/.venv/Scripts/python.exe -m experiments.experiment_xxx` (module lookup relies on cwd=src; path constants are absolute and cwd-independent)
  - Queue runner (reusable): `"F:\precastconcrete-qc\.venv\Scripts\python.exe" "F:\precastconcrete-qc\scripts\queue_runner.py"` (idempotent; currently 9/9 all DONE)
  - Queue status: read `out/train_logs/_runner_status.json`; progress history: `out/train_logs/progress.md`

## Task history

> Organized by task (not by conversation), newest first. Numbers increase from T-001 and are never reused.

> Maintaining agent: created by Agent_20260928163100 · last modified by Agent_20261009213000

### T-011 Untrack result data + experiment code hygiene cleanup (pre-submission)

- **Completed**: 2026-10-09
- **Executing agent**: Agent_20261009213000
- **Background**: the user asked that the repository keep only code and result data — clean out temp and manuscript-writing files, keep result data out of git, remove the rework traces in experiments and regularize experiment naming; do not touch `E:\OneDrive\预制构件分割\JBE\`.
- **File cleanup**: deleted `temp/` (the jbe manuscript extracts and jbe_review) and the manuscript copies under `temp_scripts/out/` (v2–v6 docx+md, ~130 MB); deleted the two revision-guidance docs in docs/ (the paragraph-by-paragraph guide and the supplementary-material comments, backed up first to `F:\pcqc_backup_20260928\docs_deleted_20261009\`); docs/ now contains only this handover document.
- **Git untracking**: `git rm --cached` on the 5 tracked files under `out/` (the B1 JSON, split_test_keys, and the three train_logs files); `.gitignore` now ignores `out/` as a whole instead of `out/checkpoints/`, adds `temp/`, and drops stale pre-reorganization lines such as `experiments/checkpoints`.
- **Experiments trace cleanup**: the copy-pasted 4-line `--out` anchoring patches in the 5 E/B scripts (left over from P-006) were consolidated into `lib.utils.anchor_project_path` (new; exported in `__init__`), leaving one call line per script; the `--model m3` branch of `experiment_s4_noise.py` was restructured to "load the model once, one unified loop", with unchanged output filenames (`S4_summary*.json`); `experiment_e5_deep.py` was in fact an E5 helper module rather than an experiment entry and was renamed `_e5_deep_predict.py` to join the `_`-prefixed helper family (the two imports in e5_measurement updated in step). The phase1/phase2 names were kept: the ids carry meaning (stage-1/stage-2 main baselines) and are strongly coupled to the `phase*_baseline_*` checkpoint directories and the historical logs.
- **Verification**: all 59 .py files under src byte-compile; all 7 module names referenced by queue_runner exist; `--help` import chains pass for all 8 experiment entry points; `anchor_project_path` is byte-for-byte equivalent to the original patch formula.
- **Related**: [T-010](#t-010-second-review-after-user-edits-three-way-caption--wording--number-audit), [P-006](#p-006-e-series---out-resolved-against-cwd), [P-007](#p-007-gitignore-ignoring-scripts-nearly-repeated-p-001)

### T-010 Second review after user edits: three-way caption / wording / number audit

- **Completed**: 2026-10-08
- **Executing agent**: Agent_20261008153000
- **Background**: after the user applied the T-009 checklist in Word, re-verify caption correctness, humanize-writing compliance, and factual wording; carried out by three sub-agents (visual acceptance / humanize wording review / technical number check) plus this agent's own re-check.
- **Confirmed fixed**: figure captions C1–C5 all in place (Fig 13 split into (a)(b) with the geometric-baseline note and kept on one page); B1/B2/B3/B5/B6 applied; abstract at 245 words.
- **New must-fix findings**: ① displayed equation numbers skip (7) (they show 1–6, 8–11), so every in-text reference to Eq.(7)–(10) is off by one; ② Figure 10/11 captions mismatched with their images (the noise-axis figure carries no caption, the kNN figure carries the Figure 10 caption, and the Figure 11 caption dangles); ③ §4.2.3 PointNeXt "metal-class IoU 0.780 at pipeline level" cites the wrong quantity (0.780 is sleeve; rsIoU should be 0.852); ④ the Limitations sleeve p90 range 9.0–33.1 should be 5.8–33.1 (measured 5.8 at the low level); ⑤ "monotonically" does not hold (the S4 difference falls back at t=0.1, two places); ⑥ the noise axis is described as starting from "clean scans" in two places, but the code shows t=0 is the low level (`experiment_s4_noise.py` `_interpolate_noise`); ⑦ the captions of Figures 7–17 and the Table 6 note all turned blue.
- **Carried over unfixed**: B4 (five→four deleted only the 400, creating an in-sentence contradiction), B7 [42]+Data availability, B8 AI statement, A2 Heading4 `%4`, D1 fig07/08 y-axis labels.
- **humanize verdict**: AI flavor medium-high — 4 defensive contrasts (heaviest in the M2 paragraph, then the §4.5.2 ending, §4.3, and the start of §4.6), dense colon/semicolon-then-explain patterns, near sentence-level repetition between §2.3 and §2.3.1, metric dumping in the conclusion; "statistically indistinguishable" appears three times without test support.
- **Output**: `docs/稿件逐段修改指导.md` rewritten as a pure action checklist (A Word global / B body in order / C caption rebinding / D in-figure Origin edits / E submission package / F self-check, all with replacement text); the number checks all reproduced against the final JSONs in out/ (Tables 5/6/8 cell by cell; the cited values of E3/E4/E5/S3/S4 all consistent).
- **Third-pass check (2026-10-09, after user edits)**: equation numbers now consecutive 1–10, Fig 10/11 captions rebound, the §4.2.3/§4.3/M2 rewrites are in, and most of the B series is done; **new regressions** — the §4.6.1 rsIoU cross-reference broke when the fields refreshed (the bookmark points at the cmIoU equation, now displayed as Eq. (9); must relink to (10)); A2 Heading4 `%4`, A3 blue captions (style-level 0000FF), B20 five→four, B23 [42], and B24 AI statement remain unfixed. The checklist was rewritten against the third pass (pending items only); the humanize-writing skill absorbed new rule 11 (evidence pointers belong in the sentence, not in parentheses).
- **Related**: [T-009](#t-009-figure-swap-manuscript-review--jbe-package-deletion--repo-datacodedoc-cleanup)

### T-009 Figure-swap manuscript review + JBE package deletion + repo data/code/doc cleanup

- **Completed**: 2026-10-08
- **Executing agent**: Agent_20261008153000
- **Manuscript review verdict**: all 17 embedded figures are byte-identical (MD5) to the authoritative set (image13 = the merged fig13+fig13b with built-in (a)(b) labels — T-008 leftover ③ resolved); R1/R2/R5 applied, title now multi-level numbered. **Rendered via Word COM to PDF for a real render check**, producing the remaining-items list in `docs/稿件逐段修改指导.md` (third-pass version):
  - New must-fix findings: ① the Heading4 numbering template is `%4`, rendering four-level subsections as bare "1/2/3/4" (should be `%1.%2.%3.%4`); ② §4.6.2 "five radius values …400 mm" contradicts Fig 15's four points (100–300) (a legacy-figure remnant; recommend changing the text to match the figure); ③ §4.2.2 "(c)–(g)" should be "(c)–(f)" (the new Fig 8 has 6 panels); ④ the Fig 7/8(b) y-axis label "Clean mIoU" contradicts the caption's cmIoU (Origin side);
  - Unchanged old items: §4.6.1 "Eq. (9)"→(10) (the cross-reference points at the wrong bookmark; F9 does not fix it), the half-finished captions of Figs 9–13 (the "图注：" prefix / missing closing parenthesis / no (a)(b) split / split across pages), R6/R8 broken sentences, R9 [42], R12 AI statement, the Table 8 Δ sign.
- **Cleanup executed**: deleted `JBE修改包\` (its guide copy was identical to the docs version, figures_py was a copy of the authoritative set, and the CSVs are regenerable by make_plot_csvs); emptied temp_scripts and consolidated the 7 valid scripts into `scripts/` (README records their purposes; make_plot_csvs output path changed to `out/plot_csvs/`; summarize's manuscript path became an optional argument; all smoke-tested); deleted `mock/origin` (0.8G) + `mock/add_noise` (2.1G) (referenced only by the generator pipeline; training/evaluation read only inuse); deleted `out/figures/` (12M of drafts and retired styles); cleared `__pycache__`; removed the `scripts/` ignore from `.gitignore` (preventing a repeat of P-001); deleted the three completed guidance docs in docs/ (the two untracked ones backed up first to `F:\pcqc_backup_20260928\docs_deleted_20261008\`); rewrote the revision guide as the remaining-actions checklist.
- **Non-deletions verified**: `_dbg_*.py`/`diag_features`/`selftest` are diagnostics documented in the baselines README (evidence scripts for the sleeve conclusion), kept; `experiment_e5_deep.py` is an active E5 module, kept; the three repo-hygiene items (matplotlib/dead code/_filter comments) were done long ago, untouched.
- **Related**: [T-008](#t-008-post-holiday-figures-jbe-figure-dir-confirmed-as-the-authoritative-complete-set-oct-08-morning-redraw-dropped)

### T-008 Post-holiday figures: JBE figure dir confirmed as the authoritative complete set (Oct-08 morning redraw dropped)

- **Completed**: 2026-10-08
- **Executing agent**: Agent_20260930 (Oct-08 session)
- **Correction**: the user's feedback was that the correct style is the original grid style. Verified that the holiday session (10-02 00:52) had already produced all figures and archived them under `E:\OneDrive\预制构件分割\JBE\figure\` (fig01–fig17+fig13b, 18 TIFFs; fig07/08 redrawn in Origin with the cmIoU protocol unified; fig16/17 from the leakage-corrected data), each checked for data correctness. **The 9 Arial red-blue-black style figures redrawn that morning do not match the manuscript style and were retired** (backup `F:\precastconcrete-qc\out\figures\panels_v2\` + composites; the OneDrive `成品图\` folder was deleted); the 6 figures_py files are that authoritative set's fig09–13b.
- **Leftovers**: ① the fig07/08 (b) panel y-axis label "Clean mIoU" should become "cmIoU" in Origin; ② when inserting fig13+fig13b into Word, place them side by side with (a)(b) labels; ③ the holiday session's plotting script `script_20261002_make_fig9_13.py` was never saved to disk (recorded in the README, file lost — same trap as P-001); ④ the `add_expierment\稿件修改意见.md` referenced by the README was not found in the JBE directory.
- **Guide updated**: §2 became the "figure cross-reference table" (the figure set is complete); added R15 (§4.2.2 two instances of "Figure 8 (e)" should be "(d)").
- **Related**: [T-007](#t-007-second-checklist-remaining-items-after-the-user-applied-one-revision-round)

### T-007 Second checklist: remaining items after the user applied one revision round

- **Completed**: 2026-09-30
- **Executing agent**: Agent_20260930
- **Outcome**: re-extracted the 11:53 Manuscript.docx and checked the application of the round-one 74 instructions paragraph by paragraph — abstract / contribution paragraph / 2.2.3 / Table 4 note / Table 5 / Table 7 / Table 8 / the three ablation paragraphs / 4.2.3 / 4.3 / 4.5 body / §4.7 / conclusion / Limitations all correctly applied; **rewrote `docs/稿件逐段修改指导.md` as the second-pass version** (R1–R14 remaining items + F1–F9 figure checklist).
- **Remaining must-fix**: R1 §4.2.1 old and new paragraphs duplicated (the old paragraph with 0.638/0.896 not deleted); R2 Table 6 not replaced (still the legacy protocol); R3/R4 §4.6 figure numbers 9/10 clash with the new figures (→14/15); R4 §4.7 still uses the old Figs 11/12 (the body already cites 16/17).
- **New findings**: §4.6.1 "as defined in Eq. (9)" should be (10) (the rsIoU definition, a legacy relic); the broken sentence at the start of the conclusion "a class-balanced strategy local patch sampling strategy"; `_s1_trainer.py:69` verified cm_fn=cm_iou(concrete, metal, metal) — **the history cm column of Fig 7b/8b is already the unified three-class quantity and can be plotted directly, no recomputation needed** (simpler than the round-one instructions).
- **Figure tasks**: 5 figures use the draft images directly (Figs 9–13); Fig 16 = fig11a+11b merged into a dual panel; **the overview panel of Fig 17(a) must be drawn fresh** (three curves from S4_summary.json); Fig 7b/8b replotted from the history cm.
- **Related**: [T-006](#t-006-paragraph-by-paragraph-jbe-revision-guide--jbe-length-audit)

### T-006 Paragraph-by-paragraph JBE revision guide + JBE length audit

- **Completed**: 2026-09-30
- **Executing agent**: Agent_20260930 (new session)
- **Outcome**: produced `docs/稿件逐段修改指导.md` — paragraph-level verdicts for `E:\OneDrive\预制构件分割\JBE\Manuscript.docx` (keep / adjust / rewrite / add) with concrete operations, all numbers re-checked against the final JSONs in `out/` (recheck scripts `temp_scripts/script_20260930_10*.py`). Also checked the JBE Guide for Authors: **no word/page limit for the body**, abstract ≤250 words (current draft 217 ✓), Highlights 3–5 items × ≤85 characters, numbered subsections required, research data Option C mandatory.
- **Key new finding (covered by neither draft)**: under the unified-protocol re-evaluation, **M2 (without class-balanced sampling) ties with S1** (E1: mIoU 0.899 vs 0.899, rs 0.901 vs 0.899) — the old draft's "M2 lowers rsIoU by 0.030" no longer holds; the guide provides an honest rewrite and asks that contribution three in the introduction and conclusion (2) be softened.
- **7 draft errata** (guide §2): R1 "M1 drops 0.118"→0.099; R4 val-optimal r=40→**50 mm**; "never yields sleeve"→max IoU 0.033; R2 leakage-gap endpoints (max +0.013 at t=0.8, not the strongest level); R5 the cm reversal happens only at t=1.0, not t≈0.8; E5 low level 3.1→3.0 mm; R3 the tiered wording of the PointNeXt survival sentence.
- **Related output**: `docs/稿件逐段修改指导.md` (main deliverable, replacing the never-saved final-wording file).

### T-005 Draft deliverables: 8 figures + revised-manuscript markdown

- **Completed**: 2026-09-29
- **Executing agent**: Agent_20260928163100
- **Outcome**: re-ran the M3 single-stage 11-level noise axis (S4 gained `--model m3`, producing `out/sensitivity/S4_summary_m3.json`); drew 8 draft figures (`out/figures/`, scientific-plotting spec, TIFF 600 dpi); wrote the revised-manuscript `docs/论文修改稿_补充实验.md` (six R1–R6 sections in English + Chinese annotations + recomputed tables).
- **Core new findings**: the cascade's mIoU advantage over the single-stage model widens monotonically with noise (+0.010 clean → +0.027 at t=1.0); the cm-protocol reversal at the strong-noise end (0.869 vs 0.863); the E5 per-level breakdown shows that at the medium operating point the cascade's longitudinal-bar error is 5.9 mm (within the ±10 mm tolerance) vs 15.3 mm for the single stage (out of tolerance), and sleeve offset 11.8 vs 72.1 mm — the cascade narrative migrates from "segmentation metrics" to "noise axis + measurement-tolerance reliability". Revised E3 understanding: the three backbones' per-level FPS survival rates are nearly identical; PointNeXt's difference lies in the downsampling chain length (the metal class drops to zero at 128→32); the VRAM explanation does not hold (9.3GB < 16GB).
- **Key changes**: `src/experiments/experiment_s4_noise.py` (--model argument, _load_m3/_evaluate_m3, None-safe printing, save before print); `temp_scripts/script_20260929_173000_make_figures.py` (plotting script).
- **Related pit**: [P-006](#p-006-e-series---out-resolved-against-cwd)

### T-004 Supplementary experiment queue fully complete (9/9)

- **Completed**: 2026-09-29
- **Executing agent**: Agent_20260928163100
- **Outcome**: runner v2 exited normally with rc=0 at 09-29 14:02, all 9 steps OK/SKIPPED_DONE. Key outputs and numbers:
  - `a5_wce` (the M5 ablation): test mIoU 0.864 / cm 0.887, M5 settled (`out/checkpoints/ablation_A5_wce/summary.json`);
  - `E4` geometric-baseline val calibration + test report (`out/E4/b2_*.json`, metal IoU 0.326); `E3` FPS survival (`out/E3/`, pointnet++ deep-layer sparse-class survival ~0.0001, mechanism evidence); `E1` unified cmIoU recompute of Tables 5/6/8 (`out/E1/E1_tables.json`);
  - `S3/S4` corrected protocol + `--all-keys` leakage control (`out/sensitivity/`): the corrected protocol is slightly conservative overall (S3 differs by 0.002–0.007, S4 by 0.002–0.013); the conclusions survive the leakage fix;
  - `E5` segmentation→measurement loop (`out/E5/`): cascade longitudinal-bar spacing error p90=1.86 mm (84% of components valid) vs geometric baseline p90=96 mm (45%); sleeve matching 14/16 vs geometric 0/16; the geometric baseline wins only on unoccluded hairpin bars (~0 error vs the cascade's ~19.5 mm) — discussion material.
- **Key changes**: the `--out` anchoring patch applied to the 5 E/B scripts (see P-006); queue ops recorded in `out/train_logs/progress.md`.
- **Related pit**: [P-006](#p-006-e-series---out-resolved-against-cwd)

### T-003 Repository reorganization (src/ + out/ + docs/ layout)

- **Completed**: 2026-09-28
- **Executing agent**: Agent_20260928163100
- **Outcome**: the root collapsed from 7 miscellaneous directories to `src/ out/ docs/ mock/ temp_scripts/`; all code path constants made absolute (cwd-independent); a full backup taken to `F:\pcqc_backup_20260928\` before the reorganization; all 7 automated verifications passed; the queue resumed under the new layout with runner v2.
- **Key changes**:
  - `lib/` → `src/lib/`, `experiments/` → `src/experiments/` (`__pycache__` cleared);
  - `experiments/checkpoints/` → `out/checkpoints/` (12 `s2_*.pt` training caches ~2.2 GB judged retired → backed up to `deprecated_s2_cache/`);
  - `reports/` split: result JSONs → `out/E3|B1|dataset/`, logs → `out/train_logs/` (incl. `E4_b2_run.log` → `14_E4_b2_run_partial.log`), documents → `docs/`;
  - deleted `.trae/`, `.vscode/` (user request; originals in the backup);
  - path edits: `src/lib/utils/common.py` (`_PROJECT_ROOT` four-level dirname, added `OUT_BASE`/`SENSITIVITY_BASE`, `CHECKPOINT_BASE` → `out/checkpoints`), `src/lib/pointcloud_generator/_config.py` (relative → absolute paths), S1–S4 sensitivity outputs → `out/sensitivity/`, E/B series `--out` defaults → `out/`, `_dbg_summarize.py` reads `out/B1/`;
  - `.gitignore` gained `out/checkpoints/`, `temp_scripts/`;
  - the queue runner rewritten as v2 (cwd=src, new marker-file paths).
- **Related pits**: [P-002](#p-002-_configpy-miscounted-the-project-root-depth), [P-003](#p-003-generator-path-constants-depended-on-cwd), [P-005](#p-005-migration-skeleton-pre-created-the-target-dir-and-clashed-with-the-directory-move)

### T-002 Supplementary queue resumed (first attempt, pre-reorganization)

- **Completed**: 2026-09-28
- **Executing agent**: Agent_20260928163100
- **Outcome**: confirmed the 13-step training queue had only `a5_m5_wce` left (interrupted at ep50/120 on 9-23, checkpoint intact); rebuilt the queue runner v1 and started the a5 resume in the background; set up a 2-hour health-check automation. Later actively stopped because of the directory reorganization (T-003), succeeded by its v2 runner.
- **Key change**: `temp_scripts/script_20260928_162030_queue_runner.py` (v1, superseded by v2 and moved to the backup)
- **Related pits**: [P-001](#p-001-queue-scripts-lost-never-entered-version-control), [P-004](#p-004-zcode-session-crash-orphaned-background-tasks)

### T-001 Supplementary experiment system + 13-step main training queue (pre-reorganization history, multi-session relay)

- **Completed**: 2026-09-23 (when the training queue was interrupted)
- **Executing agent**: unregistered (historical sessions before the numbering rule)
- **Outcome**: per the review self-check report §2.1–2.5, completed all code fixes and 12 of the 13 training-queue steps: phase1/phase2 baselines × {pointmlp, pointnet++, pointnext (b8 re-run), pointnet}, ablations A1a/A1b/A2(M3)/A4(focal) all produced weights and summaries; only step13 `a5_m5_wce` remained (interrupted at ep50/120, best val mIoU 0.846). Fully recorded in `out/train_logs/progress.md` (83 patrol records).
- **Key changes**: S3/S4 gained test-key filtering (fixing the evaluation-set leakage) and the `--all-keys` leakage-control argument; `_s1_trainer` gained a batch-size argument (pointnext b8); the E1/E3/E5/B2/E5-deep experiment scripts and `lib/measurement` were all added in this phase.
- **Related pits**: [P-001](#p-001-queue-scripts-lost-never-entered-version-control), [P-004](#p-004-zcode-session-crash-orphaned-background-tasks)

## Current task

> Maintaining agent: created by Agent_20260928163100 · last modified by Agent_20261009213000

<!-- After the current task completes: organize it into a history entry above, then restore this placeholder -->

**Waiting for the user to submit the JBE package** (all materials ready under `E:\OneDrive\预制构件分割\JBE\`: Manuscript, Title Page, Highlights, Declaration of Interest, Supplementary material, Cover letter new.docx). After submission the project enters reviewer-comment handling; nothing is outstanding on the repository side.

## Pitfall log

> Every entry must link back to the task entry that produced it.

> Maintaining agent: created by Agent_20260928163100 · last modified by Agent_20261008153000

### P-007 .gitignore ignoring scripts/ nearly repeated P-001

- **Related tasks**: [T-009](#t-009-figure-swap-manuscript-review--jbe-package-deletion--repo-datacodedoc-cleanup), [P-001](#p-001-queue-scripts-lost-never-entered-version-control)
- **Recording agent**: Agent_20261008153000
- **Symptom**: while organizing the scripts, found that `.gitignore` had been ignoring the whole `scripts/` directory all along — scripts placed there per the convention would never enter version control.
- **Cause**: the early template ignored scripts/ together with temp_scripts/; one root cause of P-001 (queue scripts lost, never committed) was never truly removed.
- **Resolution**: removed the `scripts/` ignore line (temp_scripts/ stays ignored); the 7 valid scripts consolidated into scripts/ should enter version control with the next commit.

### P-001 Queue scripts lost: never entered version control

- **Related tasks**: [T-002](#t-002-supplementary-queue-resumed-first-attempt-pre-reorganization), [T-001](#t-001-supplementary-experiment-system--13-step-main-training-queue-pre-reorganization-history-multi-session-relay)
- **Recording agent**: Agent_20260928163100
- **Symptom**: while resuming the queue, found `train_queue.py`, `post_queue_runner.py`, and `handover.md` missing from both disk and git history.
- **Cause**: these ops files sat at the repo root and were never `git add`-ed; a directory-cleanup commit (c692f19 and its kind) wiped them out, with no way to recover them from the repository.
- **Resolution**: rebuilt the runner from the progress.md records (v1→v2); from now on, key ops scripts must either enter version control or have a copy in the external backup directory.

### P-002 _config.py miscounted the project-root depth

- **Related tasks**: [T-003](#t-003-repository-reorganization-src--out--docs-layout)
- **Recording agent**: Agent_20260928163100
- **Symptom**: after the reorganization, `_config.py` resolved `_PROJECT_ROOT` to `…\src` (it should be the repo root), and `mock/inuse/test` could not be found.
- **Cause**: from `src/lib/pointcloud_generator/_config.py` back to the root takes **four** dirnames; it was written with three — the count came from the stale memory of "how deep the source sits under the root" in the old layout.
- **Resolution**: changed to four dirnames and verified; the verification script (`temp_scripts/script_20260928_173000_verify.py`) caught it before anything reached the run stage.

### P-003 Generator path constants depended on cwd

- **Related tasks**: [T-003](#t-003-repository-reorganization-src--out--docs-layout)
- **Recording agent**: Agent_20260928163100
- **Symptom**: even with common.py absolutized, `test_keys()` still could not find the dataset.
- **Cause**: `lib/pointcloud_generator/_config.py` had its own set of relative path constants (`"mock/inuse/test"` etc.); the old project masked the problem with cwd=repo-root, and moving the runtime cwd to src exposed it.
- **Resolution**: all path constants in that file were rebased onto `_PROJECT_ROOT`; a repo-wide grep confirmed no other cwd dependencies (docstring examples aside).

### P-004 ZCode session crash orphaned background tasks

- **Related tasks**: [T-002](#t-002-supplementary-queue-resumed-first-attempt-pre-reorganization), [T-001](#t-001-supplementary-experiment-system--13-step-main-training-queue-pre-reorganization-history-multi-session-relay)
- **Recording agent**: Agent_20260928163100
- **Symptom**: after several ZCode crashes/reloads, background terminal task entries vanished; training processes were either killed with the session or survived unmanaged and invisible.
- **Cause**: background tasks are bound to the session lifecycle.
- **Resolution**: a 2-hour health-check automation serves as the safety net (alive → record progress; dead → restart from the checkpoint); training itself has epoch-level checkpoints, so the loss was bounded.

### P-005 Migration skeleton pre-created the target dir and clashed with the directory move

- **Related tasks**: [T-003](#t-003-repository-reorganization-src--out--docs-layout)
- **Recording agent**: Agent_20260928163100
- **Symptom**: the migration script's first run failed at `mv("experiments/checkpoints", "out/checkpoints")`.
- **Cause**: the skeleton step pre-created an empty `out/checkpoints`; shutil.move with an existing target directory stuffs the source inside the target — the guard logic caught it correctly, but the script aborted.
- **Resolution**: the skeleton no longer pre-creates that directory (the move creates it); before re-running, confirm any leftover empty directories are cleared.

### P-006 E-series --out resolved against cwd

- **Related tasks**: [T-004](#t-004-supplementary-experiment-queue-fully-complete-99), [T-003](#t-003-repository-reorganization-src--out--docs-layout)
- **Recording agent**: Agent_20260928163100
- **Symptom**: the E4 script finished 56 minutes with rc=0, yet the runner judged FAIL — the results had landed in `src/out/E4/` instead of `out/E4/`.
- **Cause**: the script's `--out out/E4` is a relative path, resolved against cwd (after the reorganization the experiment steps run with cwd=src); the reorganization only changed the default value string and did not handle runtime-relative paths.
- **Resolution**: the 5 E/B scripts insert an "anchor relative paths to the project root" patch after parse_args (`temp_scripts/script_20260929_053000_fix_out_anchor.py`); the misplaced E4/E3 results were moved back manually, and the runner's retry round recognized them as SKIPPED_DONE, so no compute was wasted. Lesson: **whenever the runner runs with cwd=X and product paths are passed on the command line, the script itself must anchor absolute paths**.

## Change ledger

> Every block-level write to this document is registered here; block ownership follows this table.
> Agent id = `Agent_yyyyMMddHHmmss`, the timestamp being the completion moment of that agent's first task in this project; content predating the id rule is marked "unregistered".

| Time | Agent | Section | Operation | Summary |
|---|---|---|---|---|
| 2026-10-09 22:10 | Agent_20261009213000 | All | Add/split | Split this document into a bilingual pair: `handover.md` (English) + `handover.zh-CN.md` (Chinese), content equivalent; restored the missing P-001 heading in both versions |
| 2026-10-09 22:00 | Agent_20261009213000 | All | Add/modify | T-011: deleted temp/ and the manuscript copies and the two revision-guidance docs in docs/ (backed up first); out/ fully untracked from git and gitignored; the 5 `--out` anchoring patches consolidated into lib.utils.anchor_project_path; the s4 m3 branch restructured; experiment_e5_deep → _e5_deep_predict; scripts/ committed for the first time |
| 2026-10-08 15:40 | Agent_20261008153000 | All | Add/clean | T-009: figure-swap review (MD5 + PDF rendering), rewrote the revision guide as the remaining checklist; deleted the JBE package, mock/origin+add_noise, out/figures, and 3 old guidance docs; consolidated 7 temp_scripts into scripts/ with smoke tests; removed the scripts/ ignore from .gitignore; added P-007 |
| 2026-09-30 15:50 | Agent_20260930 | Ledger | Append | Generated `docs/JBE修改包/` (README + guide copy + 9 plotting CSVs, script `temp_scripts/script_20260930_153000_make_csvs.py`), copied wholesale to `E:\OneDrive\预制构件分割\JBE\JBE修改包\`; note that phase2 history fields are val_cm/val_rs_iou (differently named from phase1's cm) |
| 2026-09-30 14:30 | Agent_20260930 | History/Current task/Ledger | Append | T-007: second review, guide rewritten as the remaining-items version (R1–R14 + figures F1–F9); found the §4.2.1 duplicate paragraph, the unreplaced Table 6, the §4.6 figure-number clash, and Fig 17(a) to be drawn fresh; verified Fig 7b/8b can use the history cm directly |
| 2026-09-30 10:50 | Agent_20260930 | History/Current task/Ledger | Append | T-006: paragraph-by-paragraph guide `docs/稿件逐段修改指导.md` (7 draft errata, the M2 tie finding, and the JBE length audit: no body limit / 250-word abstract); current task now points at the guide |
| 2026-09-29 19:40 | Agent_20260928163100 | Current task/Ledger | Append | Completed the final wording pass `docs/稿件修改文字_终稿.md` per the humanize-writing spec (M1–M12, paragraph-replaceable), fixing the JBE submission stance |
| 2026-09-29 14:10 | Agent_20260928163100 | History/Current task/Pitfalls/Ledger | Append/Settle | Settled T-004 (queue ALL DONE 9/9); current task moved to figures + body-revision checklist; added P-006 |
| 2026-09-28 17:55 | Agent_20260928163100 | All | Create | Initial handover document: snapshot refreshed to the post-reorganization layout; T-001~T-003, P-001~P-005 |
