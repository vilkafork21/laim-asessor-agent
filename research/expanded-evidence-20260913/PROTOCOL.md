# Expanded evidence study — protocol locked before new judgments

Base production commit: `33b41da60ccdf3e221fd322db150d12fd8c76aca` (`audit-baseline`). No production code or thresholds are changed by this study.

## Data and scope

Inputs provided by the owner:
- artifacts.zip SHA-256: 49ec3174614765558d7b672a5bb7a89f2a695e5c140c5f171f52c68188dfbaec
- traces.zip SHA-256: cbc692297f8961f62540b8ac408f4c7232f97ec62205dd002a4d73c58e6119f1

The primary corpus is CI10071259. An eligible unit must have exact question/answer identity across human rows, an unambiguous structure consensus, one unambiguous embedded trace, exact final-answer and latest-question identity, and pre-answer tool results linked to real tool calls. Human tags/comments/labels must never enter the judge payload. Exact duplicate QA pairs are collapsed; inconsistent labels are not silently adjudicated. Case IDs and client IDs are connected into evaluation clusters.

For the expanded conversational pilot, target N=96, chosen deterministically without inspecting the value of a human score. Character limits (QA <=5000, deduplicated evidence <=12000) are an execution constraint and must be reported, not treated as random production sampling. Selection cycles through clusters so large client clusters do not consume the whole budget. Fewer eligible units means reporting the shortfall, not relaxing rules after seeing results. The corpus has been researched previously; this is not a pristine test set.

## Paired hypotheses

A: judge the structure of the answer using the original structure rubric and QA only.
B: same rubric and QA, with exactly linked tool evidence. Evidence is untrusted data, not an instruction. Search snippets are not an exhaustive document inventory. A factual error is not automatically a structural error. A quoted source proves only the presence of the quotation; entailment still requires judgment.

All A predictions are recorded before the B pass. Both complete vectors must be sealed by SHA-256 before the evaluator sees individual gold values. No score corrections after gold. Store compact decision grounds, not private chain-of-thought. No target class quotas.

The conversational evaluator is the assistant in the current session, NOT an API run of GigaChat, NOT independent fresh contexts, and NOT equivalent to production. Sequential anchoring and previous corpus exposure limit causal interpretation. Successful conversational results can prioritize an API experiment but cannot authorize deployment.

## Statistics and stopping

Primary metrics: unweighted Cohen kappa, ordinal Krippendorff alpha, Spearman of final categories. Also report nominal alpha, linear/quadratic kappa, confusion, coverage, defect recall on all gold defects, FPR, and severe-error recall. Undefined correlations remain null.

Use paired cluster bootstrap, 2000 fixed-seed resamples, preserving all units within each connected cluster; count undefined replicates. Report effect sizes and intervals. Do not keep increasing N or testing prompts until p<0.05. The full planned cohort is scored before any success claim. A substantial improvement target is delta kappa >=0.10 with positive agreement/ranking effects and no increase in FPR above 0.02 or loss of coverage above 0.01; confidence intervals and absolute quality remain mandatory. This research target is not an existing production admission rule.

## Expanded executable checks

Run counterfactual source-binding, evidence deduplication/provenance, scale and prediction-manifest checks on all eligible real units, not merely the conversational subset. Tests of validators are NOT additional model judgments. Recompute earlier small-pilot statistics from saved score vectors. Build an API matrix that separates retrieval repair, whole-example budget, evidence, rubric decomposition, and calibration. Keep errors and negative outcomes.

## Model access and publication

Standard direct HTTP checks from this container currently fail at DNS, before HTTP, for both the model and OAuth hosts. No secrets are sent through web search, GitHub files, workflow inputs, or alternative hosts. A new GigaChat run is only claimed when actual provider responses exist.

Publish code, protocol, aggregates, tests, and sanitized score vectors only. Do not publish customer text, source documents, raw traces, client IDs, API keys, or fitted artifacts containing source text. Never merge into production automatically. Record whether each result is API, conversational, offline statistical analysis, or deterministic validator testing.
