Question: {question}

Gold/reference for offline utility judging only:
{gold_reference}

Student reasoning attempt:
{reason_attempt}

Student uncertainty summary:
{uncertainty_summary}

Allowed actions for this example:
{allowed_actions}

Important:
- You may only recommend actions from this allowed action list.
- Do not penalize a candidate because an unavailable action would have been better.
- Do not mention or prefer actions that are not available for this example.

Student ranked candidates (DO NOT execute them — score helpfulness under the rubric only):
{student_candidates}

Process features (system-measured, not model self-reports):
{process_features}

Known semantic hints (rule-derived, you may refine them):
{semantic_hints}

Semantic tag evidence (source/text/confidence diagnostics):
{semantic_tag_evidence}

Task family metadata, for applying dataset-specific rubrics only:
{dataset_boundary}

Do not use the dataset name as a shortcut.
Judge only the candidate payload, gold/reference, metadata labels, and allowed actions.

Important: You must first output candidate_evidence for every candidate, then candidate_utility.
Follow the payload-first rules exactly. Score every candidate as emitted by the student.
Do NOT invent tool outputs. Do NOT use gold/reference to create or repair action_input. Do NOT make absolute external factual claims.
Return JSON only.
