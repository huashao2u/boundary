Question: {question}

Gold/reference for offline utility judging only:
{gold_reference}

Student reasoning attempt:
{reason_attempt}

Student uncertainty summary:
{uncertainty_summary}

Student ranked candidates (DO NOT execute them — score helpfulness under the rubric only):
{student_candidates}

Process features (system-measured, not model self-reports):
{process_features}

Known semantic hints (rule-derived, you may refine them):
{semantic_hints}

Semantic tag evidence (source/text/confidence diagnostics):
{semantic_tag_evidence}

Dataset / boundary metadata (for your context only, not for the student):
{dataset_boundary}

Important: You must first output candidate_evidence for every candidate, then candidate_utility.
Follow the payload-first rules exactly. Score every candidate as emitted by the student.
Do NOT invent tool outputs. Do NOT use gold/reference to create or repair action_input. Do NOT make absolute external factual claims.
Return JSON only.
