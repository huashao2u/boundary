Finalize-after-tool guidance for factual questions.

Purpose:
- Finish the question using your own reasoning and the search observation.
- Prefer ANSWER when the observation provides enough evidence for a concise factual answer.
- Do not invent unsupported facts beyond the observation.
- Do not leave the answer only in reasoning or brief_rationale.

Required ANSWER behavior:
- If the search result contains the answer, choose ANSWER and put the final response in `action_input.answer`.
- If the search result is insufficient and SEARCH is still allowed, choose SEARCH with a narrower query only when another search is likely to materially improve the answer.
- If no reliable answer is available after the allowed search step, choose ANSWER with a brief uncertainty statement in `action_input.answer`.

One-shot finalize example:
Original question: What occupation did Jane Doe have?
Current action taken: SEARCH
Current action input: {"query": "Jane Doe occupation"}
Current tool observation: {"results": ["Jane Doe was a botanist and lecturer."]}
Good final JSON:
{
  "reasoning": {
    "attempt": "The search result identifies the occupation directly.",
    "observation_summary": "Jane Doe was a botanist and lecturer.",
    "remaining_uncertainty": "None."
  },
  "final_decision": {
    "action": "ANSWER",
    "confidence": 0.9,
    "brief_rationale": "The observation directly answers the factual question.",
    "action_input": {"answer": "Jane Doe was a botanist and lecturer."}
  }
}
