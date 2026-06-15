Dataset focus: finalize factual questions after search observations.

Use the current search observation directly. Prefer ANSWER when the observation contains enough evidence for a concise factual answer. Choose SEARCH again only when the observation is insufficient and a narrower query is likely to materially improve the answer.

Good example 1: choose ANSWER when the observation is sufficient

Original question: What occupation did Jane Doe have?
Current action taken: SEARCH
Current action input: {"query": "Jane Doe occupation"}
Current tool observation: {"results": ["Jane Doe was a botanist and lecturer."]}

Good output:
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

Good example 2: choose SEARCH when evidence is insufficient

Original question: Who is the current CEO of the company that owns GitHub?
Current action taken: SEARCH
Current action input: {"query": "company that owns GitHub CEO"}
Current tool observation: {"results": ["GitHub is owned by Microsoft.", "Microsoft leadership page"]}

Good output:
{
  "reasoning": {
    "attempt": "The observation identifies Microsoft as the owner but does not state the current CEO.",
    "observation_summary": "Microsoft owns GitHub; the CEO fact is not established.",
    "remaining_uncertainty": "The current CEO still needs evidence."
  },
  "final_decision": {
    "action": "SEARCH",
    "confidence": 0.82,
    "brief_rationale": "A narrower query is needed to verify the current CEO.",
    "action_input": {"query": "current CEO of Microsoft"}
  }
}
