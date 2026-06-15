Good example 1:

Input:
Question:
Who wrote the novel Pride and Prejudice?

Available actions:
["ANSWER", "SEARCH"]

Return exactly 2 candidates with different actions. ANSWER must appear.

Output:
{
  "reasoning": {
    "attempt": "This is a stable, widely known literary fact. Pride and Prejudice was written by Jane Austen.",
    "uncertainty_summary": "No meaningful uncertainty; external evidence is not necessary.",
    "need_external_help": false
  },
  "candidates": [
    {
      "rank": 1,
      "action": "ANSWER",
      "confidence": 0.97,
      "brief_rationale": "The fact is stable and well-known.",
      "action_input": {
        "answer": "Jane Austen"
      }
    },
    {
      "rank": 2,
      "action": "SEARCH",
      "confidence": 0.35,
      "brief_rationale": "Search could verify the fact, but it is unnecessary.",
      "action_input": {
        "query": "Pride and Prejudice author"
      }
    }
  ]
}

Good example 2:

Input:
Question:
Who is the current CEO of the company that owns GitHub?

Available actions:
["ANSWER", "SEARCH"]

Return exactly 2 candidates with different actions. ANSWER must appear.

Output:
{
  "reasoning": {
    "attempt": "The question depends on current corporate leadership. This information may change, so external verification is needed.",
    "uncertainty_summary": "The current CEO could have changed after my training data.",
    "need_external_help": true
  },
  "candidates": [
    {
      "rank": 1,
      "action": "SEARCH",
      "confidence": 0.94,
      "brief_rationale": "The answer depends on current company leadership and should be verified.",
      "action_input": {
        "query": "current CEO of Microsoft company that owns GitHub"
      }
    },
    {
      "rank": 2,
      "action": "ANSWER",
      "confidence": 0.45,
      "brief_rationale": "A direct answer may be stale because the role is time-sensitive.",
      "action_input": {
        "answer": "The company that owns GitHub is Microsoft, but the current CEO should be verified with up-to-date sources."
      }
    }
  ]
}
