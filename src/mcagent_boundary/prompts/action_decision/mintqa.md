Dataset focus: factual questions.

Answer directly when the fact is common and stable. Use SEARCH only when the fact is obscure, entity-specific, current, or likely to be hallucinated from memory.

Good example 1: choose ANSWER for stable known facts

Question: Who wrote the novel Pride and Prejudice?
Current allowed actions: ANSWER, SEARCH

Good output:
{
  "reasoning": {
    "attempt": "This is a stable, widely known literary fact. Pride and Prejudice was written by Jane Austen.",
    "uncertainty_summary": "No meaningful uncertainty; external evidence is not necessary.",
    "need_external_help": false
  },
  "decision": {
    "action": "ANSWER",
    "confidence": 0.97,
    "brief_rationale": "The fact is stable and well-known.",
    "action_input": {"answer": "Jane Austen"}
  }
}

Good example 2: choose SEARCH for current or uncertain facts

Question: Who is the current CEO of the company that owns GitHub?
Current allowed actions: ANSWER, SEARCH

Good output:
{
  "reasoning": {
    "attempt": "GitHub is owned by Microsoft, but the current CEO is a corporate leadership fact that could change.",
    "uncertainty_summary": "The current role holder should be verified.",
    "need_external_help": true
  },
  "decision": {
    "action": "SEARCH",
    "confidence": 0.93,
    "brief_rationale": "The answer depends on current leadership information.",
    "action_input": {"query": "current CEO of Microsoft company that owns GitHub"}
  }
}
