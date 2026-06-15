Dataset focus: factual questions.

Use the fixed reasoning attempt as the current state. Prefer ANSWER when the state supports a stable, well-known fact. Choose SEARCH only when the state indicates the fact is obscure, entity-specific, current, or otherwise easy to hallucinate.

Good example 1: choose ANSWER for stable known facts

User state:
Question: Who wrote the novel Pride and Prejudice?
Current allowed actions: ANSWER, SEARCH
Original student reasoning attempt: This is a stable and widely known literary fact. Pride and Prejudice was written by Jane Austen.

Good output:
{
  "action": "ANSWER",
  "confidence": 0.97,
  "brief_rationale": "The fixed state already supports a stable known fact.",
  "action_input": {"answer": "Jane Austen"}
}

Good example 2: choose SEARCH for current or uncertain facts

User state:
Question: Who is the current CEO of the company that owns GitHub?
Current allowed actions: ANSWER, SEARCH
Original student reasoning attempt: GitHub is owned by Microsoft, but the current CEO is a time-sensitive corporate leadership fact and should be verified.

Good output:
{
  "action": "SEARCH",
  "confidence": 0.93,
  "brief_rationale": "The fixed state identifies a current leadership fact that should be verified.",
  "action_input": {"query": "current CEO of Microsoft company that owns GitHub"}
}
