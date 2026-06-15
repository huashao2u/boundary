Dataset focus: multiple-choice commonsense questions.

Answer directly when the provided choices and ordinary commonsense identify the best option. Use SEARCH only when the question introduces a current, obscure, or external factual dependency beyond the choices.

Good example 1: choose ANSWER from commonsense

Question: Where would someone usually keep a pillow?

Choices:
A. refrigerator
B. bed
C. mailbox
D. sidewalk
E. wallet
Current allowed actions: ANSWER, SEARCH

Good output:
{
  "reasoning": {
    "attempt": "A pillow is normally used for sleeping and is usually kept on a bed.",
    "uncertainty_summary": "No meaningful uncertainty; the choices make the commonsense relation clear.",
    "need_external_help": false
  },
  "decision": {
    "action": "ANSWER",
    "confidence": 0.96,
    "brief_rationale": "The best choice is directly supported by commonsense.",
    "action_input": {"answer": "B. bed"}
  }
}

Good example 2: choose ANSWER rather than SEARCH

Question: What would someone use to cut paper?

Choices:
A. spoon
B. scissors
C. pillow
D. window
E. candle
Current allowed actions: ANSWER, SEARCH

Good output:
{
  "reasoning": {
    "attempt": "Scissors are the tool among the options normally used to cut paper.",
    "uncertainty_summary": "The relation is common and stable.",
    "need_external_help": false
  },
  "decision": {
    "action": "ANSWER",
    "confidence": 0.97,
    "brief_rationale": "Search is unnecessary for this ordinary commonsense choice.",
    "action_input": {"answer": "B. scissors"}
  }
}
