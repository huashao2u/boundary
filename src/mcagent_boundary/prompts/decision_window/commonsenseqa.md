Dataset focus: multiple-choice commonsense questions.

Use the fixed reasoning attempt as the current state. Prefer ANSWER when the state identifies a plausible choice from everyday commonsense. Choose SEARCH only if the fixed state explicitly shows that the answer depends on external, current, or obscure factual evidence not available from the choices.

Good example 1: choose ANSWER for ordinary commonsense

User state:
Question: Where would someone usually keep a pillow?

Choices:
A. refrigerator
B. bed
C. mailbox
D. sidewalk
E. wallet
Current allowed actions: ANSWER, SEARCH
Original student reasoning attempt: A pillow is used for sleeping, so the best choice is bed.

Good output:
{
  "action": "ANSWER",
  "confidence": 0.96,
  "brief_rationale": "The fixed state already supports the commonsense choice.",
  "action_input": {"answer": "B. bed"}
}

Good example 2: avoid unnecessary SEARCH

User state:
Question: What would someone use to cut paper?

Choices:
A. spoon
B. scissors
C. pillow
D. window
E. candle
Current allowed actions: ANSWER, SEARCH
Original student reasoning attempt: Scissors are commonly used to cut paper; the other choices are not suitable tools.

Good output:
{
  "action": "ANSWER",
  "confidence": 0.97,
  "brief_rationale": "The answer follows from ordinary commonsense and the provided choices.",
  "action_input": {"answer": "B. scissors"}
}
