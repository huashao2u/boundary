Dataset focus: finalize multiple-choice commonsense questions after a tool observation.

Use any observation as optional context, but prefer ANSWER when the commonsense choice is clear from the original question and choices. Choose SEARCH again only if the observation shows that the question actually depends on external factual evidence.

Good example 1: choose ANSWER when the observation is unnecessary

Original question: Where would someone usually keep a pillow?

Choices:
A. refrigerator
B. bed
C. mailbox
D. sidewalk
E. wallet
Current action taken: SEARCH
Current action input: {"query": "where is a pillow usually kept"}
Current tool observation: {"results": ["A pillow is commonly used on a bed for sleeping."]}

Good output:
{
  "reasoning": {
    "attempt": "The observation is consistent with the commonsense answer. The best choice is bed.",
    "observation_summary": "The result supports bed as the normal place for a pillow.",
    "remaining_uncertainty": "None."
  },
  "final_decision": {
    "action": "ANSWER",
    "confidence": 0.96,
    "brief_rationale": "The observation and choices support the same commonsense answer.",
    "action_input": {"answer": "B. bed"}
  }
}

Good example 2: choose ANSWER when search did not add useful evidence

Original question: What would someone use to cut paper?

Choices:
A. spoon
B. scissors
C. pillow
D. window
E. candle
Current action taken: SEARCH
Current action input: {"query": "tool used to cut paper"}
Current tool observation: {"results": []}

Good output:
{
  "reasoning": {
    "attempt": "Even without useful search results, the provided choices make the commonsense answer clear: scissors are used to cut paper.",
    "observation_summary": "The search observation did not add evidence.",
    "remaining_uncertainty": "None; the choices are sufficient."
  },
  "final_decision": {
    "action": "ANSWER",
    "confidence": 0.94,
    "brief_rationale": "The original choices are enough for a direct commonsense answer.",
    "action_input": {"answer": "B. scissors"}
  }
}
