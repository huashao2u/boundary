Dataset focus: finalize after a clarification observation.

Use the previous action history and the current observation. Prefer ANSWER when the clarification supplies enough information for a useful response. Choose CLARIFY again only when a critical slot is still missing and CLARIFY is allowed.

Good example 1: choose ANSWER when clarification resolves the missing slots

Original question: Book a table for dinner tomorrow.
Current action taken: CLARIFY
Current action input: {"question": "Which restaurant, what time, and how many people should the dinner reservation be for?"}
Current tool observation: {"user_reply": "At 7 PM for four people at Sakura in Seattle.", "status": "clarified"}

Good output:
{
  "reasoning": {
    "attempt": "The clarification supplies the restaurant, time, party size, and city.",
    "observation_summary": "The user wants Sakura in Seattle tomorrow at 7 PM for four people.",
    "remaining_uncertainty": "None for a useful acknowledgement."
  },
  "final_decision": {
    "action": "ANSWER",
    "confidence": 0.9,
    "brief_rationale": "The missing booking slots have been supplied.",
    "action_input": {"answer": "Got it - the dinner reservation details are Sakura in Seattle tomorrow at 7 PM for four people."}
  }
}

Good example 2: choose CLARIFY when a critical slot remains missing

Original question: Book a table for dinner tomorrow.
Current action taken: CLARIFY
Current action input: {"question": "Which restaurant, what time, and how many people should the dinner reservation be for?"}
Current tool observation: {"user_reply": "For four people at 7 PM.", "status": "partially_clarified"}

Good output:
{
  "reasoning": {
    "attempt": "The user supplied time and party size, but the restaurant or location is still missing.",
    "observation_summary": "Four people at 7 PM is known; restaurant/location is not.",
    "remaining_uncertainty": "The booking target remains unspecified."
  },
  "final_decision": {
    "action": "CLARIFY",
    "confidence": 0.86,
    "brief_rationale": "A specific booking target is still required.",
    "action_input": {"question": "Which restaurant or city should I use for the dinner reservation?"}
  }
}
