Dataset focus: underspecified user intentions.

Use the fixed reasoning attempt as the current state. Prefer ANSWER when the state already contains enough information to respond helpfully. Choose CLARIFY only when a missing slot blocks a useful response.

Good example 1: choose ANSWER when the request is sufficiently specified

User state:
Question: Remind me to call my dentist tomorrow at 9 AM.
Current allowed actions: ANSWER, CLARIFY
Original student reasoning attempt: The user gave the task, target, date, and time, so no critical slot is missing.

Good output:
{
  "action": "ANSWER",
  "confidence": 0.9,
  "brief_rationale": "The fixed state contains the necessary reminder details.",
  "action_input": {"answer": "Got it - I will remind you to call your dentist tomorrow at 9 AM."}
}

Good example 2: choose CLARIFY when a critical slot is missing

User state:
Question: Book a table for dinner tomorrow.
Current allowed actions: ANSWER, CLARIFY
Original student reasoning attempt: The request lacks the restaurant, time, party size, and possibly city. I cannot make a useful booking response without those details.

Good output:
{
  "action": "CLARIFY",
  "confidence": 0.94,
  "brief_rationale": "The fixed state identifies missing booking slots that block a useful response.",
  "action_input": {"question": "Which restaurant, what time, and how many people should the dinner reservation be for?"}
}
