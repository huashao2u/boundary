Dataset focus: underspecified user intentions.

Decide whether the user request has enough information for a useful response. Prefer ANSWER when a helpful response can be given directly. Use CLARIFY only when a missing critical slot blocks the response.

Good example 1: choose ANSWER when the request is sufficiently specified

Question: Remind me to call my dentist tomorrow at 9 AM.
Current allowed actions: ANSWER, CLARIFY

Good output:
{
  "reasoning": {
    "attempt": "The user provided the task, target, date, and time.",
    "uncertainty_summary": "No critical information is missing.",
    "need_external_help": false
  },
  "decision": {
    "action": "ANSWER",
    "confidence": 0.9,
    "brief_rationale": "The reminder request is sufficiently specified.",
    "action_input": {"answer": "Got it - I will remind you to call your dentist tomorrow at 9 AM."}
  }
}

Good example 2: choose CLARIFY when a critical slot is missing

Question: Book a table for dinner tomorrow.
Current allowed actions: ANSWER, CLARIFY

Good output:
{
  "reasoning": {
    "attempt": "The request lacks the restaurant, time, party size, and possibly the city. Those details materially change the booking.",
    "uncertainty_summary": "Critical booking slots are missing.",
    "need_external_help": true
  },
  "decision": {
    "action": "CLARIFY",
    "confidence": 0.94,
    "brief_rationale": "A useful booking response requires the missing slots.",
    "action_input": {"question": "Which restaurant, what time, and how many people should the dinner reservation be for?"}
  }
}
