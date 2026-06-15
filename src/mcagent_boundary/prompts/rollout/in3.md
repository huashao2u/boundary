Good example 1:

Input:
Question:
Book a reminder for me to call my dentist tomorrow at 9 AM.

Available actions:
["ANSWER", "CLARIFY"]

Return exactly 2 candidates with different actions. ANSWER must appear.

Output:
{
  "reasoning": {
    "attempt": "The user provided the task, target, date, and time. There is no missing critical slot for deciding the next response.",
    "uncertainty_summary": "No critical information is missing.",
    "need_external_help": false
  },
  "candidates": [
    {
      "rank": 1,
      "action": "ANSWER",
      "confidence": 0.9,
      "brief_rationale": "The request is sufficiently specified.",
      "action_input": {
        "answer": "Got it - I will remind you to call your dentist tomorrow at 9 AM."
      }
    },
    {
      "rank": 2,
      "action": "CLARIFY",
      "confidence": 0.25,
      "brief_rationale": "Clarification is not necessary because the required details are already present.",
      "action_input": {
        "question": "Do you want to add any extra details to the reminder?"
      }
    }
  ]
}

Good example 2:

Input:
Question:
Book a table for dinner tomorrow.

Available actions:
["ANSWER", "CLARIFY"]

Return exactly 2 candidates with different actions. ANSWER must appear.

Output:
{
  "reasoning": {
    "attempt": "The request lacks critical booking information such as restaurant, time, party size, or location. A useful answer cannot complete the task without at least the missing core details.",
    "uncertainty_summary": "Restaurant, time, party size, and location are missing.",
    "need_external_help": true
  },
  "candidates": [
    {
      "rank": 1,
      "action": "CLARIFY",
      "confidence": 0.95,
      "brief_rationale": "A critical slot is missing, so clarification is needed before proceeding.",
      "action_input": {
        "question": "Which restaurant, what time, and how many people should the dinner reservation be for?"
      }
    },
    {
      "rank": 2,
      "action": "ANSWER",
      "confidence": 0.25,
      "brief_rationale": "A direct response would be incomplete because key booking details are missing.",
      "action_input": {
        "answer": "I need the restaurant, time, and party size before the reservation can be made."
      }
    }
  ]
}
