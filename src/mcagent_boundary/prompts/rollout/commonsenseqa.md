Good example 1:

Input:
Question:
Where would someone usually keep a pillow?

Choices:
A. refrigerator
B. bed
C. mailbox
D. sidewalk
E. wallet

Available actions:
["ANSWER", "SEARCH"]

Return exactly 2 candidates with different actions. ANSWER must appear.

Output:
{
  "reasoning": {
    "attempt": "A pillow is normally used for sleeping and is usually kept on a bed. The choices make the commonsense answer clear.",
    "uncertainty_summary": "No external evidence is needed.",
    "need_external_help": false
  },
  "candidates": [
    {
      "rank": 1,
      "action": "ANSWER",
      "confidence": 0.96,
      "brief_rationale": "The provided choices support a direct commonsense answer.",
      "action_input": {
        "answer": "B. bed"
      }
    },
    {
      "rank": 2,
      "action": "SEARCH",
      "confidence": 0.25,
      "brief_rationale": "Search is unnecessary for this common everyday relation.",
      "action_input": {
        "query": "where is a pillow usually kept"
      }
    }
  ]
}

Good example 2:

Input:
Question:
What would someone use to cut paper?

Choices:
A. spoon
B. scissors
C. pillow
D. window
E. candle

Available actions:
["ANSWER", "SEARCH"]

Return exactly 2 candidates with different actions. ANSWER must appear.

Output:
{
  "reasoning": {
    "attempt": "Scissors are the tool among the choices normally used to cut paper.",
    "uncertainty_summary": "The relation is common and stable.",
    "need_external_help": false
  },
  "candidates": [
    {
      "rank": 1,
      "action": "ANSWER",
      "confidence": 0.97,
      "brief_rationale": "This is directly answerable from commonsense and the choices.",
      "action_input": {
        "answer": "B. scissors"
      }
    },
    {
      "rank": 2,
      "action": "SEARCH",
      "confidence": 0.2,
      "brief_rationale": "External search would add cost without improving the answer.",
      "action_input": {
        "query": "tool used to cut paper"
      }
    }
  ]
}
