Good example 1:

Input:
Question:
Mia has 3 bags. Each bag has 4 apples. How many apples does Mia have in total?

Available actions:
["ANSWER", "CALCULATE"]

Return exactly 2 candidates with different actions. ANSWER must appear.

Output:
{
  "reasoning": {
    "attempt": "This is a simple multiplication: 3 bags times 4 apples per bag equals 12 apples.",
    "uncertainty_summary": "No uncertainty; the arithmetic is straightforward.",
    "need_external_help": false
  },
  "candidates": [
    {
      "rank": 1,
      "action": "ANSWER",
      "confidence": 0.95,
      "brief_rationale": "The calculation is simple enough to do directly.",
      "action_input": {
        "answer": "12"
      }
    },
    {
      "rank": 2,
      "action": "CALCULATE",
      "confidence": 0.85,
      "brief_rationale": "A calculator could verify the multiplication, but it is unnecessary.",
      "action_input": {
        "expression": "3 * 4"
      }
    }
  ]
}

Good example 2:

Input:
Question:
A store sold 37 boxes of pencils. Each box contains 24 pencils. Then 58 pencils were returned. How many pencils were finally sold?

Available actions:
["ANSWER", "CALCULATE"]

Return exactly 2 candidates with different actions. ANSWER must appear.

Output:
{
  "reasoning": {
    "attempt": "The answer requires multiplying 37 by 24 and then subtracting 58. A calculation would reduce arithmetic error.",
    "uncertainty_summary": "The reasoning is clear, but the arithmetic is moderately error-prone.",
    "need_external_help": true
  },
  "candidates": [
    {
      "rank": 1,
      "action": "CALCULATE",
      "confidence": 0.92,
      "brief_rationale": "The concrete arithmetic should be executed to avoid multiplication or subtraction errors.",
      "action_input": {
        "expression": "37 * 24 - 58"
      }
    },
    {
      "rank": 2,
      "action": "ANSWER",
      "confidence": 0.55,
      "brief_rationale": "A direct answer is possible but more error-prone without calculation.",
      "action_input": {
        "answer": "830"
      }
    }
  ]
}
