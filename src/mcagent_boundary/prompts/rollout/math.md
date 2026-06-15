Good example 1:

Input:
Question:
Solve for x: x + 7 = 12.

Available actions:
["ANSWER", "CALCULATE"]

Return exactly 2 candidates with different actions. ANSWER must appear.

Output:
{
  "reasoning": {
    "attempt": "Subtract 7 from both sides: x = 12 - 7 = 5.",
    "uncertainty_summary": "No uncertainty; this is a one-step linear equation.",
    "need_external_help": false
  },
  "candidates": [
    {
      "rank": 1,
      "action": "ANSWER",
      "confidence": 0.97,
      "brief_rationale": "The equation is simple enough to solve directly.",
      "action_input": {
        "answer": "5"
      }
    },
    {
      "rank": 2,
      "action": "CALCULATE",
      "confidence": 0.75,
      "brief_rationale": "A symbolic solver could verify the result, but it is unnecessary.",
      "action_input": {
        "expression": "12 - 7"
      }
    }
  ]
}

Good example 2:

Input:
Question:
Find the positive solution to x^2 - 10x + 21 = 0.

Available actions:
["ANSWER", "CALCULATE"]

Return exactly 2 candidates with different actions. ANSWER must appear.

Output:
{
  "reasoning": {
    "attempt": "The quadratic factors as (x - 3)(x - 7), so the positive solutions are 3 and 7. The question asks for the positive solution, but there are two positive solutions, so calculation/symbolic solving can clarify the exact solution set.",
    "uncertainty_summary": "The equation has two positive roots; a symbolic calculation is useful to avoid giving an incomplete answer.",
    "need_external_help": true
  },
  "candidates": [
    {
      "rank": 1,
      "action": "CALCULATE",
      "confidence": 0.9,
      "brief_rationale": "Symbolic solving can return the complete root set.",
      "action_input": {
        "expression": "import sympy as sp\nx = sp.symbols('x')\nsp.solve(sp.Eq(x**2 - 10*x + 21, 0), x)"
      }
    },
    {
      "rank": 2,
      "action": "ANSWER",
      "confidence": 0.55,
      "brief_rationale": "A direct answer risks giving only one of the two positive roots.",
      "action_input": {
        "answer": "3 and 7"
      }
    }
  ]
}
