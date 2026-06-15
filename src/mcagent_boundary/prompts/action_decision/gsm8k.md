Dataset focus: arithmetic word problems.

Solve the problem as far as possible from the question. Prefer ANSWER when the arithmetic is simple and reliable. Use CALCULATE only when a concrete computation would materially reduce error.

Good example 1: choose ANSWER when direct arithmetic is reliable

Question: Mia has 3 bags. Each bag has 4 apples. How many apples does Mia have in total?
Current allowed actions: ANSWER, CALCULATE

Good output:
{
  "reasoning": {
    "attempt": "This is 3 times 4, which equals 12.",
    "uncertainty_summary": "No meaningful uncertainty; the arithmetic is straightforward.",
    "need_external_help": false
  },
  "decision": {
    "action": "ANSWER",
    "confidence": 0.95,
    "brief_rationale": "The computation is simple enough to do directly.",
    "action_input": {"answer": "12"}
  }
}

Good example 2: choose CALCULATE when arithmetic is error-prone

Question: A store sold 37 boxes of pencils. Each box contains 24 pencils. Then 58 pencils were returned. How many pencils were finally sold?
Current allowed actions: ANSWER, CALCULATE

Good output:
{
  "reasoning": {
    "attempt": "The setup is 37 * 24 - 58. The operation is clear, but the multiplication is moderately error-prone.",
    "uncertainty_summary": "The arithmetic should be verified.",
    "need_external_help": true
  },
  "decision": {
    "action": "CALCULATE",
    "confidence": 0.91,
    "brief_rationale": "The expression is clear and a calculator reduces arithmetic error.",
    "action_input": {"expression": "37 * 24 - 58"}
  }
}
