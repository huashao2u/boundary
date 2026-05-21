Finalize-after-tool guidance for GSM8K arithmetic.

Purpose:
- Convert the calculator observation into the final concise numeric answer.
- Do not leave the answer only in reasoning or brief_rationale.

Required ANSWER behavior:
- If the observation has a usable `result`, choose ANSWER and set `action_input.answer` to that result or the final value derived from it.
- For whole-number float results such as `8.0`, answer with `8` unless the question asks for a decimal.
- Do not include units unless the question explicitly asks for them.

One-shot finalize example:
Original question: A box has 25 peaches in each of 5 baskets. After 5 peaches are eaten, the rest are packed into boxes of 15. How many boxes are there?
Current action taken: CALCULATE
Current action input: {"expression": "(25 * 5 - 5) / 15"}
Current tool observation: {"result": "8.0", "backend": "python_sandbox"}
Good final JSON:
{
  "reasoning": {
    "attempt": "The calculator result is 8.0 boxes, which is a whole number.",
    "observation_summary": "The observation establishes the final count as 8.",
    "remaining_uncertainty": "None."
  },
  "final_decision": {
    "action": "ANSWER",
    "confidence": 1.0,
    "brief_rationale": "The tool result directly answers the question.",
    "action_input": {"answer": "8"}
  }
}
