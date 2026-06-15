Dataset focus: finalize arithmetic word problems after a tool observation.

Use the previous action history and the current observation. Prefer ANSWER when the observation plus simple reasoning gives the final requested value. Choose CALCULATE again only when another non-trivial computation is necessary and allowed.

Good example 1: choose ANSWER when the observation is sufficient

Original question: A box has 25 peaches in each of 5 baskets. After 5 peaches are eaten, the rest are packed into boxes of 15. How many boxes are there?
Current action taken: CALCULATE
Current action input: {"expression": "(25 * 5 - 5) / 15"}
Current tool observation: {"result": "8.0", "backend": "python_sandbox"}

Good output:
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

Good example 2: choose CALCULATE when one more computation is necessary

Original question: A shop has 6 shelves with 18 cans on each shelf. It sells 27 cans, then packs the rest equally into 9 boxes. How many cans go in each box?
Current action taken: CALCULATE
Current action input: {"expression": "6 * 18 - 27"}
Current tool observation: {"result": "81", "backend": "python_sandbox"}

Good output:
{
  "reasoning": {
    "attempt": "The observation gives the cans remaining, but the question asks for cans per box, so division by 9 is still needed.",
    "observation_summary": "There are 81 cans remaining.",
    "remaining_uncertainty": "The per-box count has not been computed."
  },
  "final_decision": {
    "action": "CALCULATE",
    "confidence": 0.9,
    "brief_rationale": "One more arithmetic step is required to answer the requested quantity.",
    "action_input": {"expression": "81 / 9"}
  }
}
