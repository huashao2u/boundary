Finalize-after-tool guidance for mathematical problems.

Purpose:
- Finish the problem using your own mathematical reasoning and the calculator or symbolic observation.
- Prefer ANSWER when the requested result can be derived from the history and observation.
- Continue one short reasoning step when the observation is intermediate, unsimplified, or uses Python semantics that differ from contest math notation.
- Do not leave the final answer only in reasoning or brief_rationale.

Required ANSWER behavior:
- If the observation is sufficient, choose ANSWER and put the final response in `action_input.answer`.
- If the observation is an expression such as `101/20` and the question asks for the closest whole number, compute the requested final value and answer that value.
- If the observation is a whole-number float such as `263.000000000000`, answer with `263`.
- If the observation is wrong because Python interpreted the expression differently from standard math, reason from the original question and answer the intended math result when clear.
- If another calculation is necessary and CALCULATE is allowed, choose CALCULATE with a complete executable expression.
- Do not choose CALCULATE again when the remaining step is simple simplification, rounding, formatting, or extracting the requested value.

One-shot finalize example:
Original question: The ratio (10^2000 + 10^2002) / (10^2001 + 10^2001) is closest to which whole number?
Current action taken: CALCULATE
Current action input: {"expression": "from sympy import Rational; Rational(10**2000 + 10**2002, 2*10**2001)"}
Current tool observation: {"result": "101/20", "backend": "python_sandbox"}
Good final JSON:
{
  "reasoning": {
    "attempt": "The observation gives 101/20, which is 5.05. The closest whole number is 5.",
    "observation_summary": "The calculated ratio is 101/20.",
    "remaining_uncertainty": "None."
  },
  "final_decision": {
    "action": "ANSWER",
    "confidence": 1.0,
    "brief_rationale": "The observation is sufficient after rounding to the requested whole number.",
    "action_input": {"answer": "5"}
  }
}
