Dataset focus: arithmetic word problems.

Use the fixed reasoning attempt as the current state. Prefer ANSWER when that state already supports a reliable final numeric answer. Choose CALCULATE only when the state identifies the right arithmetic but the computation is error-prone or not yet verified.

Good example 1: choose ANSWER from a sufficient fixed state

User state:
Question: Mia has 3 bags. Each bag has 4 apples. How many apples does Mia have in total?
Current allowed actions: ANSWER, CALCULATE
Original student reasoning attempt: This is 3 times 4, which equals 12. The arithmetic is straightforward.

Good output:
{
  "action": "ANSWER",
  "confidence": 0.95,
  "brief_rationale": "The fixed state already contains a reliable final value.",
  "action_input": {"answer": "12"}
}

Good example 2: choose CALCULATE when verification is useful

User state:
Question: A store sold 37 boxes of pencils. Each box contains 24 pencils. Then 58 pencils were returned. How many pencils were finally sold?
Current allowed actions: ANSWER, CALCULATE
Original student reasoning attempt: The total is 37 times 24, then subtract 58. I know the setup, but the multiplication could be easy to slip on.

Good output:
{
  "action": "CALCULATE",
  "confidence": 0.91,
  "brief_rationale": "The fixed state has the right expression but the arithmetic should be verified.",
  "action_input": {"expression": "37 * 24 - 58"}
}
