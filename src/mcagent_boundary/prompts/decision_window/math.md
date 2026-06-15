Dataset focus: mathematical problems involving algebra, symbolic manipulation, arithmetic, or contest reasoning.

Use the fixed reasoning attempt as the current state. Prefer ANSWER when the state already gives the requested result reliably. Choose CALCULATE only when executable Python or sympy would materially reduce arithmetic or symbolic error.

Good example 1: choose ANSWER from a sufficient fixed state

User state:
Question: Solve for x: x + 7 = 12.
Current allowed actions: ANSWER, CALCULATE
Original student reasoning attempt: Subtracting 7 from both sides gives x = 5. This is a one-step equation.

Good output:
{
  "action": "ANSWER",
  "confidence": 0.97,
  "brief_rationale": "The fixed state already gives the requested solution.",
  "action_input": {"answer": "5"}
}

Good example 2: choose CALCULATE when symbolic verification is useful

User state:
Question: Find the positive solutions to x^2 - 10x + 21 = 0.
Current allowed actions: ANSWER, CALCULATE
Original student reasoning attempt: The quadratic should factor into two linear terms, but I want to make sure I do not omit one root.

Good output:
{
  "action": "CALCULATE",
  "confidence": 0.88,
  "brief_rationale": "The fixed state flags a possible incomplete root set, so symbolic solving is useful.",
  "action_input": {"expression": "import sympy as sp\nx = sp.symbols('x')\nsp.solve(sp.Eq(x**2 - 10*x + 21, 0), x)"}
}
