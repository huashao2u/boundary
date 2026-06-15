Dataset focus: mathematical problems involving arithmetic, algebra, symbolic manipulation, or contest reasoning.

Solve the problem as far as possible from the question. Prefer ANSWER when the derivation is short and reliable. Use CALCULATE only when executable Python or sympy would materially reduce computation or symbolic error.

Good example 1: choose ANSWER when the derivation is reliable

Question: Find the positive solutions to x^2 - 10x + 21 = 0.
Current allowed actions: ANSWER, CALCULATE

Good output:
{
  "reasoning": {
    "attempt": "The quadratic factors as (x - 3)(x - 7), giving two positive roots, 3 and 7.",
    "uncertainty_summary": "The factorization is straightforward; no external tool is needed.",
    "need_external_help": false
  },
  "decision": {
    "action": "ANSWER",
    "confidence": 0.88,
    "brief_rationale": "The root set can be derived directly.",
    "action_input": {"answer": "3 and 7"}
  }
}

Good example 2: choose CALCULATE when symbolic solving reduces error

Question: Solve x^4 - 5*x^2 + 4 = 0.
Current allowed actions: ANSWER, CALCULATE

Good output:
{
  "reasoning": {
    "attempt": "This is a polynomial equation where symbolic solving can avoid missing roots.",
    "uncertainty_summary": "The full root set should be verified.",
    "need_external_help": true
  },
  "decision": {
    "action": "CALCULATE",
    "confidence": 0.9,
    "brief_rationale": "A symbolic calculation can return the complete root set.",
    "action_input": {"expression": "import sympy as sp\nx = sp.symbols('x')\nsp.solve(sp.Eq(x**4 - 5*x**2 + 4, 0), x)"}
  }
}
