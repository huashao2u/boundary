Dataset focus: finalize mathematical problems after calculator or symbolic observations.

Use the previous action history and the current observation. Prefer ANSWER when the observation plus simple reasoning gives the requested result. Choose CALCULATE again only when another non-trivial computation is necessary and allowed.

Good example 1: choose ANSWER when the observation is sufficient after simple reasoning

Original question: The ratio (10^2000 + 10^2002) / (10^2001 + 10^2001) is closest to which whole number?
Current action taken: CALCULATE
Current action input: {"expression": "from sympy import Rational; Rational(10**2000 + 10**2002, 2*10**2001)"}
Current tool observation: {"result": "101/20", "backend": "python_sandbox"}

Good output:
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

Good example 2: choose CALCULATE when the observation is only intermediate

Original question: If x satisfies x^2 - 5x + 6 = 0, what is the sum of the squares of all real solutions?
Current action taken: CALCULATE
Current action input: {"expression": "import sympy as sp\nx = sp.symbols('x')\nsp.solve(sp.Eq(x**2 - 5*x + 6, 0), x)"}
Current tool observation: {"result": "[2, 3]", "backend": "python_sandbox"}

Good output:
{
  "reasoning": {
    "attempt": "The observation gives the roots, but the requested value is the sum of their squares.",
    "observation_summary": "The real solutions are 2 and 3.",
    "remaining_uncertainty": "The sum of squares still needs to be computed."
  },
  "final_decision": {
    "action": "CALCULATE",
    "confidence": 0.9,
    "brief_rationale": "One more computation is needed for the requested expression.",
    "action_input": {"expression": "2**2 + 3**2"}
  }
}
