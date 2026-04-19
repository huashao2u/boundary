from __future__ import annotations

import ast
import operator
from typing import Any


SAFE_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
SAFE_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_BINARY_OPS:
        return SAFE_BINARY_OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_UNARY_OPS:
        return SAFE_UNARY_OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Unsupported calculator expression.")


class CalculatorTool:
    name = "CALCULATE"

    def run(self, action_input: dict[str, Any], sample, history: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        expression = str(action_input.get("expression", "")).strip()
        if not expression:
            raise ValueError("CalculatorTool requires `expression`.")
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree)
        observation = {"expression": expression, "result": str(result)}
        return observation, False, {"helpful": True}
