from __future__ import annotations

import ast
import io
import math
import operator
import re
import json
import subprocess
import sys
import textwrap
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
SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "range": range,
    "sqrt": math.sqrt,
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "ceiling": math.ceil,
    "factorial": math.factorial,
    "pow": pow,
}
SAFE_CONSTANTS = {"pi": math.pi, "e": math.e, "E": math.e}
_SAFE_TEXT_RE = re.compile(r"^[0-9A-Za-z_\s\+\-\*\/\^\.\(\),=\[\]!;]+$")
_SAFE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")
_PYTHON_SANDBOX_DEFAULT_TIMEOUT_SEC = 3.0
_PYTHON_SANDBOX_DEFAULT_MEMORY_MB = 512
_PYTHON_SANDBOX_OUTPUT_MAX_CHARS = 512
_PYTHON_SANDBOX_ALLOWED_IMPORTS = {
    "collections",
    "cmath",
    "decimal",
    "datetime",
    "fractions",
    "functools",
    "itertools",
    "math",
    "numpy",
    "statistics",
    "sympy",
    "_strptime",
}
_PYTHON_SANDBOX_FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}
_PYTHON_SANDBOX_FORBIDDEN_MODULES = {
    "builtins",
    "importlib",
    "inspect",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
}


def _normalize_expression(expression: str) -> str:
    return (
        expression.replace("（", "(")
        .replace("）", ")")
        .replace("，", ",")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("π", "pi")
        .replace("^", "**")
        .strip()
    )


def _normalize_python_snippet(expression: str) -> str:
    expression = _normalize_expression(expression)
    # Python permits semicolons between simple statements, but not before
    # compound statements. Models often emit "x = 1; for ...", so normalize the
    # common math-control-flow forms into real lines before parsing.
    expression = re.sub(r";\s*(for|while|if)\b", r"\n\1", expression)
    return expression


def _evaluate(node: ast.AST, names: dict[str, Any] | None = None) -> Any:
    names = names or {}
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, names)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if node.id in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id]
        raise ValueError("Unsupported calculator expression.")
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_BINARY_OPS:
        return SAFE_BINARY_OPS[type(node.op)](_evaluate(node.left, names), _evaluate(node.right, names))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_UNARY_OPS:
        return SAFE_UNARY_OPS[type(node.op)](_evaluate(node.operand, names))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
        if node.keywords:
            raise ValueError("Unsupported calculator expression.")
        args = [_evaluate(arg, names) for arg in node.args]
        if node.func.id == "range":
            if not 1 <= len(args) <= 3:
                raise ValueError("Unsupported calculator expression.")
            if any(int(arg) != arg for arg in args):
                raise ValueError("Unsupported calculator expression.")
            return range(*[int(arg) for arg in args])
        return SAFE_FUNCTIONS[node.func.id](*args)
    if isinstance(node, ast.List):
        return [_evaluate(item, names) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate(item, names) for item in node.elts)
    if isinstance(node, ast.GeneratorExp):
        if len(node.generators) != 1:
            raise ValueError("Unsupported calculator expression.")
        generator = node.generators[0]
        if generator.ifs or not isinstance(generator.target, ast.Name):
            raise ValueError("Unsupported calculator expression.")
        target = generator.target.id
        values = _evaluate(generator.iter, names)

        def _items():
            for value in values:
                local_names = dict(names)
                local_names[target] = value
                yield _evaluate(node.elt, local_names)

        return _items()
    raise ValueError("Unsupported calculator expression.")


def _safe_eval_arithmetic(expression: str, names: dict[str, Any] | None = None) -> Any:
    tree = ast.parse(expression, mode="eval")
    generator_targets = {
        comp.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.GeneratorExp)
        for comp in node.generators
        if isinstance(comp.target, ast.Name)
    }
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Call,
        ast.List,
        ast.Tuple,
        ast.GeneratorExp,
        ast.comprehension,
        ast.Store,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError("Unsupported calculator expression.")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("Unsupported calculator expression.")
        if isinstance(node, ast.Name):
            valid_names = set(names or {}) | set(SAFE_CONSTANTS) | set(SAFE_FUNCTIONS) | generator_targets
            if node.id not in valid_names:
                raise ValueError("Unsupported calculator expression.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS:
                raise ValueError("Unsupported calculator expression.")
    return _evaluate(tree, names)


def _format_result(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _split_statements(expression: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(expression):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char in {";", "\n"} and depth == 0:
            part = expression[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    final = expression[start:].strip()
    if final:
        parts.append(final)
    return parts


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _safe_eval_value(expression: str, names: dict[str, Any] | None = None) -> Any:
    try:
        symbolic = _solve_symbolic(expression)
    except Exception:
        symbolic = None
    if symbolic is not None:
        return symbolic
    return _safe_eval_arithmetic(expression, names)


def _evaluate_print_statement(statement: str, names: dict[str, Any]) -> str | None:
    match = re.fullmatch(r"print\((.*)\)", statement.strip(), flags=re.DOTALL)
    if not match:
        return None
    values = [
        _format_result(_safe_eval_value(part, names))
        for part in _split_top_level_commas(match.group(1))
    ]
    return " ".join(values)


def _sympy_locals(symbol_names: set[str] | None = None) -> dict[str, Any]:
    import sympy as sp

    locals_dict: dict[str, Any] = {
        "sqrt": sp.sqrt,
        "root": sp.root,
        "log": sp.log,
        "ln": sp.log,
        "exp": sp.exp,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "floor": sp.floor,
        "ceiling": sp.ceiling,
        "ceil": sp.ceiling,
        "Abs": sp.Abs,
        "abs": sp.Abs,
        "factorial": sp.factorial,
        "Eq": sp.Eq,
        "pi": sp.pi,
        "E": sp.E,
    }
    for name in symbol_names or set():
        if len(name) == 1 and name.isalpha():
            locals_dict[name] = sp.Symbol(name)
    return locals_dict


def _parse_sympy_expression(expression: str, *, symbol_names: set[str]):
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        factorial_notation,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    return parse_expr(
        expression,
        local_dict=_sympy_locals(symbol_names),
        global_dict={"__builtins__": {}, "Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational},
        transformations=standard_transformations + (
            convert_xor,
            implicit_multiplication_application,
            factorial_notation,
        ),
        evaluate=True,
    )


def _equation_from_text(text: str, *, symbol_names: set[str]):
    import sympy as sp

    stripped = text.strip()
    eq_match = re.fullmatch(r"Eq\((.*)\)", stripped)
    if eq_match:
        parts = _split_top_level_commas(eq_match.group(1))
        if len(parts) == 2:
            return sp.Eq(
                _parse_sympy_expression(parts[0], symbol_names=symbol_names),
                _parse_sympy_expression(parts[1], symbol_names=symbol_names),
            )
    if "=" in stripped and "==" not in stripped:
        left, right = stripped.split("=", 1)
        return sp.Eq(
            _parse_sympy_expression(left, symbol_names=symbol_names),
            _parse_sympy_expression(right, symbol_names=symbol_names),
        )
    return _parse_sympy_expression(stripped, symbol_names=symbol_names)


def _float_from_sympy(value: Any) -> float:
    import sympy as sp

    numeric = sp.N(value)
    if getattr(numeric, "is_real", None) is False:
        raise ValueError("Unsupported calculator expression.")
    return float(numeric)


def _solve_symbolic(expression: str) -> float | None:
    if "__" in expression or not _SAFE_TEXT_RE.match(expression):
        raise ValueError("Unsupported calculator expression.")
    match = re.fullmatch(r"solve\((.*)\)\s*(?:\[0\])?", expression)
    if match:
        parts = _split_top_level_commas(match.group(1))
        if len(parts) not in {1, 2}:
            raise ValueError("Unsupported calculator expression.")
        names = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", parts[0])) - set(_sympy_locals())
        if len(parts) == 2:
            symbol_name = parts[1].strip()
        else:
            candidates = sorted(name for name in names if len(name) == 1 and name.isalpha())
            if len(candidates) != 1:
                raise ValueError("Unsupported calculator expression.")
            symbol_name = candidates[0]
        if not re.fullmatch(r"[A-Za-z]", symbol_name):
            raise ValueError("Unsupported calculator expression.")
        equation = _equation_from_text(parts[0], symbol_names={symbol_name})
    elif "=" in expression and "==" not in expression:
        names = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", expression)) - set(_sympy_locals())
        candidates = sorted(name for name in names if len(name) == 1 and name.isalpha())
        if len(candidates) != 1:
            return None
        symbol_name = candidates[0]
        equation = _equation_from_text(expression, symbol_names={symbol_name})
    else:
        names = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", expression)) - set(_sympy_locals())
        candidates = sorted(name for name in names if len(name) == 1 and name.isalpha())
        if len(candidates) != 1:
            return None
        symbol_name = candidates[0]
        equation = _parse_sympy_expression(expression, symbol_names={symbol_name})
    import sympy as sp

    solutions = sp.solve(equation, sp.Symbol(symbol_name))
    if not solutions:
        raise ValueError("Unsupported calculator expression.")
    return _float_from_sympy(solutions[0])


def evaluate_expression(expression: str) -> Any:
    expression = _normalize_expression(expression)
    if not expression or len(expression) > 500 or "__" in expression or not _SAFE_TEXT_RE.match(expression):
        raise ValueError("Unsupported calculator expression.")
    statements = _split_statements(expression)
    if len(statements) == 1:
        printed = _evaluate_print_statement(statements[0], {})
        if printed is not None:
            return printed
        return _safe_eval_value(statements[0], {})
    names: dict[str, Any] = {}
    last_value: Any = None
    output = io.StringIO()
    for statement in statements:
        printed = _evaluate_print_statement(statement, names)
        if printed is not None:
            print(printed, file=output)
            last_value = printed
        elif "=" in statement and "==" not in statement and _SAFE_NAME_RE.match(statement.split("=", 1)[0].strip()):
            name, expr = statement.split("=", 1)
            name = name.strip()
            if not _SAFE_NAME_RE.match(name):
                raise ValueError("Unsupported calculator expression.")
            names[name] = _safe_eval_value(expr.strip(), names)
            last_value = names[name]
        else:
            last_value = _safe_eval_value(statement, names)
    printed_output = output.getvalue().strip()
    if printed_output:
        return printed_output.splitlines()[-1]
    return last_value


def _python_sandbox_validate_import(node: ast.Import | ast.ImportFrom) -> None:
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    else:
        if node.level:
            raise ValueError("Relative imports are not allowed in calculator sandbox.")
        names = [node.module or ""]
    for name in names:
        root = name.split(".", 1)[0]
        if root not in _PYTHON_SANDBOX_ALLOWED_IMPORTS or root in _PYTHON_SANDBOX_FORBIDDEN_MODULES:
            raise ValueError(f"Import not allowed in calculator sandbox: {name}")


def _python_sandbox_validate_ast(tree: ast.AST) -> None:
    forbidden_nodes = (
        ast.AsyncFor,
        ast.AsyncFunctionDef,
        ast.Await,
        ast.ClassDef,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
        ast.Raise,
        ast.With,
        ast.AsyncWith,
        ast.Yield,
        ast.YieldFrom,
    )
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            raise ValueError(f"Unsupported calculator sandbox syntax: {type(node).__name__}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _python_sandbox_validate_import(node)
        if isinstance(node, ast.Name) and (
            node.id in _PYTHON_SANDBOX_FORBIDDEN_NAMES
            or node.id.split(".", 1)[0] in _PYTHON_SANDBOX_FORBIDDEN_MODULES
            or "__" in node.id
        ):
            raise ValueError(f"Name not allowed in calculator sandbox: {node.id}")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or "__" in node.attr):
            raise ValueError(f"Attribute not allowed in calculator sandbox: {node.attr}")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _PYTHON_SANDBOX_FORBIDDEN_NAMES:
                raise ValueError(f"Call not allowed in calculator sandbox: {func.id}")


def _python_sandbox_return_target(target: ast.expr) -> ast.expr | None:
    if isinstance(target, ast.Name):
        return ast.Name(id=target.id, ctx=ast.Load())
    if isinstance(target, (ast.Tuple, ast.List)):
        values: list[ast.expr] = []
        for elt in target.elts:
            converted = _python_sandbox_return_target(elt)
            if converted is None:
                return None
            values.append(converted)
        if isinstance(target, ast.Tuple):
            return ast.Tuple(elts=values, ctx=ast.Load())
        return ast.List(elts=values, ctx=ast.Load())
    return None


def _python_sandbox_prepare_code(expression: str) -> str:
    expression = _normalize_python_snippet(expression)
    if not expression or len(expression) > 4000 or "__" in expression:
        raise ValueError("Unsupported calculator sandbox expression.")
    try:
        tree = ast.parse(expression, mode="exec")
    except SyntaxError as exc:
        try:
            symbolic_result = evaluate_expression(expression)
        except Exception:
            raise exc
        return f"print({str(_format_result(symbolic_result))!r})"
    _python_sandbox_validate_ast(tree)
    if tree.body:
        last = tree.body[-1]
        value_to_print: ast.expr | None = None
        if isinstance(last, ast.Expr):
            value_to_print = last.value
            # Avoid printing twice when the model already used print(...).
            if isinstance(value_to_print, ast.Call) and isinstance(value_to_print.func, ast.Name) and value_to_print.func.id == "print":
                value_to_print = None
        elif isinstance(last, ast.Assign) and len(last.targets) == 1:
            value_to_print = _python_sandbox_return_target(last.targets[0])
        elif isinstance(last, ast.AugAssign):
            value_to_print = _python_sandbox_return_target(last.target)
        elif isinstance(last, (ast.For, ast.While)) and last.body and isinstance(last.body[-1], ast.Expr):
            value_to_print = last.body[-1].value
        if value_to_print is not None:
            tree.body.append(
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id="print", ctx=ast.Load()),
                        args=[value_to_print],
                        keywords=[],
                    )
                )
            )
            ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _extract_calculator_expression(action_input: Any) -> str:
    if isinstance(action_input, str):
        return action_input.strip()
    if not isinstance(action_input, dict):
        return str(action_input).strip()

    for key in ("expression", "expr", "code", "python", "formula", "calculation", "input", "query"):
        value = action_input.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    if len(action_input) == 1:
        value = next(iter(action_input.values()))
        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


def _python_sandbox_child_source() -> str:
    return r'''
import contextlib
import cmath
import datetime
import io
import json
import math
import resource
import sys

payload = json.loads(sys.stdin.read())
code = payload["code"]
memory_mb = int(payload.get("memory_mb", 512))
output_max_chars = int(payload.get("output_max_chars", 512))

try:
    memory_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (max(1, int(payload.get("cpu_sec", 3))), max(1, int(payload.get("cpu_sec", 3)))))
except Exception:
    pass

allowed_roots = {"collections", "cmath", "datetime", "decimal", "fractions", "functools", "itertools", "math", "numpy", "statistics", "sympy", "_strptime"}

def sandbox_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level:
        raise ImportError("relative imports are not allowed")
    root = name.split(".", 1)[0]
    if root not in allowed_roots:
        raise ImportError(f"import not allowed: {name}")
    return __import__(name, globals, locals, fromlist, level)

safe_builtins = {
    "__import__": sandbox_import,
    "ArithmeticError": ArithmeticError,
    "Exception": Exception,
    "ValueError": ValueError,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}

import sympy as sp
from sympy import Abs, Eq, Matrix, Rational, expand, factor, simplify, solve, sqrt, symbols

def sandbox_log(value, base=None):
    try:
        if base is None:
            numeric = math.log(float(value))
        else:
            numeric = math.log(float(value), float(base))
        rounded = round(numeric)
        return rounded if abs(numeric - rounded) < 1e-12 else numeric
    except Exception:
        if base is None:
            return sp.log(value)
        return sp.simplify(sp.log(value) / sp.log(base))

single_letter_symbols = {chr(code): sp.Symbol(chr(code)) for code in range(ord("a"), ord("z") + 1)}
safe_globals = {
    "__builtins__": safe_builtins,
    "Abs": Abs,
    "E": sp.E,
    "I": sp.I,
    "Matrix": Matrix,
    "Rational": Rational,
    "acos": sp.acos,
    "asin": sp.asin,
    "atan": sp.atan,
    "ceil": sp.ceiling,
    "ceiling": sp.ceiling,
    "cmath": cmath,
    "comb": math.comb,
    "cos": sp.cos,
    "datetime": datetime,
    "math": math,
    "floor": sp.floor,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "ln": sandbox_log,
    "log": sandbox_log,
    "oo": sp.oo,
    "perm": math.perm,
    "pi": sp.pi,
    "sp": sp,
    "sympy": sp,
    "Eq": Eq,
    "expand": expand,
    "factor": factor,
    "simplify": simplify,
    "solve": solve,
    "sqrt": sqrt,
    "symbols": symbols,
}
safe_globals.update(single_letter_symbols)
stdout = io.StringIO()
try:
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, "<calculator_sandbox>", "exec"), safe_globals, safe_globals)
except Exception as exc:
    print(json.dumps({"ok": False, "error_type": type(exc).__name__, "message": str(exc)}))
else:
    output = stdout.getvalue().strip()
    if len(output) > output_max_chars:
        output = output[: output_max_chars // 2] + "..." + output[-output_max_chars // 2 :]
    print(json.dumps({"ok": True, "output": output}))
'''


def evaluate_expression_python_sandbox(
    expression: str,
    *,
    timeout_sec: float = _PYTHON_SANDBOX_DEFAULT_TIMEOUT_SEC,
    memory_mb: int = _PYTHON_SANDBOX_DEFAULT_MEMORY_MB,
    output_max_chars: int = _PYTHON_SANDBOX_OUTPUT_MAX_CHARS,
) -> str:
    code = _python_sandbox_prepare_code(expression)
    payload = {
        "code": code,
        "memory_mb": memory_mb,
        "cpu_sec": max(1, int(math.ceil(timeout_sec))),
        "output_max_chars": output_max_chars,
    }
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", _python_sandbox_child_source()],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Calculator sandbox exceeded {timeout_sec:.1f}s timeout.") from exc
    if result.returncode != 0 and not result.stdout.strip():
        message = (result.stderr or "").strip()
        raise RuntimeError(message or f"Calculator sandbox failed with exit code {result.returncode}.")
    raw = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    try:
        payload_out = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError((result.stderr or result.stdout or "").strip() or "Calculator sandbox returned invalid output.") from exc
    if not payload_out.get("ok"):
        error_type = str(payload_out.get("error_type") or "RuntimeError")
        message = str(payload_out.get("message") or "Calculator sandbox execution failed.")
        raise RuntimeError(f"{error_type}: {message}")
    output = str(payload_out.get("output") or "").strip()
    if not output:
        raise ValueError("Calculator sandbox produced no output.")
    return output


class CalculatorTool:
    name = "CALCULATE"

    def __init__(
        self,
        *,
        backend: str = "safe_eval",
        timeout_sec: float = _PYTHON_SANDBOX_DEFAULT_TIMEOUT_SEC,
        memory_mb: int = _PYTHON_SANDBOX_DEFAULT_MEMORY_MB,
        output_max_chars: int = _PYTHON_SANDBOX_OUTPUT_MAX_CHARS,
    ):
        self.backend = backend
        self.timeout_sec = timeout_sec
        self.memory_mb = memory_mb
        self.output_max_chars = output_max_chars

    def run(self, action_input: dict[str, Any], sample, history: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        expression = _extract_calculator_expression(action_input)
        if not expression:
            raise ValueError("CalculatorTool requires an expression/code payload.")
        if self.backend == "python_sandbox":
            result = evaluate_expression_python_sandbox(
                expression,
                timeout_sec=self.timeout_sec,
                memory_mb=self.memory_mb,
                output_max_chars=self.output_max_chars,
            )
        elif self.backend == "safe_eval":
            result = _format_result(evaluate_expression(expression))
        else:
            raise ValueError(f"Unsupported calculator backend: {self.backend}")
        observation = {"expression": expression, "result": _format_result(result), "backend": self.backend}
        return observation, False, {"helpful": True, "calculator_backend": self.backend}
