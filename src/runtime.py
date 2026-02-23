"""
RLM Runtime — Python-side helpers for the Recursive Language Model CLI.

This module runs in a persistent Python subprocess. It provides:
  - `context`: the full prompt/document as a string variable
  - `llm_query(sub_context, instruction)`: bridge to parent LLM for sub-queries
  - `FINAL(x)`: set final answer string and terminate loop
  - `FINAL_VAR(x)`: set final answer from a variable

Communication protocol (line-delimited JSON over stdio):
  -> stdout: {"type":"llm_query","sub_context":"...","instruction":"...","id":"..."}
  <- stdin:  {"type":"llm_result","id":"...","result":"..."}
  -> stdout: {"type":"exec_done","stdout":"...","stderr":"...","has_final":bool,"final_value":"..."|null}

All protocol I/O uses saved references to the original sys.stdout/sys.stdin
so that exec'd code can freely redirect sys.stdout for print() capture.
"""

import json
import sys
import uuid
import io
import traceback
import asyncio

# Real stdio handles — saved before exec() can redirect sys.stdout/sys.stderr.
_real_stdout = sys.stdout
_real_stdin = sys.stdin

# Will be set by the TypeScript host before each execution
context: str = ""

# Sentinel — when set to a non-None value, the loop terminates
__final_result__ = None


def FINAL(x):
    """Set the final answer as a string and terminate the RLM loop."""
    global __final_result__
    __final_result__ = str(x)


def FINAL_VAR(x):
    """Set the final answer from a variable and terminate the RLM loop."""
    global __final_result__
    __final_result__ = str(x) if x is not None else None


def llm_query(sub_context: str, instruction: str = "") -> str:
    """Send a sub-context and instruction to the parent LLM and return the response.

    Can be called synchronously from regular code, or used with await in async code.
    For parallel queries, use:
        results = await asyncio.gather(
            async_llm_query(ctx1, instr1),
            async_llm_query(ctx2, instr2),
        )
    """
    # If called with just one arg, treat the whole thing as context+instruction combined
    if not instruction:
        instruction = ""

    request_id = uuid.uuid4().hex[:12]
    request = {
        "type": "llm_query",
        "sub_context": sub_context,
        "instruction": instruction,
        "id": request_id,
    }
    _real_stdout.write(json.dumps(request) + "\n")
    _real_stdout.flush()

    # Block until the TypeScript host sends back the result
    while True:
        line = _real_stdin.readline()
        if not line:
            raise RuntimeError("REPL stdin closed unexpectedly")
        line = line.strip()
        if not line:
            continue
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            continue
        if response.get("type") == "llm_result" and response.get("id") == request_id:
            return response.get("result", "")


async def async_llm_query(sub_context: str, instruction: str = "") -> str:
    """Async wrapper around llm_query for use with asyncio.gather().

    Usage:
        import asyncio
        results = await asyncio.gather(
            async_llm_query(chunk1, "summarize"),
            async_llm_query(chunk2, "summarize"),
        )
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, llm_query, sub_context, instruction)


def _execute_code(code: str) -> None:
    """Execute a code snippet in the module's global scope, capturing output."""
    global __final_result__
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        sys.stdout = captured_stdout
        sys.stderr = captured_stderr
        # Support both sync and async code (await expressions)
        try:
            # Try to compile as regular code first
            compiled = compile(code, "<repl>", "exec")
            exec(compiled, globals())
        except SyntaxError as e:
            if "await" in str(code):
                # Code contains await — run it in an async context
                async_code = f"async def __async_exec__():\n"
                for line in code.split("\n"):
                    async_code += f"    {line}\n"
                async_code += "\nimport asyncio\nasyncio.get_event_loop().run_until_complete(__async_exec__())"
                exec(compile(async_code, "<repl>", "exec"), globals())
            else:
                raise e
    except Exception:
        traceback.print_exc(file=captured_stderr)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    stdout_val = captured_stdout.getvalue()
    stderr_val = captured_stderr.getvalue()

    result = {
        "type": "exec_done",
        "stdout": stdout_val,
        "stderr": stderr_val,
        "has_final": __final_result__ is not None,
        "final_value": str(__final_result__) if __final_result__ is not None else None,
    }
    _real_stdout.write(json.dumps(result) + "\n")
    _real_stdout.flush()


def _main_loop() -> None:
    """Read execution requests from stdin in a loop."""
    while True:
        line = _real_stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        if msg.get("type") == "exec":
            _execute_code(msg.get("code", ""))
        elif msg.get("type") == "set_context":
            global context
            context = msg.get("value", "")
            ack = {"type": "context_set"}
            _real_stdout.write(json.dumps(ack) + "\n")
            _real_stdout.flush()
        elif msg.get("type") == "reset_final":
            global __final_result__
            __final_result__ = None
            ack = {"type": "final_reset"}
            _real_stdout.write(json.dumps(ack) + "\n")
            _real_stdout.flush()
        elif msg.get("type") == "shutdown":
            break


if __name__ == "__main__":
    ready = {"type": "ready"}
    _real_stdout.write(json.dumps(ready) + "\n")
    _real_stdout.flush()
    _main_loop()
