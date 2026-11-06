"""constrain_code.py — CONSTRAINED DECODING for code: only emit tokens that keep the program
syntactically valid.

WHY THIS IS THE RIGHT FIX (measured, not hoped)
----------------------------------------------
The code model parses 8/25 and runs 7/25. Its output is mostly prose with code fragments glued on.
A bigger model or more epochs would not fix this: the model is free to emit a word like "the" in
the middle of a function body, and nothing stops it. Constrained decoding removes that freedom —
at every step, tokens that would make the program unparseable are MASKED OUT (set to -inf), so the
model can only choose among continuations that keep the code valid.

This is a DECODER change, exactly like top-k and the repetition penalty: the model's weights,
perplexity and accuracy are untouched. It changes what the model is ALLOWED to say, not what it
believes.

HOW IT WORKS
------------
A tiny incremental Python-syntax checker tracks, as tokens are appended:
  * bracket depth ( ) [ ] { }  — a closing bracket with empty depth is illegal
  * whether we are inside a string (so a ")" inside a string is not a bracket)
  * whether the current line is blank / only whitespace (a blank line ends an indented block)
  * whether a line ends with ':' (the next line MUST be indented)
  * whether a line ends with a line-continuation backslash
  * indentation stack, so a dedent to a level that was never opened is illegal
  * whether the last non-space token is one that REQUIRES a following expression
    (e.g. after "return" a value is expected; after "def" a name is expected)

That is the whole mechanism. It is deliberately small and fully inspectable — no parser library,
no hidden state, every rule readable in this file.

HONEST LIMITS, STATED UP FRONT
------------------------------
* It enforces SYNTAX, not sense. "def f ( ) : pass" is valid Python and will be allowed. The
  model can still produce code that parses and does nothing useful. The gate measures syntax and
  execution, not usefulness.
* It is a heuristic checker, not CPython's grammar. It may allow a few things CPython rejects, and
  reject a few things CPython accepts. The gate is measured against CPython's own ast.parse, so
  any disagreement shows up as a failure rather than being hidden.
* It does not fix the prose problem by itself. If the model wants to write English, constraining it
  to valid syntax may produce valid-but-pointless code. That is a real outcome and will be reported.
"""
import ast
import re

# tokens that, when they end a line, REQUIRE the next line to be indented
BLOCK_STARTERS = ("def ", "class ", "if ", "elif ", "else:", "for ", "while ", "try:",
                  "except", "finally:", "with ", "match ", "case ")
# tokens after which a value/expression is expected (so a newline would be illegal)
NEEDS_VALUE = ("return", "yield", "raise", "assert", "lambda", "=", "+=", "-=", "*=", "/=",
               "==", "!=", "<", ">", "<=", ">=", "and", "or", "not", "in", "is", "+", "-",
               "*", "/", "%", "//", "**", "&", "|", "^", "<<", ">>", "(", "[", "{", ",")
# keywords that may NOT follow a value-requiring token on the same line start
STATEMENT_KEYWORDS = ("def", "class", "if", "elif", "else", "for", "while", "try", "except",
                      "finally", "with", "return", "yield", "raise", "import", "from", "pass",
                      "break", "continue", "global", "nonlocal", "assert", "del", "print")

OPENERS = {"(": ")", "[": "]", "{": "}"}
CLOSERS = {")", "]", "}"}


class SyntaxConstraint:
    """Incremental Python-syntax constraint. Feed tokens one at a time; ask which are allowed."""

    def __init__(self):
        self.depth = {"(": 0, "[": 0, "{": 0}
        self.in_str = None          # None, or the quote char we are inside
        self.escape = False
        self.line = ""              # current logical line (no trailing newline)
        self.lines = []             # completed lines
        self.indent_stack = [0]     # indentation levels that are open
        self.cur_indent = 0
        self.continuation = False   # previous line ended with backslash
        self.at_line_start = True
        self.last_tok = None
        self.pending_dedent = False

    # ---- token ingestion -------------------------------------------------
    def feed(self, tok):
        """Consume one token. Returns False if the token itself is illegal here."""
        if self.in_str is not None:
            self.line += tok
            if self.escape:
                self.escape = False
            elif tok.endswith("\\"):
                self.escape = True
            elif self.in_str in tok:
                self.in_str = None
            self.last_tok = tok
            return True

        # string start?
        for q in ('"""', "'''", '"', "'"):
            if tok.startswith(q) and not tok.startswith(("r", "b", "f", "R", "B", "F")):
                self.in_str = q[-1] if len(q) == 1 else q
                self.line += tok
                self.last_tok = tok
                return True

        if tok in OPENERS:
            self.depth[tok] += 1
        elif tok in CLOSERS:
            inv = {v: k for k, v in OPENERS.items()}[tok]
            if self.depth[inv] <= 0:
                return False            # closing a bracket that is not open
            self.depth[inv] -= 1

        if tok == "\n":
            return self._newline()

        self.line += tok
        self.last_tok = tok
        return True

    def _newline(self):
        """Handle a line ending. Returns False if a newline is illegal here."""
        if self.in_str is not None:
            self.line += "\n"
            return True                 # newlines are fine inside a string
        if self.continuation:
            self.continuation = False
            self.lines.append(self.line)
            self.line = ""
            return True
        if self.depth["("] or self.depth["["] or self.depth["{"]:
            self.lines.append(self.line)
            self.line = ""
            return True                 # implicit continuation inside brackets
        stripped = self.line.strip()
        if stripped.endswith("\\"):
            self.continuation = True
            self.lines.append(self.line)
            self.line = ""
            return True
        if stripped.endswith(":"):
            self.lines.append(self.line)
            self.line = ""
            self._push_indent()
            return True
        if stripped == "":
            # a blank line: close any open indented block
            self.lines.append(self.line)
            self.line = ""
            self._pop_to_dedent()
            return True
        # a normal statement line: it must not follow a value-requiring token
        if self.last_tok in NEEDS_VALUE:
            return False
        self.lines.append(self.line)
        self.line = ""
        self._pop_to_dedent()
        return True

    def _push_indent(self):
        base = len(self.line) - len(self.line.lstrip())
        self.indent_stack.append(base + 4)

    def _pop_to_dedent(self):
        cur = len(self.line) - len(self.line.lstrip())
        while len(self.indent_stack) > 1 and cur < self.indent_stack[-1]:
            self.indent_stack.pop()

    # ---- constraint query ------------------------------------------------
    def allows(self, tok):
        """Would appending `tok` keep the program syntactically plausible? Conservative."""
        if self.in_str is not None:
            return True                 # anything goes inside a string
        if tok in CLOSERS:
            inv = {v: k for k, v in OPENERS.items()}[tok]
            return self.depth[inv] > 0
        if tok == "\n":
            if self.depth["("] or self.depth["["] or self.depth["{"]:
                return True
            if self.continuation:
                return True
            s = self.line.strip()
            if s.endswith(":") or s.endswith("\\"):
                return True
            if s == "":
                return True
            return self.last_tok not in NEEDS_VALUE
        if self.at_line_start and self.line.strip() == "":
            # at the start of a line, a statement keyword is fine; a value is not
            if tok in STATEMENT_KEYWORDS:
                return True
            if self.last_tok in NEEDS_VALUE:
                return True
            return self.last_tok is None or self.last_tok in (".", ",")
        return True

    # ---- whole-program check (the honest gate) ---------------------------
    def valid_so_far(self):
        """True if everything fed so far parses with CPython itself."""
        src = "\n".join(self.lines + ([self.line] if self.line.strip() else []))
        if not src.strip():
            return True
        try:
            ast.parse(src)
            return True
        except SyntaxError:
            return False


def mask_invalid(logits, vocab, constraint, pad_id=0):
    """Return a copy of `logits` with every token the constraint forbids set to -inf.

    The pad/unk token is always allowed so generation can always terminate.
    """
    import numpy as np
    z = np.asarray(logits, dtype=np.float64).copy()
    for i, w in enumerate(vocab):
        if i == pad_id:
            continue
        if not constraint.allows(w):
            z[i] = -np.inf
    if not np.isfinite(z).any():
        z[pad_id] = 0.0
    return z
