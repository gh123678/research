"""机械规范化论文 Markdown 中的 ASCII 数学名；跳过 fenced code blocks。"""

import re
from pathlib import Path


BASE = Path(r"C:\Users\Admin\Desktop\research\icrl_softmax\论文_草稿")
FILES = [
    "abstract.md",
    "introduction.md",
    "related_work.md",
    "preliminaries.md",
    "method_experiments.md",
    "conclusion.md",
]

GREEK = {
    "Phi": r"\Phi",
    "alpha": r"\alpha",
    "beta": r"\beta",
    "gamma": r"\gamma",
    "delta": r"\delta",
    "epsilon": r"\varepsilon",
    "rho": r"\rho",
    "eta": r"\eta",
    "phi": r"\phi",
    "psi": r"\psi",
    "pi": r"\pi",
    "infty": r"\infty",
}


def normalise_line(line: str) -> str:
    for name, command in GREEK.items():
        line = re.sub(
            rf"(?<!\\)\b{name}\b",
            lambda _match, replacement=command: replacement,
            line,
        )
    line = re.sub(r"(?<!\\)\b(sum|max|limsup)(?=_)", r"\\\1", line)
    line = re.sub(r"(?<!\\)\b(log|exp)(?=\()", r"\\\1", line)
    line = re.sub(
        r"(?<!\\)\bsoftmax(?=\()", r"\\operatorname{softmax}", line
    )
    line = re.sub(r"\b([TQV])\^\\star", r"\1^{\\star}", line)
    line = re.sub(r"\b([TQV])\*", r"\1^{\\star}", line)
    line = re.sub(r"\^T\b", r"^\\top", line)
    line = line.replace("<=", r"\le ").replace(">=", r"\ge ").replace("->", r"\to ")
    return line


for filename in FILES:
    path = BASE / filename
    lines = path.read_text(encoding="utf-8").splitlines()
    output = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            output.append(line)
        elif in_code:
            output.append(line)
        else:
            output.append(normalise_line(line))
    path.write_text("\n".join(output) + "\n", encoding="utf-8")
    print(path)
