"""用 pandoc + xelatex 把论文 markdown 编译成排版良好的 PDF。

流程：
  full_paper.md（裸 LaTeX 公式）
    → wrap_math.wrap() 包裹 $...$ / $$...$$
    → POST_FIXES 纠正冠词/连字符误包
    → pandoc + xelatex → full_paper.pdf

用法：
  python tools/build_pdf.py
  python tools/build_pdf.py --src 论文_草稿/full_paper_中文版.md \
      --out tmp/pdfs/full_paper_zh_candidate.pdf --lang zh
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wrap_math

BASE = r"C:\Users\Admin\Desktop\research\icrl_softmax"
DEFAULT_SRC = os.path.join(BASE, "论文_草稿", "full_paper.md")
DEFAULT_OUT_PDF = os.path.join(BASE, "论文_草稿", "full_paper.pdf")
# 可用 ICRL_PDF_OUT 指向另一份输出，以免覆盖正在被阅读器占用的 PDF。
OUT_TMP = os.path.join(BASE, "tools", "_build_tmp.md")

XELATEX = r"C:\Users\Admin\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe"
PANDOC = r"C:\Users\Admin\anaconda3\Library\bin\pandoc.exe"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CSS = os.path.join(BASE, "tools", "paper.css")

# 论文专属后处理：纠正 wrap_math 对冠词 / 连字符复合词的误包。
# 每个条目是 (误包后的串, 正确串)。
POST_FIXES = [
    ("$(1/n)\\sum_t \\delta_t \\phi_t. A$", "$(1/n)\\sum_t \\delta_t \\phi_t.$ A"),
    ("$\\mathcal{S}, a$", "$\\mathcal{S}$ a"),
    ("$\\mathcal{A}, a$", "$\\mathcal{A}$ a"),
    ("$P(s' \\mid s, a), a$", "$P(s' \\mid s, a)$ a"),
    ("$\\gamma \\in [0, 1). A$", "$\\gamma \\in [0, 1)$ A"),
    ("$a \\tau-scaled$", "a $\\tau$-scaled"),
    ("$large-\\tau$", "large-$\\tau$"),
]


MATH_WORD_FIXES = [
    (r"(?<!\\)\balpha\b", r"\\alpha"),
    (r"(?<!\\)\bbeta\b", r"\\beta"),
    (r"(?<!\\)\bgamma\b", r"\\gamma"),
    (r"(?<!\\)\bdelta\b", r"\\delta"),
    (r"(?<!\\)\bepsilon\b", r"\\varepsilon"),
    (r"(?<!\\)\bxi\b", r"\\xi"),
    (r"(?<!\\)\brho\b", r"\\rho"),
    (r"(?<!\\)(?<![A-Za-z])eta(?![A-Za-z])", r"\\eta"),
    (r"(?<!\\)\bphi\b", r"\\phi"),
    (r"(?<!\\)\bPhi\b", r"\\Phi"),
    (r"(?<!\\)\bpsi\b", r"\\psi"),
    (r"(?<!\\)\bpi\b", r"\\pi"),
    (r"(?<!\\)\bmu\b", r"\\mu"),
    (r"(?<!\\)\binfty\b", r"\\infty"),
    (r"(?<!\\)\blimsup\b", r"\\limsup"),
    (r"(?<!\\)\bsum\b", r"\\sum"),
    (r"(?<!\\)\bmax\b", r"\\max"),
    (r"(?<!\\)\bexp\b", r"\\exp"),
    (r"(?<!\\)\blog\b", r"\\log"),
    (r"(?<!\\)\bsoftmax\b", r"\\operatorname{softmax}"),
]


def normalise_math_notation(text: str) -> str:
    """只在 $...$ / $$...$$ 内把 ASCII 数学名规范成 LaTeX 命令。"""
    pieces = re.split(r"(\$\$|\$)", text)
    in_math = False
    delimiter = None
    for index, piece in enumerate(pieces):
        if piece in {"$", "$$"}:
            if not in_math:
                in_math, delimiter = True, piece
            elif piece == delimiter:
                in_math, delimiter = False, None
            continue
        if not in_math:
            continue
        piece = re.sub(r"\^\{?pi'\}?", r"^{\\pi'}", piece)
        piece = re.sub(r"\^\{?star\}?", r"^{\\star}", piece)
        greek_subscripts = {
            "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma",
            "delta": r"\delta", "epsilon": r"\varepsilon", "eta": r"\eta",
            "xi": r"\xi", "rho": r"\rho", "phi": r"\phi", "psi": r"\psi",
            "pi": r"\pi", "mu": r"\mu", "infty": r"\infty",
        }
        for bare_name, latex_name in greek_subscripts.items():
            if bare_name == "infty":
                continue
            piece = re.sub(
                rf"(?<!\\)\b{bare_name}(?=_)",
                lambda _match, replacement=latex_name: replacement,
                piece,
            )
        piece = re.sub(
            r"_\{?(alpha|beta|gamma|delta|epsilon|eta|rho|phi|psi|pi|infty)\}?",
            lambda match: "_{" + greek_subscripts[match.group(1)] + "}",
            piece,
        )
        piece = re.sub(
            r"\^\{?(alpha|beta|gamma|delta|epsilon|eta|rho|phi|psi|pi|infty)\}?",
            lambda match: "^{" + greek_subscripts[match.group(1)] + "}",
            piece,
        )
        for pattern, replacement in MATH_WORD_FIXES:
            piece = re.sub(pattern, replacement, piece)
        piece = re.sub(r"\\widetilde\s*\\pi", r"\\widetilde{\\pi}", piece)
        piece = piece.replace(
            r"\operatorname{\operatorname{softmax}}",
            r"\operatorname{softmax}",
        )
        piece = re.sub(r"\b([TQV])\*", r"\1^{\\star}", piece)
        piece = re.sub(r"\^T\b", r"^{\\top}", piece)
        piece = piece.replace("<=", r"\le ").replace(">=", r"\ge ").replace("->", r"\to ")
        piece = re.sub(r"\|\|([^|]+)\|\|", r"\\lVert \1 \\rVert", piece)
        pieces[index] = piece
    return "".join(pieces)


def yaml_quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def extract_meta(text: str):
    """抽首个 # 标题和其后作者行作为元数据，并从正文移除。"""
    lines = text.split("\n")
    title = ""
    i = 0
    while i < len(lines) and not lines[i].lstrip().startswith("# "):
        i += 1
    if i < len(lines):
        title = lines[i].lstrip()[2:].strip()
        lines[i] = ""
    j = i + 1
    author = ""
    while j < len(lines) and not lines[j].strip():
        j += 1
    if j < len(lines) and not lines[j].lstrip().startswith("**Keywords"):
        author = lines[j].strip()
        lines[j] = ""
    body = "\n".join(lines)
    # markdown 转义括号 → 字面括号（元数据里 pandoc 不再做 markdown 转义）
    author = author.replace("\\[", "[").replace("\\]", "]")
    return title, author, body


def build_with_html_mathml(out_pdf: str, document_lang: str) -> bool:
    """Fallback renderer: Pandoc MathML + a headless Chromium print pass."""
    html_path = os.path.join(BASE, "tools", "_build_tmp.html")
    html_cmd = [
        PANDOC,
        OUT_TMP,
        "-o",
        html_path,
        "--standalone",
        "--to=html5",
        "--mathml",
        "--metadata",
        f"lang:{document_lang}",
        "--embed-resources",
        "--resource-path",
        BASE,
        "--css",
        CSS,
    ]
    html_run = subprocess.run(
        html_cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=BASE,
    )
    if html_run.stderr.strip():
        sys.stderr.write(html_run.stderr)
    if html_run.returncode != 0:
        return False

    browser = EDGE if os.path.exists(EDGE) else CHROME
    if not os.path.exists(browser):
        return False
    os.makedirs(os.path.join(BASE, "tmp"), exist_ok=True)
    profile = tempfile.mkdtemp(prefix="paper-pdf-", dir=os.path.join(BASE, "tmp"))
    before = (
        (os.path.getmtime(out_pdf), os.path.getsize(out_pdf))
        if os.path.exists(out_pdf)
        else None
    )
    browser_cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--disable-gpu-compositing",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--allow-file-access-from-files",
        "--run-all-compositor-stages-before-draw",
        f"--user-data-dir={profile}",
        f"--print-to-pdf={out_pdf}",
        Path(html_path).as_uri(),
    ]
    browser_run = subprocess.run(
        browser_cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=BASE,
    )
    after = (
        (os.path.getmtime(out_pdf), os.path.getsize(out_pdf))
        if os.path.exists(out_pdf)
        else None
    )
    return (
        browser_run.returncode == 0
        and after is not None
        and after[1] > 0
        and after != before
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the English or Chinese manuscript PDF with Pandoc and XeLaTeX."
    )
    parser.add_argument(
        "--src",
        default=DEFAULT_SRC,
        help="Markdown source path (default: 论文_草稿/full_paper.md).",
    )
    parser.add_argument(
        "--out",
        default=os.environ.get("ICRL_PDF_OUT", DEFAULT_OUT_PDF),
        help=(
            "PDF output path (default: 论文_草稿/full_paper.pdf; "
            "ICRL_PDF_OUT remains supported)."
        ),
    )
    parser.add_argument(
        "--lang",
        choices=("en", "zh"),
        default="en",
        help="Document language and font profile (default: en).",
    )
    return parser.parse_args()


def resolve_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(BASE) / path
    return str(path.resolve())


def main():
    args = parse_args()
    src = resolve_path(args.src)
    out_pdf = resolve_path(args.out)
    document_lang = "zh-CN" if args.lang == "zh" else "en-US"

    with open(src, encoding="utf-8") as f:
        text = f.read()

    title, author, body = extract_meta(text)
    body = wrap_math.wrap(body)
    body = normalise_math_notation(body)
    # 裸 Unicode 数学符号 → 数学模式（正文 "δ field"/"max-δ"、参考标题 "Softmax ≥ Linear"）
    body = body.replace('δ', '$\\delta$').replace('≥', '$\\ge$')
    for a, b in POST_FIXES:
        body = body.replace(a, b)
    # 规范标题层级：### 小节 → ## 小节（# 一级标题保留；避免 LaTeX 的 run-in subsubsection）
    body = re.sub(r"^### ", "## ", body, flags=re.M)

    md = (
        "---\n"
        f"title: {yaml_quote(title)}\n"
        f"author: {yaml_quote(author)}\n"
        f"lang: {yaml_quote(document_lang)}\n"
        "---\n\n" + body
    )
    with open(OUT_TMP, "w", encoding="utf-8") as f:
        f.write(md)

    cmd = [
        PANDOC, OUT_TMP, "-o", out_pdf,
        f"--pdf-engine={XELATEX}",
        "--standalone",
        "--resource-path", BASE,
        "-V", "geometry:margin=1in",
        "-V", "fontsize=11pt",
        "-V", "linestretch=1.1",
    ]
    if args.lang == "zh":
        cmd.extend([
            "-V", "classoption=chinese",
            "-V", "mainfont=Times New Roman",
            "-V", "mathfont=Cambria Math",
            "-V", "CJKmainfont=Microsoft YaHei",
        ])
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    print("pandoc:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=BASE)
    sys.stdout.write(r.stdout)
    if r.stderr.strip():
        sys.stderr.write(r.stderr)
    if r.returncode != 0:
        print("\nXeLaTeX 不可用，切换到 MathML/Chromium PDF 渲染。")
        if not build_with_html_mathml(out_pdf, document_lang):
            sys.exit(f"两种 PDF 渲染路径均失败 (XeLaTeX exit {r.returncode})")
    print(f"\n已生成：{out_pdf}")


if __name__ == "__main__":
    main()
