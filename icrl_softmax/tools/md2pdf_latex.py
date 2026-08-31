"""中文 + LaTeX 数学 markdown → PDF（一键，pandoc + xelatex 真渲染公式）。

与 md2pdf.py（fpdf2，只支持简单希腊字母）不同，本脚本用 xelatex 真渲染
矩阵 \\begin{pmatrix}、分数 \\frac、\\tag 编号、\\binom、\\lVert 等复杂结构。

用法：
  python tools/md2pdf_latex.py 某篇.md [另一篇.md ...]   # 输出到同目录同名 .pdf
  python tools/md2pdf_latex.py 某篇.md -o 输出.pdf        # 单文件指定输出
  python tools/md2pdf_latex.py 某篇.md --fix               # 自动把裸 Unicode 数学符号包进 $...$

裸符号说明：markdown 里写在 $...$ 之外的 ≈、×、≤、∑ 等 Unicode 数学符号，
正文拉丁字体缺字形，会在 PDF 里显示成空白块。脚本默认只警告；加 --fix 自动替换成
对应 LaTeX 命令（≈ → $\\approx$）。
"""
import argparse
import os
import re
import subprocess
import sys
import unicodedata

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PANDOC = r"C:\Users\Admin\anaconda3\Library\bin\pandoc.exe"
XELATEX = r"C:\Users\Admin\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe"
CJK = "Microsoft YaHei"

# 裸 Unicode 数学符号 → LaTeX 命令（--fix 用；顺序：多字符在前避免误替）
UNI2TEX = {
    "≈": r"\approx", "≤": r"\le", "≥": r"\ge", "≠": r"\ne", "∈": r"\in",
    "⊗": r"\otimes", "∞": r"\infty", "→": r"\to", "←": r"\leftarrow",
    "∇": r"\nabla", "∑": r"\sum", "∏": r"\prod", "×": r"\times",
    "·": r"\cdot", "±": r"\pm", "−": "-",
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\varepsilon", "θ": r"\theta", "λ": r"\lambda", "μ": r"\mu",
    "π": r"\pi", "σ": r"\sigma", "τ": r"\tau", "φ": r"\phi",
    "ψ": r"\psi", "ρ": r"\rho", "ω": r"\omega", "Δ": r"\Delta",
}


def scan_bare_math(text):
    """找出数学环境（$...$ / $$...$$）之外的裸 Unicode 数学符号。
    返回 [(行号, 字符, U+xxxx, Unicode 名)]。"""
    out = []
    for i, line in enumerate(text.split("\n"), 1):
        inmath = False
        j, n = 0, len(line)
        while j < n:
            c = line[j]
            if c == "$":
                if j + 1 < n and line[j + 1] == "$":
                    j += 2
                    inmath = not inmath
                    continue
                j += 1
                inmath = not inmath
                continue
            if not inmath:
                cp = ord(c)
                if (0x2200 <= cp <= 0x22FF) or (0x2190 <= cp <= 0x21FF) \
                        or (0x1D400 <= cp <= 0x1D7FF) or (0x03B1 <= cp <= 0x03C9) \
                        or c in "×±·−":
                    out.append((i, c, "U+%04X" % cp, unicodedata.name(c, "?")))
            j += 1
    return out


def fix_bare(text):
    """把数学环境外的裸符号替换成对应 LaTeX 命令包在 $...$ 里（不碰 $...$ 内部）。"""
    out = []
    inmath = False
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "$":
            if i + 1 < n and text[i + 1] == "$":
                out.append("$$")
                i += 2
                inmath = not inmath
                continue
            out.append("$")
            i += 1
            inmath = not inmath
            continue
        if not inmath and c in UNI2TEX:
            out.append("$" + UNI2TEX[c] + "$")
        else:
            out.append(c)
        i += 1
    return "".join(out)


def compile_one(md_path, pdf_path):
    md_path = os.path.abspath(md_path)
    pdf_path = os.path.abspath(pdf_path)
    cwd = os.path.dirname(md_path)
    cmd = [
        PANDOC, md_path, "-o", pdf_path,
        f"--pdf-engine={XELATEX}",
        "--standalone",
        "-V", f"CJKmainfont={CJK}",
        "-V", "geometry:margin=1in",
        "-V", "fontsize=11pt",
        "-V", "linestretch=1.1",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=cwd)
    err = r.stderr or ""
    problems = [ln for ln in err.splitlines()
                if re.search(r"(?i)error|undefined control|missing character|! ", ln)]
    if r.returncode != 0:
        print(f"[FAIL] {md_path}")
        if re.search(r"(?i)permission denied", err):
            print("    目标 PDF 被占用（可能正在 PDF 阅读器中打开），请关闭后重试。")
        for p in problems:
            print("   ", p)
        return False
    for p in problems:
        print(f"[warn] {p}")
    print(f"[OK] {pdf_path}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", help="单输入时的输出路径")
    ap.add_argument("--fix", action="store_true",
                    help="自动把裸 Unicode 数学符号替换成 LaTeX 命令")
    args = ap.parse_args()

    ok = True
    for i, src in enumerate(args.inputs):
        with open(src, encoding="utf-8") as f:
            text = f.read()
        bare = scan_bare_math(text)
        if bare:
            print(f"[bare-symbol] {src} 有 {len(bare)} 个数学环境外的裸符号：")
            for ln, ch, cp, name in bare:
                print(f"    行 {ln}: {ch}  ({cp}, {name})")
            if args.fix:
                text = fix_bare(text)
                with open(src, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"    --fix：已自动替换")
        dst = args.output if (args.output and len(args.inputs) == 1) \
            else src.rsplit(".", 1)[0] + ".pdf"
        if not compile_one(src, dst):
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
