"""把论文 markdown 里裸写的 LaTeX 公式包裹成 $...$（display 公式用 $$...$$）。

背景：md 里的公式是裸写 LaTeX（\\phi^\\top w、\\delta_t、\\sum_t ...），pandoc 直接转
LaTeX 会把 ^ 转义成 \\^{}、_ 转义成 \\_（文本重音/下划线），公式渲染错。必须先包
数学模式。本脚本用「锚点 + 受控扩展」识别公式片段：

  锚点  = \\命令（含可选 {arg}）、^上标、_下标
  填充  = 单字母变量、数字、数学符号（= + - * / ( ) [ ] , . | ' 等）
  边界  = 多字母英文单词、词+结尾标点、数字+句号

用法：
  python tools/wrap_math.py 输入.md [输出.md] [--dry-run]
"""
import argparse
import re
import sys

# 数学锚点：\命令（含可选 {} 参数）、^ 上标、_ 下标、\, 间距命令
ANCHOR = re.compile(
    r'\\[a-zA-Z]+\{[^{}]*\}'      # \operatorname{...}, \mathbb{...}, \mathcal{...}, \text{...}
    r'|\\[a-zA-Z]+'               # \phi, \sum, \max, \qquad, \leftarrow ...
    r'|\\[,;!:]'                  # \, \; 间距
    r'|\^\{[^{}]*\}'              # ^{...}
    r'|\^\\?[A-Za-z0-9]'          # ^\top, ^d, ^2
    r'|_\{[^{}]*\}'               # _{...}
    r'|_\\?[A-Za-z0-9]'           # _t, _j
    r"|\b(?:alpha|beta|gamma|delta|epsilon|eta|xi|rho|phi|psi|pi|mu|infty|limsup)\b"
)

# 可作为公式一部分的字符（不含 |，它是 markdown 表格分隔符）
FILL = re.compile(r'^[A-Za-z0-9+\-*/=()\[\],.\'·×⊗]+$')
# 英文单词（含连字符复合词，如 multiplied-back、score-scaled）→ 边界
# 注意：每段须 ≥2 字母，避免把单字母数学变量 w/d/n/Z/A 误判为英文词
WORD = re.compile(r"^[A-Za-z]{2,}(?:-[A-Za-z]{2,})*$")
# 显式数学 token：2 个字母但属公式（dw = 参数更新），避免被当成英文词
MATH_WORDS = {"dw", "ds", "dv"}


def is_fill(tok: str) -> bool:
    if not tok:
        return False
    if tok in MATH_WORDS:
        return True
    if WORD.match(tok):
        return False
    # 含连续 2+ 字母的 token（(Section、(iii、max, 等）视为英文，非公式
    if re.search(r'[A-Za-z]{2,}', tok):
        return False
    return bool(FILL.match(tok))


def _wrap_plain_line(line: str) -> str:
    # 用占位符保护 ^{...} / _{...} 内部空格：tokenize 时不拆散，包裹后再还原。
    # 直接删空格会让 \mathbb{R}^{D \times D} 变成 D\timesD（\timesD 成未定义命令）
    line = re.sub(r'(\^|_)\{([^{}]*)\}',
                  lambda m: m.group(1) + '{' + m.group(2).replace(' ', '\x00') + '}',
                  line)
    parts = re.split(r'(\s+)', line)   # [tok, ws, tok, ws, ...]
    n = len(parts)
    marked = set()
    for i in range(0, n, 2):
        if parts[i] and ANCHOR.search(parts[i]):
            marked.add(i)
    if not marked:
        return line
    # 向左右扩展纳入相邻填充 token（跳过空格，空格不阻断）
    for i in range(0, n, 2):
        if i not in marked:
            continue
        j = i - 2
        while j >= 0:
            if parts[j] and is_fill(parts[j]):
                marked.add(j)
                j -= 2
            else:
                break
        j = i + 2
        while j < n:
            if parts[j] and is_fill(parts[j]):
                marked.add(j)
                j += 2
            else:
                break
    # 重建：连续 marked token（含其间空格）包 $...$
    out = []
    i = 0
    while i < n:
        if i in marked:
            out.append('$')
            while i < n and (i in marked or (parts[i].isspace() and i + 1 in marked)):
                out.append(parts[i])
                i += 1
            out.append('$')
        else:
            out.append(parts[i])
            i += 1
    result = ''.join(out)
    result = result.replace('\x00', ' ')   # 还原组内空格
    # display 检测：整行就是一个公式 → $$...$$
    s = result.strip()
    if s.startswith('$') and s.endswith('$') and s.count('$') == 2:
        inner = s[1:-1]
        result = '$$' + inner + '$$'
    return result


def wrap_line(line: str) -> str:
    """Wrap bare math while preserving any math spans already marked by the author."""
    if "$" not in line:
        return _wrap_plain_line(line)
    pieces = re.split(r"(\$\$[^$]*\$\$|\$[^$]*\$)", line)
    out = []
    for piece in pieces:
        if piece.startswith("$") and piece.endswith("$"):
            out.append(piece)
        else:
            out.append(_wrap_plain_line(piece))
    return "".join(out)


def wrap(md_text: str) -> str:
    out_lines = []
    in_code = False
    in_display_math = False
    for line in md_text.split('\n'):
        if line.strip().startswith('```'):
            in_code = not in_code
            out_lines.append(line)
            continue
        if in_code:
            out_lines.append(line)
            continue
        # Display math commonly spans several Markdown lines.  The earlier
        # line-local wrapper protected only the opening/closing ``$$`` line,
        # then re-wrapped every line inside ``aligned``/``gathered``/``cases``
        # with a fresh pair of dollars.  Track the fence across lines so that
        # author-supplied display environments remain byte-for-byte intact.
        display_delimiters = line.count("$$")
        if in_display_math or display_delimiters:
            out_lines.append(line)
            if display_delimiters % 2 == 1:
                in_display_math = not in_display_math
            continue
        out_lines.append(wrap_line(line))
    return '\n'.join(out_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('output', nargs='?')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    with open(args.input, encoding='utf-8') as f:
        src = f.read()
    dst = wrap(src)

    if not args.dry_run:
        out_path = args.output or args.input
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(dst)
        print(f"已写入：{out_path}")

    # 统计被包裹的公式（打印容错：GBK 控制台遇 Unicode 符号不崩）
    wrapped = re.findall(r'\$[^$]+\$', dst)
    try:
        print(f"包裹公式数：{len(wrapped)}")
        print("---- 公式清单 ----")
        for w in wrapped:
            print(w.encode('gbk', errors='replace').decode('gbk'))
        print("------------------")
    except Exception:
        pass


if __name__ == '__main__':
    main()
