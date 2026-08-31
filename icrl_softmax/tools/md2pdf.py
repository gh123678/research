"""Markdown → PDF 转换（中文友好，手机可读）。

用法：python tools/md2pdf.py 输入.md [输出.pdf]
依赖：fpdf2（已装 2.8.7）+ Windows 微软雅黑字体。
流程：markdown → HTML（tables/fenced_code 扩展）→ fpdf2 write_html 渲染。
"""
import re
import sys
import markdown
from fpdf import FPDF
from fpdf.fonts import FontFace

FONT = "C:/Windows/Fonts/msyh.ttc"      # 微软雅黑（正文/中文）
FONT_BOLD = "C:/Windows/Fonts/simhei.ttf"  # 黑体（粗体变体，用于标题/表格头）
MONO = "C:/Windows/Fonts/consola.ttf"   # Consolas（代码/公式等宽）
SYM = "C:/Windows/Fonts/seguisym.ttf"   # Segoe UI Symbol（数学符号 fallback，如 ⊗）


_TEX = {
    "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\delta": "δ",
    "\\varepsilon": "ε", "\\theta": "θ", "\\lambda": "λ", "\\mu": "μ",
    "\\pi": "π", "\\sigma": "σ", "\\tau": "τ", "\\phi": "φ",
    "\\varphi": "φ", "\\Phi": "Φ", "\\psi": "ψ", "\\rho": "ρ",
    "\\omega": "ω", "\\Delta": "Δ", "\\nabla": "∇",
    "\\sum": "Σ", "\\prod": "∏",
    "\\cdot": "·", "\\times": "×", "\\otimes": "⊗", "\\pm": "±",
    "\\mid": "|",
    "\\to": "→", "\\rightarrow": "→", "\\leftarrow": "←",
    "\\approx": "≈", "\\in": "∈", "\\infty": "∞",
    "\\le": "≤", "\\leq": "≤", "\\ge": "≥", "\\geq": "≥",
    "\\ne": "≠", "\\neq": "≠",
    "\\max": "max", "\\min": "min",
    "\\qquad": "  ", "\\quad": " ",
}


def _tex_to_unicode(md_text):
    """把 LaTeX 数学宏替换成 Unicode（只影响 PDF 显示，不改源文件）。
    顺序：先特殊宏，再查表宏，最后 \frac（此时参数已不含反斜杠）。"""
    t = md_text
    t = re.sub(r"\\mathbb\{R\}", "R", t)   # 雅黑缺 ℝ/𝔼/𝒮 字形，退而用正体
    t = re.sub(r"\\mathbb\{E\}", "E", t)
    t = re.sub(r"\\mathcal\{([A-Za-z])\}", r"\1", t)   # \mathcal{S} → S
    t = re.sub(r"\\operatorname\{([a-zA-Z]+)\}", r"\1", t)
    t = re.sub(r"\\text\{([^{}]*)\}", r"\1", t)
    # 按完整 token（\ + 字母序列）查表，避免 \in 误拆 \infty、\to 误拆 \top
    t = re.sub(r"\\([a-zA-Z]+)", lambda m: _TEX.get(m.group(0), m.group(0)), t)
    t = re.sub(r"\\frac\{([^{}\\]*)\}\{([^{}\\]*)\}", r"(\1)/(\2)", t)
    # \top 是转置符号，前面常有 ^ 上标：^\top → ^T；单独 \top → T
    t = t.replace("\\top", "T")
    return t


def _strip_table_markup(md_text):
    """fpdf2 write_html 不支持 <td> 内嵌套 <code>/<strong>，
    表格行统一剥掉反引号与加粗标记（表头仍由 <th> 加粗）。"""
    out = []
    for line in md_text.splitlines():
        if line.strip().startswith("|"):
            line = line.replace("`", "").replace("**", "")
            # 剥反引号后代码里的乘法 * 会裸露，被 markdown 误判成斜体 <em>
            # （fpdf2 不支持 <td> 内嵌 <em>），统一转成 ×
            line = line.replace("*", "×")
        out.append(line)
    return "\n".join(out)


def md2pdf(md_path, pdf_path):
    with open(md_path, encoding="utf-8") as f:
        md_text = f.read()
    html = markdown.markdown(_tex_to_unicode(_strip_table_markup(md_text)),
                             extensions=["tables", "fenced_code"])

    pdf = FPDF(format="A4")
    pdf.add_font("YaHei", "", FONT)
    pdf.add_font("YaHei", "B", FONT_BOLD)   # 粗体变体：黑体
    # 中文无斜体字形，斜体/粗斜体映射到同一字体（避免 write_html 遇到 <em> 时
    # 请求未注册的 yaheiI 而抛 "Undefined font"）
    pdf.add_font("YaHei", "I", FONT)
    pdf.add_font("YaHei", "BI", FONT_BOLD)
    pdf.add_font("Mono", "", MONO)
    pdf.add_font("Mono", "B", MONO)
    pdf.add_font("Mono", "I", MONO)
    pdf.add_font("Mono", "BI", MONO)
    pdf.add_font("Sym", "", SYM)
    pdf.set_fallback_fonts(["YaHei", "Sym"])  # 先退雅黑；数学符号（⊗ 等）退 Segoe UI Symbol
    pdf.add_page()
    pdf.set_margin(16)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_font("YaHei", "", 10.5)
    pdf.write_html(html, tag_styles={
        "code": FontFace(family="Mono", size_pt=9.5),
        "pre": FontFace(family="Mono", size_pt=9.5),
    })
    pdf.output(pdf_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    md = sys.argv[1]
    pdf = sys.argv[2] if len(sys.argv) > 2 else md.rsplit(".", 1)[0] + ".pdf"
    md2pdf(md, pdf)
    print(f"OK: {pdf}")
