"""把正式分节 md 拼成一篇完整论文 full_paper.md（统一章节号、修正跨节引用）。

顺序：Abstract → 1 Introduction → 2 Related Work → 3 Foundations and Guarantees
      → 4 Method → 5 Experiments → 6 Discussion → 7 Conclusion
      → References
"""
import re

D = r"C:\Users\Admin\Desktop\research\icrl_softmax\论文_草稿"


def read(n):
    return open(f"{D}\\{n}", encoding="utf-8").read()


def strip_title(t):
    ls = t.splitlines()
    while ls and not ls[0].strip():
        ls.pop(0)
    if ls and ls[0].lstrip().startswith("#"):
        ls = ls[1:]
    while ls and not ls[0].strip():
        ls.pop(0)
    return "\n".join(ls).strip() + "\n"


def split_by_h2(t):
    blocks, titles, cur, buf = {}, {}, None, []
    for line in t.splitlines():
        m = re.match(r"^## (\d+)\.\s+(.+)$", line)
        if m:
            if cur is not None:
                blocks[cur] = "\n".join(buf)
            cur = int(m.group(1))
            titles[cur] = m.group(2).strip()
            buf = []
        else:
            buf.append(line)
    if cur is not None:
        blocks[cur] = "\n".join(buf)
    return blocks, titles


def renum(block, offset):
    return re.sub(r"^### (\d+)\.(\d+)",
                  lambda m: f"### {int(m.group(1)) + offset}.{m.group(2)}",
                  block, flags=re.M)


blocks, titles = split_by_h2(read("method_experiments.md"))
method = renum(blocks[3], 1)  # 3.x → 4.x
exp = renum(blocks[4], 1)     # 4.x → 5.x
disc = renum(blocks[5], 1)    # 5.x → 6.x

# 跨节引用修正（Preliminaries 在第 3 节、Experiments 在第 5 节）
method = method.replace("Section 2", "Section 3")
exp = re.sub(r"Section 4\.(\d)", r"Section 5.\1", exp)
disc = disc.replace("Section 2", "Section 3")
disc = re.sub(r"Section 4\.(\d)", r"Section 5.\1", disc)

out = []
intro = read("introduction.md").split("## References")[0]  # introduction.md 内嵌引用冗余，references.md 已单独有
out.append("# Standard Softmax Attention Supports In-Context Policy Improvement\n\n")
out.append("Anonymous Authors\n\n")
out.append("**Keywords:** in-context reinforcement learning; softmax attention; policy improvement; SARSA; Expected SARSA; approximate greedification\n\n")
out.append("# Abstract\n\n" + strip_title(read("abstract.md")))
out.append("# 1. Introduction\n\n" + strip_title(intro))
out.append("# 2. Related Work\n\n" + strip_title(read("related_work.md")))
out.append("# 3. End-to-End Construction and Guarantees\n\n" + strip_title(read("preliminaries.md")))
out.append("# 4. Method\n\n" + method.strip())
out.append("# 5. Experiments\n\n" + exp.strip())
out.append("# 6. Discussion\n\n" + disc.strip())
out.append("# 7. Conclusion\n\n" + strip_title(read("conclusion.md")))
out.append("# References\n\n" + strip_title(read("references.md")))

with open(f"{D}\\full_paper.md", "w", encoding="utf-8") as f:
    f.write("\n\n".join(out) + "\n")
print("OK full_paper.md")
