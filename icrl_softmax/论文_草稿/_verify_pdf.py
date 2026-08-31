import fitz, re, sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

doc = fitz.open(sys.argv[1])
txt = ''.join(p.get_text() for p in doc)
print('pages:', doc.page_count)

# 1) 任何残留反斜杠宏（\ + 字母）都应没有
leftover = re.findall(r'\\[a-zA-Z]+', txt)
print('leftover LaTeX macros:', sorted(set(leftover)) if leftover else 'NONE')

# 2) 关键数字
nums = ['0.42', '0.4315', '0.4434', '0.5323', '0.4266', '0.4206', '0.428', '0.461', '0.376',
        '0.53', '0.38', '5.31', '5.26', '5.08', '3.14', '5.49', '5.42', '0.34',
        '0.92', '5.0', '0.29', '0.43', '0.45', '0.44',
        '2.50', '4.89', '5.22', '3.15', 'R^2']
missing = [n for n in nums if n not in txt]
print('missing nums:', missing if missing else 'NONE')

# 3) 关键符号
for s in ['δ', 'φ', 'τ', 'Σ', '→', '≈', '∈', '≥', '∞', 'ψ', 'ρ', '⊗', '|']:
    if s not in txt:
        print('MISSING SYMBOL:', s)
print('symbols OK')

# 4) 结构
for h in ['In-Context Policy Improvement', 'Keywords:', 'Abstract', '1. Introduction',
          '2. Related Work', '3. Preliminaries', '4. Method', '5. Experiments',
          '6. Discussion', '7. Conclusion', 'References', 'Appendix A']:
    if h not in txt:
        print('MISSING HEADER:', h)
print('headers OK')
