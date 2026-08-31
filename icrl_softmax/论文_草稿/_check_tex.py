import re, glob, sys
sys.path.insert(0, r"C:\Users\Admin\Desktop\research\icrl_softmax\tools")
from md2pdf import _TEX

# 特殊正则处理过的宏（md2pdf 里 re.sub 覆盖）
handled = {'mathbb', 'operatorname', 'text', 'frac', 'top'}

all_macros = set()
for f in glob.glob('*.md'):
    if f.startswith('_check'):
        continue
    t = open(f, encoding='utf-8').read()
    for m in re.finditer(r'\\([a-zA-Z]+)', t):
        all_macros.add(m.group(1))

# 真正未被 _TEX 或特殊正则覆盖的
missing = {m for m in all_macros
           if '\\' + m not in _TEX and m not in handled}
print('ALL macro count:', len(all_macros))
print('---TRULY MISSING:', sorted(missing))

# 也检查一下 \mathbb{...} 的花体内容是否都处理
for f in glob.glob('*.md'):
    if f.startswith('_check'):
        continue
    t = open(f, encoding='utf-8').read()
    for m in re.finditer(r'\\mathbb\{([^{}]*)\}', t):
        c = m.group(1)
        if c not in ('R', 'E'):
            print(f'{f}: mathbb{{{c}}} NOT handled')
    for m in re.finditer(r'\\mathcal\{([^{}]*)\}', t):
        print(f'{f}: mathcal{{{m.group(1)}}} -> needs sub')
