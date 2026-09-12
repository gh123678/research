# 文档工具

本目录保留通用 Markdown/PDF 排版工具和图形工具。旧论文拼接器、旧稿专用校验器和旧手机教程生成器已随过时材料删除。

使用 `build_pdf.py` 时，请显式指定实际存在的输入和所需输出；该旧通用工具的默认路径来自已删除的旧稿，不应直接使用默认参数。

```text
python tools/build_pdf.py --src <现有文档.md> --out <输出文件.pdf> --lang zh
```

构建过程会按需重新生成 `_build_tmp.md` 和 `_build_tmp.html`。当前研究内容从 [当前研究状态](../ACTIVE_WORKSPACE.md) 阅读。
