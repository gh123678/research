"""Verify that the English and Chinese Section 3 manuscripts stay synchronized.

The checker is intentionally independent of the PDF build.  It validates the
authoritative English sectional source, the generated English full manuscript,
and the hand-maintained Chinese full manuscript against a shared manifest.

The built-in manifest can be replaced (top-level key by top-level key) with a
JSON file:

    python tools/verify_bilingual_manuscript.py --manifest path/to/manifest.json

Exit codes:
    0  all checks passed
    1  manuscript differences or claim-boundary violations were found
    2  a file, manifest, or parser configuration error prevented validation
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import difflib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_ROOT / "论文_草稿"


DEFAULT_MANIFEST: dict[str, Any] = {
    "section": "3",
    "headings": [
        {
            "number": "3.1",
            "en": "Exact model class and complete prompt",
            "zh": "精确模型类别与完整 prompt",
        },
        {
            "number": "3.2",
            "en": "Proof of Theorem 3.1: retrieval and residual formation",
            "zh": "定理 3.1 的证明：检索与 residual 构造",
        },
        {
            "number": "3.3",
            "en": "Proof of Theorem 3.1: write-back and assembly",
            "zh": "定理 3.1 的证明：写回与组装",
        },
        {
            "number": "3.4",
            "en": "From an update operator to control",
            "zh": "从更新算子到控制",
        },
        {
            "number": "3.5",
            "en": "Internal Boltzmann action attention",
            "zh": "内部 Boltzmann 动作注意力",
        },
        {
            "number": "3.6",
            "en": "Softmax approximation of greedy action selection",
            "zh": "Softmax 对贪心动作选择的近似",
        },
        {
            "number": "3.7",
            "en": "Stronger guarantee for the approximate-greedy route",
            "zh": "近似贪心路径的更强保证",
        },
        {
            "number": "3.8",
            "en": "Finite-logit deviations without equality masks",
            "zh": "无等值掩码的有限 logit 偏差",
        },
        {
            "number": "3.9",
            "en": "Policy consequence",
            "zh": "策略层面的结论",
        },
    ],
    "results": [
        {
            "number": "3.1",
            "en_kind": "Theorem",
            "zh_kind": "定理",
            "en_title_contains": "structured equality routing",
            "zh_title_contains": "结构化等值路由",
        },
        {
            "number": "3.2",
            "en_kind": "Corollary",
            "zh_kind": "推论",
            "en_title_contains": "sequential SARSA equivalence",
            "zh_title_contains": "顺序 SARSA 等价",
        },
        {
            "number": "3.3",
            "en_kind": "Proposition",
            "zh_kind": "命题",
            "en_title_contains": "Expected-SARSA identity",
            "zh_title_contains": "Expected-SARSA 恒等式",
        },
        {
            "number": "3.4",
            "en_kind": "Lemma",
            "zh_kind": "引理",
            "en_title_contains": "entropy bound for softmax action aggregation",
            "zh_title_contains": "softmax 动作聚合的熵界",
        },
        {
            "number": "3.5",
            "en_kind": "Corollary",
            "zh_kind": "推论",
            "en_title_contains": "greedy-target deviation",
            "zh_title_contains": "贪心目标偏差",
        },
        {
            "number": "3.6",
            "en_kind": "Theorem",
            "zh_kind": "定理",
            "en_title_contains": "bounded perturbation of Bellman optimality",
            "zh_title_contains": "Bellman 最优性的有界扰动",
        },
        {
            "number": "3.7",
            "en_kind": "Proposition",
            "zh_kind": "命题",
            "en_title_contains": "finite-logit end-to-end update error",
            "zh_title_contains": "有限 logit 端到端更新误差",
        },
        {
            "number": "3.8",
            "en_kind": "Corollary",
            "zh_kind": "推论",
        },
    ],
    "equations": [
        {"tag": "3.1", "label": "eq:prompt"},
        {"tag": "3.2", "label": "eq:block1-current"},
        {"tag": "3.3", "label": "eq:block1-next"},
        {"tag": "3.4", "label": "eq:residual-ffn"},
        {"tag": "3.5", "label": "eq:write-back"},
        {"tag": "3.6", "label": "eq:final-update"},
        {"tag": "3.7", "label": "eq:finite-retrieval"},
        {"tag": "3.8", "label": "eq:finite-write-back"},
        {"tag": "3.9", "label": "eq:finite-end-to-end-error"},
    ],
    "required_claim_limits": [
        {
            "id": "structured_equality_routing_is_external",
            "en_all": [
                r"externally supplied",
                r"input-dependent",
                r"equality (?:routing )?masks?",
            ],
            "zh_all": [
                r"外部(?:提供|给定|供应)",
                r"输入(?:相关|依赖)",
                r"(?:等值|相等).{0,24}(?:路由|掩码)",
            ],
        },
        {
            "id": "visited_null_support_is_external",
            "en_all": [
                r"visited[- ]query",
                r"null[- ]token|null token",
                r"external(?:ly)? (?:supplied )?(?:support|routing|gate)",
            ],
            "zh_all": [
                r"已访问(?:查询|query)",
                r"空 token|null token",
                r"外部.{0,24}(?:支持|路由|门控)",
            ],
        },
        {
            "id": "no_learned_routing_claim",
            "en_all": [
                r"does not (?:prove|assert).{0,180}(?:learn|discover).{0,100}routing",
            ],
            "zh_all": [
                r"不(?:证明|声称|表明).{0,180}(?:学习|学会|发现).{0,100}路由",
            ],
        },
        {
            "id": "fixed_construction_parameters",
            "en_all": [
                r"fixed.{0,160}\bm\b.{0,80}\bN\b.{0,100}\\gamma.{0,100}\\alpha",
            ],
            "zh_all": [
                r"固定.{0,160}\bm\b.{0,80}\bN\b.{0,100}\\gamma.{0,100}\\alpha",
            ],
        },
        {
            "id": "variable_stepsize_needs_extra_mechanism",
            "en_all": [
                r"variable.{0,80}(?:step size|stepsize)|Robbins--Monro",
                r"family of blocks|multiplication mechanism",
            ],
            "zh_all": [
                r"可变.{0,40}步长|Robbins--Monro",
                r"模块族|乘法机制",
            ],
        },
        {
            "id": "scratch_reset_between_rounds",
            "en_all": [
                r"(?:rebuild.{0,100}prompt|prompt.{0,100}rebuild)|scratch.{0,100}(?:clear|reset|reinitial)",
            ],
            "zh_all": [
                r"(?:重建|重新构造).{0,50}prompt|scratch.{0,100}(?:清空|重置|重新初始化)",
            ],
        },
        {
            "id": "three_update_denominators_are_distinguished",
            "en_all": [
                r"per-pair mean",
                r"global[- ]batch|1\s*/\s*N",
                r"singleton",
            ],
            "zh_all": [
                r"(?:逐对|按对|每对).{0,20}均值",
                r"全局.{0,20}批|1\s*/\s*N",
                r"单例|单样本|单转移",
            ],
        },
        {
            "id": "no_layernorm_or_dropout",
            "en_all": [r"(?:without|no) LayerNorm.{0,40}(?:and|or|/) (?:no )?dropout"],
            "zh_all": [r"(?:不含|没有|无) LayerNorm.{0,40}(?:和|或|、|/) ?dropout"],
        },
        {
            "id": "bidirectional_not_decoder_causal",
            "en_all": [
                r"bidirectional.{0,60}(?:structured )?attention",
                r"not (?:a )?decoder-only causal Transformer",
            ],
            "zh_all": [
                r"双向.{0,60}(?:结构化)?注意力",
                r"不是.{0,40}(?:仅解码器|decoder-only).{0,30}(?:因果 )?Transformer",
            ],
        },
    ],
    "forbidden_claims": [
        {
            "id": "ordinary_structured_masks",
            "en": r"\bordinary structured masks?\b",
            "zh": r"普通(?:的)?结构化掩码",
        },
        {
            "id": "fully_mask_free",
            "en": r"\bfully mask-free\b|\bmask-free (?:construction|Transformer|implementation)\b",
            "zh": r"完全无掩码|无掩码(?:构造|Transformer|实现)",
        },
        {
            "id": "model_learns_routing",
            "en": r"(?:model|Transformer).{0,60}(?:learns?|discovers?).{0,40}routing",
            "zh": r"(?:模型|Transformer).{0,60}(?:学习|学会|发现).{0,40}路由",
            "allowed_en": [
                r"does not (?:prove|assert).{0,180}(?:Transformer).{0,80}(?:learn|discover).{0,40}routing",
                r"not a claim.{0,120}(?:Transformer).{0,80}(?:learn|discover).{0,40}routing",
                r"(?:does|do|claims?) not (?:prove|assert|claim|show|establish).{0,220}(?:Transformer).{0,100}(?:learn|discover).{0,60}routing",
            ],
            "allowed_zh": [
                r"不(?:证明|声称|表明).{0,180}(?:模型|Transformer).{0,80}(?:学习|学会|发现).{0,40}路由",
                r"没有(?:证明|声称|表明).{0,180}(?:模型|Transformer).{0,80}(?:学习|学会|发现).{0,40}路由",
            ],
        },
        {
            "id": "ordinary_causal_mask",
            "en": r"\bordinary causal masks?\b",
            "zh": r"普通(?:的)?因果掩码",
        },
        {
            "id": "unqualified_standard_transformer",
            "en": r"\bstandard(?:-softmax)? Transformer\b",
            "zh": r"标准(?: softmax)? Transformer",
            "allowed_en": [
                r"structured equality routing",
                r"externally supplied.{0,80}equality",
                r"not (?:a )?standard(?:-softmax)? Transformer",
            ],
            "allowed_zh": [
                r"结构化等值路由",
                r"外部.{0,80}(?:等值|相等)",
                r"不是标准(?: softmax)? Transformer",
            ],
        },
    ],
}


HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
SUBSECTION_RE = re.compile(r"^#{2,6}[ \t]+(3\.\d+)[.。]?[ \t]+(.+?)[ \t]*$", re.MULTILINE)
BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
EN_RESULT_RE = re.compile(
    r"^(Theorem|Corollary|Proposition|Lemma)\s+(3\.\d+)"
    r"(?:\s*\((.*?)\))?\s*[.]?$",
    re.DOTALL,
)
ZH_RESULT_RE = re.compile(
    r"^(定理|推论|命题|引理)\s*(3\.\d+)"
    r"(?:\s*[（(](.*?)[）)])?\s*[。. ]?$",
    re.DOTALL,
)
TAG_RE = re.compile(r"\\tag\*?\{([^{}]+)\}")
LABEL_RE = re.compile(r"\\label\{([^{}]+)\}")


@dataclass(frozen=True)
class Heading:
    number: str
    title: str


@dataclass(frozen=True)
class ResultMarker:
    kind: str
    number: str
    title: str


@dataclass(frozen=True)
class EquationMarker:
    tag: str
    label: str


@dataclass(frozen=True)
class SectionSlice:
    text: str
    start: int
    end: int
    has_numbered_h1: bool


@dataclass
class CheckResult:
    name: str
    issues: list[str]

    @property
    def passed(self) -> bool:
        return not self.issues


class ConfigurationError(RuntimeError):
    """Raised when inputs or the manifest prevent meaningful validation."""


def _normalise_dashes(text: str) -> str:
    return text.translate(str.maketrans({"–": "-", "—": "-", "‑": "-", "−": "-"}))


def normalise_title(text: str) -> str:
    text = _normalise_dashes(text).replace("：", ":")
    return re.sub(r"\s+", " ", text.strip()).casefold()


def normalise_tag(text: str) -> str:
    return text.strip().strip("()（）")


def normalise_markdown(text: str) -> str:
    """Normalise formatting noise without hiding content differences."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"cannot read {path}: {exc}") from exc


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_manifest(path: Path | None) -> dict[str, Any]:
    manifest = dict(DEFAULT_MANIFEST)
    if path is not None:
        try:
            override = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"cannot load manifest {path}: {exc}") from exc
        if not isinstance(override, dict):
            raise ConfigurationError("manifest root must be a JSON object")
        manifest.update(override)
    validate_manifest(manifest)
    return manifest


def _require_list(manifest: dict[str, Any], key: str) -> list[Any]:
    value = manifest.get(key)
    if not isinstance(value, list):
        raise ConfigurationError(f"manifest key {key!r} must be a list")
    return value


def _duplicate_values(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def validate_manifest(manifest: dict[str, Any]) -> None:
    section = manifest.get("section")
    if not isinstance(section, str) or not re.fullmatch(r"\d+", section):
        raise ConfigurationError("manifest 'section' must be a numeric string")

    headings = _require_list(manifest, "headings")
    results = _require_list(manifest, "results")
    equations = _require_list(manifest, "equations")
    _require_list(manifest, "required_claim_limits")
    _require_list(manifest, "forbidden_claims")

    for index, item in enumerate(headings):
        if not isinstance(item, dict) or not all(k in item for k in ("number", "en", "zh")):
            raise ConfigurationError(f"headings[{index}] needs number, en, and zh")
    for index, item in enumerate(results):
        required = ("number", "en_kind", "zh_kind")
        if not isinstance(item, dict) or not all(k in item for k in required):
            raise ConfigurationError(f"results[{index}] needs number, en_kind, and zh_kind")
    for index, item in enumerate(equations):
        if not isinstance(item, dict) or not all(k in item for k in ("tag", "label")):
            raise ConfigurationError(f"equations[{index}] needs tag and label")

    uniqueness_checks = {
        "heading numbers": [str(item["number"]) for item in headings],
        "result numbers": [str(item["number"]) for item in results],
        "equation tags": [normalise_tag(str(item["tag"])) for item in equations],
        "equation labels": [str(item["label"]).strip() for item in equations],
    }
    for name, values in uniqueness_checks.items():
        duplicates = _duplicate_values(values)
        if duplicates:
            raise ConfigurationError(f"manifest has duplicate {name}: {duplicates}")


def extract_section(text: str, section_number: str, source_name: str) -> SectionSlice:
    numbered_h1 = re.compile(
        rf"^#[ \t]+{re.escape(section_number)}[.。][ \t]+.+?$",
        re.MULTILINE,
    )
    match = numbered_h1.search(text)
    if match:
        next_numbered_h1 = re.compile(r"^#[ \t]+\d+[.。][ \t]+.+?$", re.MULTILINE)
        next_match = next_numbered_h1.search(text, match.end())
        end = next_match.start() if next_match else len(text)
        return SectionSlice(text=text[match.start():end], start=match.start(), end=end, has_numbered_h1=True)

    subsection = re.compile(
        rf"^#{{2,6}}[ \t]+{re.escape(section_number)}\.\d+[.。]?[ \t]+.+?$",
        re.MULTILINE,
    )
    if subsection.search(text):
        return SectionSlice(text=text, start=0, end=len(text), has_numbered_h1=False)
    raise ConfigurationError(f"cannot locate Section {section_number} in {source_name}")


def strip_section_title(section: SectionSlice) -> str:
    text = section.text
    headings = list(HEADING_RE.finditer(text))
    if not headings:
        return text
    first = headings[0]
    title = first.group(2).strip()
    if section.has_numbered_h1 or not re.match(r"^3\.\d+\b", title):
        return text[:first.start()] + text[first.end():]
    return text


def parse_headings(section_text: str) -> list[Heading]:
    return [Heading(number=m.group(1), title=m.group(2).strip()) for m in SUBSECTION_RE.finditer(section_text)]


def parse_results(section_text: str, language: str) -> list[ResultMarker]:
    parser = EN_RESULT_RE if language == "en" else ZH_RESULT_RE
    markers: list[ResultMarker] = []
    for bold in BOLD_RE.finditer(section_text):
        content = re.sub(r"\s+", " ", bold.group(1).strip())
        match = parser.match(content)
        if not match:
            continue
        title = (match.group(3) or "").strip()
        markers.append(ResultMarker(kind=match.group(1), number=match.group(2), title=title))
    return markers


def parse_equations(section_text: str) -> tuple[list[EquationMarker], list[str]]:
    tags = [(match.start(), normalise_tag(match.group(1))) for match in TAG_RE.finditer(section_text)]
    labels = [(match.start(), match.group(1).strip()) for match in LABEL_RE.finditer(section_text)]
    issues: list[str] = []

    duplicate_tags = _duplicate_values(tag for _, tag in tags)
    duplicate_labels = _duplicate_values(label for _, label in labels)
    if duplicate_tags:
        issues.append(f"duplicate equation tags: {duplicate_tags}")
    if duplicate_labels:
        issues.append(f"duplicate equation labels: {duplicate_labels}")
    if len(tags) != len(labels):
        issues.append(f"tag/label count differs: {len(tags)} tag(s), {len(labels)} label(s)")

    paired = [
        EquationMarker(tag=tag[1], label=label[1])
        for tag, label in zip(tags, labels)
    ]
    return paired, issues


def _format_sequence(items: Sequence[Any]) -> str:
    return "[" + ", ".join(str(item) for item in items) + "]"


def check_headings(
    actual: Sequence[Heading], expected_entries: Sequence[dict[str, Any]], language: str
) -> list[str]:
    issues: list[str] = []
    expected_numbers = [str(item["number"]) for item in expected_entries]
    actual_numbers = [item.number for item in actual]
    if actual_numbers != expected_numbers:
        issues.append(
            "ordered heading numbers differ: "
            f"expected {_format_sequence(expected_numbers)}, actual {_format_sequence(actual_numbers)}"
        )

    duplicates = _duplicate_values(actual_numbers)
    if duplicates:
        issues.append(f"duplicate heading numbers: {duplicates}")

    expected_by_number = {str(item["number"]): str(item[language]) for item in expected_entries}
    for heading in actual:
        expected_title = expected_by_number.get(heading.number)
        if expected_title is None:
            issues.append(f"unexpected heading {heading.number}: {heading.title!r}")
        elif normalise_title(heading.title) != normalise_title(expected_title):
            issues.append(
                f"heading {heading.number} title differs: "
                f"expected {expected_title!r}, actual {heading.title!r}"
            )
    return issues


def check_results(
    actual: Sequence[ResultMarker], expected_entries: Sequence[dict[str, Any]], language: str
) -> list[str]:
    issues: list[str] = []
    kind_key = f"{language}_kind"
    title_key = f"{language}_title_contains"
    expected_keys = [(str(item["number"]), str(item[kind_key])) for item in expected_entries]
    actual_keys = [(item.number, item.kind) for item in actual]
    if actual_keys != expected_keys:
        issues.append(
            "ordered theorem markers differ: "
            f"expected {_format_sequence(expected_keys)}, actual {_format_sequence(actual_keys)}"
        )

    duplicates = _duplicate_values(item.number for item in actual)
    if duplicates:
        issues.append(f"duplicate theorem/result numbers: {duplicates}")

    expected_by_number = {str(item["number"]): item for item in expected_entries}
    for marker in actual:
        expected = expected_by_number.get(marker.number)
        if expected is None:
            continue
        expected_kind = str(expected[kind_key])
        if marker.kind != expected_kind:
            issues.append(
                f"result {marker.number} kind differs: expected {expected_kind!r}, actual {marker.kind!r}"
            )
        title_fragment = expected.get(title_key)
        if title_fragment and normalise_title(str(title_fragment)) not in normalise_title(marker.title):
            issues.append(
                f"result {marker.number} title lacks {title_fragment!r}: actual {marker.title!r}"
            )
    return issues


def check_equations(
    actual: Sequence[EquationMarker], expected_entries: Sequence[dict[str, Any]]
) -> list[str]:
    expected = [
        EquationMarker(tag=normalise_tag(str(item["tag"])), label=str(item["label"]).strip())
        for item in expected_entries
    ]
    if list(actual) == expected:
        return []

    expected_pairs = [f"{item.tag}/{item.label}" for item in expected]
    actual_pairs = [f"{item.tag}/{item.label}" for item in actual]
    issues = [
        "ordered equation tag/label pairs differ: "
        f"expected {_format_sequence(expected_pairs)}, actual {_format_sequence(actual_pairs)}"
    ]
    expected_set = set(expected_pairs)
    actual_set = set(actual_pairs)
    missing = [item for item in expected_pairs if item not in actual_set]
    unexpected = [item for item in actual_pairs if item not in expected_set]
    if missing:
        issues.append(f"missing equation pairs: {missing}")
    if unexpected:
        issues.append(f"unexpected equation pairs: {unexpected}")
    return issues


def check_claim_limits(
    section_text: str, entries: Sequence[dict[str, Any]], language: str
) -> list[str]:
    issues: list[str] = []
    key = f"{language}_all"
    for entry in entries:
        patterns = entry.get(key, [])
        if not isinstance(patterns, list):
            issues.append(f"manifest claim {entry.get('id', '<unnamed>')} has non-list {key}")
            continue
        missing = [
            pattern
            for pattern in patterns
            if not re.search(pattern, section_text, flags=re.IGNORECASE | re.DOTALL)
        ]
        if missing:
            issues.append(f"{entry.get('id', '<unnamed>')}: missing pattern(s) {missing}")
    return issues


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _line_containing(text: str, offset: int) -> str:
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def check_forbidden_claims(
    text: str, entries: Sequence[dict[str, Any]], language: str, source_name: str
) -> list[str]:
    issues: list[str] = []
    pattern_key = language
    allowed_key = f"allowed_{language}"
    for entry in entries:
        pattern = entry.get(pattern_key)
        if not pattern:
            continue
        allowed_patterns = entry.get(allowed_key, [])
        for match in re.finditer(str(pattern), text, flags=re.IGNORECASE | re.DOTALL):
            line = _line_containing(text, match.start())
            if any(re.search(p, line, flags=re.IGNORECASE) for p in allowed_patterns):
                continue
            issues.append(
                f"{entry.get('id', '<unnamed>')} at {source_name}:{_line_number(text, match.start())}: "
                f"{line[:220]!r}"
            )
    return issues


def compare_generated_source(source: SectionSlice, generated: SectionSlice) -> list[str]:
    source_body = normalise_markdown(strip_section_title(source))
    generated_body = normalise_markdown(strip_section_title(generated))
    if source_body == generated_body:
        return []

    diff = list(
        difflib.unified_diff(
            source_body.splitlines(),
            generated_body.splitlines(),
            fromfile="preliminaries.md",
            tofile="full_paper.md#section-3",
            lineterm="",
            n=2,
        )
    )
    preview = "\n".join(diff[:40])
    if len(diff) > 40:
        preview += f"\n... {len(diff) - 40} additional diff line(s) omitted"
    return ["generated English Section 3 differs from its authoritative source:\n" + preview]


def compare_parsed_sequences(
    name: str, left: Sequence[Any], right: Sequence[Any]
) -> CheckResult:
    if list(left) == list(right):
        return CheckResult(name=name, issues=[])
    return CheckResult(
        name=name,
        issues=[f"parsed sequences differ: left={_format_sequence(left)}, right={_format_sequence(right)}"],
    )


def run_checks(
    manifest: dict[str, Any], english_path: Path, generated_path: Path, chinese_path: Path
) -> list[CheckResult]:
    section_number = str(manifest["section"])
    english_text = read_text(english_path)
    generated_text = read_text(generated_path)
    chinese_text = read_text(chinese_path)

    english_section = extract_section(english_text, section_number, str(english_path))
    generated_section = extract_section(generated_text, section_number, str(generated_path))
    chinese_section = extract_section(chinese_text, section_number, str(chinese_path))

    english_headings = parse_headings(english_section.text)
    generated_headings = parse_headings(generated_section.text)
    chinese_headings = parse_headings(chinese_section.text)
    english_results = parse_results(english_section.text, "en")
    generated_results = parse_results(generated_section.text, "en")
    chinese_results = parse_results(chinese_section.text, "zh")
    english_equations, english_equation_parse_issues = parse_equations(english_section.text)
    generated_equations, generated_equation_parse_issues = parse_equations(generated_section.text)
    chinese_equations, chinese_equation_parse_issues = parse_equations(chinese_section.text)

    checks = [
        CheckResult(
            "generated English Section 3 matches preliminaries.md",
            compare_generated_source(english_section, generated_section),
        ),
        CheckResult(
            "English Section 3 headings match manifest",
            check_headings(english_headings, manifest["headings"], "en"),
        ),
        CheckResult(
            "generated English Section 3 headings match source",
            [] if generated_headings == english_headings else [
                f"source={_format_sequence(english_headings)}, generated={_format_sequence(generated_headings)}"
            ],
        ),
        CheckResult(
            "Chinese Section 3 headings match manifest",
            check_headings(chinese_headings, manifest["headings"], "zh"),
        ),
        CheckResult(
            "English theorem/result markers match manifest",
            check_results(english_results, manifest["results"], "en"),
        ),
        CheckResult(
            "generated English theorem/result markers match source",
            [] if generated_results == english_results else [
                f"source={_format_sequence(english_results)}, generated={_format_sequence(generated_results)}"
            ],
        ),
        CheckResult(
            "Chinese theorem/result markers match manifest",
            check_results(chinese_results, manifest["results"], "zh"),
        ),
        CheckResult(
            "English equation tags and labels match manifest",
            english_equation_parse_issues + check_equations(english_equations, manifest["equations"]),
        ),
        CheckResult(
            "generated English equation tags and labels match source",
            generated_equation_parse_issues
            + ([] if generated_equations == english_equations else [
                f"source={_format_sequence(english_equations)}, generated={_format_sequence(generated_equations)}"
            ]),
        ),
        CheckResult(
            "Chinese equation tags and labels match manifest",
            chinese_equation_parse_issues + check_equations(chinese_equations, manifest["equations"]),
        ),
        compare_parsed_sequences(
            "English and Chinese equation tag/label order agrees",
            english_equations,
            chinese_equations,
        ),
        CheckResult(
            "English claim limitations are explicit",
            check_claim_limits(
                english_section.text, manifest["required_claim_limits"], "en"
            ),
        ),
        CheckResult(
            "Chinese claim limitations are explicit",
            check_claim_limits(
                chinese_section.text, manifest["required_claim_limits"], "zh"
            ),
        ),
    ]

    # Avoid reporting the same Section 3 wording twice: scan the authoritative
    # section source, then only the generated full manuscript outside Section 3.
    generated_outside_section = (
        generated_text[:generated_section.start] + generated_text[generated_section.end:]
    )
    checks.extend(
        [
            CheckResult(
                "English Section 3 avoids forbidden broad claims",
                check_forbidden_claims(
                    english_section.text,
                    manifest["forbidden_claims"],
                    "en",
                    english_path.name,
                ),
            ),
            CheckResult(
                "English global narrative avoids forbidden broad claims",
                check_forbidden_claims(
                    generated_outside_section,
                    manifest["forbidden_claims"],
                    "en",
                    generated_path.name,
                ),
            ),
            CheckResult(
                "Chinese manuscript avoids forbidden broad claims",
                check_forbidden_claims(
                    chinese_text,
                    manifest["forbidden_claims"],
                    "zh",
                    chinese_path.name,
                ),
            ),
        ]
    )
    return checks


def print_report(checks: Sequence[CheckResult]) -> None:
    failed = [check for check in checks if not check.passed]
    print("Bilingual manuscript verification")
    print(f"Checks: {len(checks)} total, {len(checks) - len(failed)} passed, {len(failed)} failed")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}")
        for issue in check.issues:
            indented = issue.replace("\n", "\n      ")
            print(f"    - {indented}")
    if failed:
        print("\nRESULT: FAILED. Synchronize the listed manuscript structures and claim boundaries.")
    else:
        print("\nRESULT: PASSED. English source, generated Section 3, and Chinese manuscript agree.")


def print_json_report(checks: Sequence[CheckResult]) -> None:
    payload = {
        "passed": all(check.passed for check in checks),
        "checks": [
            {**asdict(check), "passed": check.passed}
            for check in checks
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--english-source",
        default=str(PAPER_DIR / "preliminaries.md"),
        help="authoritative English Section 3 source",
    )
    parser.add_argument(
        "--generated-english",
        default=str(PAPER_DIR / "full_paper.md"),
        help="generated English full manuscript",
    )
    parser.add_argument(
        "--chinese",
        default=str(PAPER_DIR / "full_paper_中文版.md"),
        help="Chinese full manuscript",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="JSON file overriding top-level keys in the built-in manifest",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable JSON report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        manifest_path = resolve_path(args.manifest) if args.manifest else None
        manifest = load_manifest(manifest_path)
        checks = run_checks(
            manifest=manifest,
            english_path=resolve_path(args.english_source),
            generated_path=resolve_path(args.generated_english),
            chinese_path=resolve_path(args.chinese),
        )
    except ConfigurationError as exc:
        if args.json:
            print(json.dumps({"passed": False, "configuration_error": str(exc)}, ensure_ascii=False, indent=2))
        else:
            print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print_json_report(checks)
    else:
        print_report(checks)
    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
