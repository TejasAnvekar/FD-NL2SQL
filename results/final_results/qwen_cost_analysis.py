#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/mnt/data1/srchowd3/FD-NL2SQL/results/final_results')
DEFAULT_OUT_DIR = ROOT / 'qwen_cost_analysis'

RUN_SPECS = [
    {
        'method_family': 'prompting_baseline',
        'method_label': 'Zero-Shot',
        'run_name': 'question_table_copy_llm_naive_aug1500_qwen30b_zeroshot',
        'tokens_source': 'prompting',
        'responses_csv': '/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs/question_table_copy_llm_naive_aug1500_qwen30b_zeroshot/responses.csv',
        'tabular_summary_json': '/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs/question_table_copy_llm_naive_aug1500_qwen30b_zeroshot/tabular_eval_v3/summary.json',
        'derived_summary_json': '/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs/question_table_copy_llm_naive_aug1500_qwen30b_zeroshot/derived_eval_v1/summary.json',
    },
    {
        'method_family': 'prompting_baseline',
        'method_label': 'Few-Shot',
        'run_name': 'question_table_copy_llm_naive_aug1500_qwen30b_fewshot',
        'tokens_source': 'prompting',
        'responses_csv': '/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs/question_table_copy_llm_naive_aug1500_qwen30b_fewshot/responses.csv',
        'tabular_summary_json': '/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs/question_table_copy_llm_naive_aug1500_qwen30b_fewshot/tabular_eval_v3/summary.json',
        'derived_summary_json': '/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs/question_table_copy_llm_naive_aug1500_qwen30b_fewshot/derived_eval_v1/summary.json',
    },
    {
        'method_family': 'prompting_baseline',
        'method_label': 'CoT',
        'run_name': 'question_table_copy_llm_naive_aug1500_qwen30b_cot',
        'tokens_source': 'prompting',
        'responses_csv': '/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs/question_table_copy_llm_naive_aug1500_qwen30b_cot/responses.csv',
        'tabular_summary_json': '/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs/question_table_copy_llm_naive_aug1500_qwen30b_cot/tabular_eval_v3/summary.json',
        'derived_summary_json': '/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs/question_table_copy_llm_naive_aug1500_qwen30b_cot/derived_eval_v1/summary.json',
    },
    {
        'method_family': 'tablegpt',
        'method_label': 'TableGPT2',
        'run_name': 'question_table_copy_llm_naive_aug1500_tablegpt2_7b',
        'tokens_source': 'prompting',
        'responses_csv': '/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs/question_table_copy_llm_naive_aug1500_tablegpt2_7b/responses.csv',
        'tabular_summary_json': '/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs/question_table_copy_llm_naive_aug1500_tablegpt2_7b/tabular_eval_v3/summary.json',
        'derived_summary_json': '/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs/question_table_copy_llm_naive_aug1500_tablegpt2_7b/derived_eval_v1/summary.json',
    },
    {
        'method_family': 'planner_executor',
        'method_label': 'SCoPE',
        'run_name': 'question_table_copy_llm_planexec_aug1500_qwen_exec_qwen_plan',
        'tokens_source': 'planner_executor',
        'all_question_results_csv': '/mnt/data2/srchowd3/fdnl2sql_runs/question_table_copy_llm_planexec_aug1500_qwen_exec_qwen_plan/all_question_results.csv',
        'tabular_summary_json': '/mnt/data2/srchowd3/fdnl2sql_runs/question_table_copy_llm_planexec_aug1500_qwen_exec_qwen_plan/tabular_eval_v3/summary.json',
        'derived_summary_json': '/mnt/data2/srchowd3/fdnl2sql_runs/question_table_copy_llm_planexec_aug1500_qwen_exec_qwen_plan/derived_eval_v1/summary.json',
    },
    {
        'method_family': 'blendsql',
        'method_label': 'BlendSQL',
        'run_name': 'qwen_full1500_batch12',
        'tokens_source': 'blendsql',
        'all_question_results_csv': '/mnt/data2/srchowd3/blendsql/fdnl2sql_runs/qwen_full1500_batch12/all_question_results.csv',
        'tabular_summary_json': '/mnt/data2/srchowd3/blendsql/fdnl2sql_runs/qwen_full1500_batch12/tabular_eval_v3/summary.json',
        'derived_summary_json': '/mnt/data2/srchowd3/blendsql/fdnl2sql_runs/qwen_full1500_batch12/derived_eval_v1/summary.json',
    },
    {
        'method_family': 'ehragent',
        'method_label': 'EhrAgent',
        'run_name': 'qwen30b_full1500_cost_merged',
        'tokens_source': 'ehragent',
        'all_question_results_csv': '/mnt/data2/srchowd3/EhrAgent/fdnl2sql_runs/qwen30b_full1500_cost_merged/all_question_results.csv',
        'tabular_summary_json': '/mnt/data2/srchowd3/EhrAgent/fdnl2sql_runs/qwen30b_full1500_cost_merged/tabular_eval_v3/summary.json',
        'derived_summary_json': '/mnt/data2/srchowd3/EhrAgent/fdnl2sql_runs/qwen30b_full1500_cost_merged/derived_eval_v1/summary.json',
    },
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Compute Qwen-only cross-method token/cost analysis and generate charts.')
    ap.add_argument('--out_dir', default=str(DEFAULT_OUT_DIR))
    ap.add_argument('--input_cost_per_million_tokens', type=float, default=-1.0)
    ap.add_argument('--output_cost_per_million_tokens', type=float, default=-1.0)
    ap.add_argument('--cost_currency', default='USD')
    return ap.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ''):
            return default
        return float(value)
    except Exception:
        return default


def estimate_cost(prompt_tokens: float, completion_tokens: float, in_cost: float, out_cost: float) -> Dict[str, Optional[float]]:
    if in_cost < 0 or out_cost < 0:
        return {
            'estimated_input_cost': None,
            'estimated_output_cost': None,
            'estimated_total_cost': None,
        }
    input_cost = (prompt_tokens / 1_000_000.0) * in_cost
    output_cost = (completion_tokens / 1_000_000.0) * out_cost
    return {
        'estimated_input_cost': input_cost,
        'estimated_output_cost': output_cost,
        'estimated_total_cost': input_cost + output_cost,
    }


def collect_prompting(spec: Dict[str, Any]) -> Dict[str, Any]:
    rows = read_csv_rows(Path(spec['responses_csv']))
    total_prompt = 0.0
    total_completion = 0.0
    total_tokens = 0.0
    error_count = 0
    stage_rows = []
    for row in rows:
        if (row.get('error') or '').strip():
            error_count += 1
        meta_raw = (row.get('model_meta_json') or '').strip()
        meta = {}
        if meta_raw:
            try:
                meta = json.loads(meta_raw)
            except Exception:
                meta = {}
        prompt = safe_float(meta.get('prompt_tokens'))
        completion = safe_float(meta.get('completion_tokens'))
        total = safe_float(meta.get('total_tokens'), prompt + completion)
        total_prompt += prompt
        total_completion += completion
        total_tokens += total
    qcount = len(rows)
    stage_rows.append({'method_label': spec['method_label'], 'stage': 'prompting', 'prompt_tokens_total': total_prompt, 'completion_tokens_total': total_completion, 'total_tokens_total': total_tokens, 'question_count': qcount})
    return {
        'question_count': qcount,
        'ok_count': qcount - error_count,
        'error_count': error_count,
        'total_model_calls': qcount,
        'avg_model_calls': (qcount / qcount) if qcount else 0.0,
        'total_prompt_tokens': total_prompt,
        'total_completion_tokens': total_completion,
        'total_tokens': total_tokens,
        'avg_prompt_tokens': (total_prompt / qcount) if qcount else 0.0,
        'avg_completion_tokens': (total_completion / qcount) if qcount else 0.0,
        'avg_total_tokens': (total_tokens / qcount) if qcount else 0.0,
        'stage_rows': stage_rows,
    }


def collect_planexec(spec: Dict[str, Any]) -> Dict[str, Any]:
    rows = read_csv_rows(Path(spec['all_question_results_csv']))
    qcount = len(rows)
    ok_count = sum(1 for r in rows if not (r.get('selector_error') or r.get('planner_error') or r.get('executor_error')))
    error_count = qcount - ok_count
    total_prompt = sum(safe_float(r.get('total_prompt_tokens')) for r in rows)
    total_completion = sum(safe_float(r.get('total_completion_tokens')) for r in rows)
    total_tokens = sum(safe_float(r.get('total_tokens')) for r in rows)
    total_model_calls = sum(safe_float(r.get('token_usage_stage_count'), 3.0) for r in rows)
    stage_rows = []
    for stage, pp, cp in [
        ('selector', 'selector_prompt_tokens', 'selector_completion_tokens'),
        ('planner', 'planner_prompt_tokens', 'planner_completion_tokens'),
        ('executor', 'executor_prompt_tokens', 'executor_completion_tokens'),
    ]:
        sp = sum(safe_float(r.get(pp)) for r in rows)
        sc = sum(safe_float(r.get(cp)) for r in rows)
        stage_rows.append({'method_label': spec['method_label'], 'stage': stage, 'prompt_tokens_total': sp, 'completion_tokens_total': sc, 'total_tokens_total': sp + sc, 'question_count': qcount})
    return {
        'question_count': qcount,
        'ok_count': ok_count,
        'error_count': error_count,
        'total_model_calls': total_model_calls,
        'avg_model_calls': (total_model_calls / qcount) if qcount else 0.0,
        'total_prompt_tokens': total_prompt,
        'total_completion_tokens': total_completion,
        'total_tokens': total_tokens,
        'avg_prompt_tokens': (total_prompt / qcount) if qcount else 0.0,
        'avg_completion_tokens': (total_completion / qcount) if qcount else 0.0,
        'avg_total_tokens': (total_tokens / qcount) if qcount else 0.0,
        'stage_rows': stage_rows,
    }


def collect_blendsql(spec: Dict[str, Any]) -> Dict[str, Any]:
    rows = read_csv_rows(Path(spec['all_question_results_csv']))
    qcount = len(rows)
    ok_count = sum(1 for r in rows if (r.get('status') or '') == 'ok')
    error_count = sum(1 for r in rows if (r.get('status') or '') == 'error')
    rel_prompt = sum(safe_float(r.get('relevance_prompt_tokens')) for r in rows)
    rel_completion = sum(safe_float(r.get('relevance_completion_tokens')) for r in rows)
    ans_prompt = sum(safe_float(r.get('answer_prompt_tokens')) for r in rows)
    ans_completion = sum(safe_float(r.get('answer_completion_tokens')) for r in rows)
    total_prompt = rel_prompt + ans_prompt
    total_completion = rel_completion + ans_completion
    total_tokens = total_prompt + total_completion
    stage_rows = [
        {'method_label': spec['method_label'], 'stage': 'relevance', 'prompt_tokens_total': rel_prompt, 'completion_tokens_total': rel_completion, 'total_tokens_total': rel_prompt + rel_completion, 'question_count': qcount},
        {'method_label': spec['method_label'], 'stage': 'answer', 'prompt_tokens_total': ans_prompt, 'completion_tokens_total': ans_completion, 'total_tokens_total': ans_prompt + ans_completion, 'question_count': qcount},
    ]
    return {
        'question_count': qcount,
        'ok_count': ok_count,
        'error_count': error_count,
        'total_model_calls': None,
        'avg_model_calls': None,
        'total_prompt_tokens': total_prompt,
        'total_completion_tokens': total_completion,
        'total_tokens': total_tokens,
        'avg_prompt_tokens': (total_prompt / qcount) if qcount else 0.0,
        'avg_completion_tokens': (total_completion / qcount) if qcount else 0.0,
        'avg_total_tokens': (total_tokens / qcount) if qcount else 0.0,
        'avg_elapsed_seconds': (sum(safe_float(r.get('elapsed_seconds')) for r in rows) / qcount) if qcount else 0.0,
        'stage_rows': stage_rows,
    }


def collect_ehragent(spec: Dict[str, Any]) -> Dict[str, Any]:
    rows = read_csv_rows(Path(spec['all_question_results_csv']))
    qcount = len(rows)
    ok_count = sum(1 for r in rows if (r.get('status') or '') == 'ok')
    error_count = sum(1 for r in rows if (r.get('status') or '') == 'error')
    total_prompt = sum(safe_float(r.get('prompt_tokens')) for r in rows)
    total_completion = sum(safe_float(r.get('completion_tokens')) for r in rows)
    total_tokens = sum(safe_float(r.get('total_tokens')) for r in rows)
    total_model_calls = sum(safe_float(r.get('num_model_calls')) for r in rows)
    stage_rows = [
        {'method_label': spec['method_label'], 'stage': 'ehragent', 'prompt_tokens_total': total_prompt, 'completion_tokens_total': total_completion, 'total_tokens_total': total_tokens, 'question_count': qcount}
    ]
    return {
        'question_count': qcount,
        'ok_count': ok_count,
        'error_count': error_count,
        'total_model_calls': total_model_calls,
        'avg_model_calls': (total_model_calls / qcount) if qcount else 0.0,
        'total_prompt_tokens': total_prompt,
        'total_completion_tokens': total_completion,
        'total_tokens': total_tokens,
        'avg_prompt_tokens': (total_prompt / qcount) if qcount else 0.0,
        'avg_completion_tokens': (total_completion / qcount) if qcount else 0.0,
        'avg_total_tokens': (total_tokens / qcount) if qcount else 0.0,
        'avg_elapsed_seconds': (sum(safe_float(r.get('elapsed_seconds')) for r in rows) / qcount) if qcount else 0.0,
        'stage_rows': stage_rows,
    }


def load_metrics(spec: Dict[str, Any]) -> Dict[str, Any]:
    tab = load_json(Path(spec['tabular_summary_json']))
    der = load_json(Path(spec['derived_summary_json']))
    return {
        'table_f1': safe_float(tab.get('avg_f1')),
        'row_jaccard': safe_float(tab.get('avg_row_jaccard')),
        'rouge_l_f1': safe_float(tab.get('avg_rouge_l_f1')),
        'chrf': safe_float(tab.get('avg_chrf')),
        'table_exact': safe_float(tab.get('exact_table_match_rate')),
        'derived_norm_target_f1': safe_float(der.get('avg_target_row_normalized_f1')),
        'derived_exact_target_f1': safe_float(der.get('avg_target_row_exact_f1')),
    }


def build_summary(spec: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if spec['tokens_source'] == 'prompting':
        usage = collect_prompting(spec)
    elif spec['tokens_source'] == 'planner_executor':
        usage = collect_planexec(spec)
    elif spec['tokens_source'] == 'blendsql':
        usage = collect_blendsql(spec)
    elif spec['tokens_source'] == 'ehragent':
        usage = collect_ehragent(spec)
    else:
        raise ValueError(spec['tokens_source'])

    metrics = load_metrics(spec)
    costs = estimate_cost(
        usage['total_prompt_tokens'],
        usage['total_completion_tokens'],
        args.input_cost_per_million_tokens,
        args.output_cost_per_million_tokens,
    )
    qcount = usage['question_count'] or 1
    avg_cost_per_question = (costs['estimated_total_cost'] / qcount) if costs['estimated_total_cost'] is not None else None
    f1_per_1k = (metrics['table_f1'] / (usage['avg_total_tokens'] / 1000.0)) if usage['avg_total_tokens'] else 0.0
    rouge_per_1k = (metrics['rouge_l_f1'] / (usage['avg_total_tokens'] / 1000.0)) if usage['avg_total_tokens'] else 0.0
    chrf_per_1k = (metrics['chrf'] / (usage['avg_total_tokens'] / 1000.0)) if usage['avg_total_tokens'] else 0.0
    return {
        'method_family': spec['method_family'],
        'method_label': spec['method_label'],
        'run_name': spec['run_name'],
        'question_count': usage['question_count'],
        'ok_count': usage['ok_count'],
        'error_count': usage['error_count'],
        'success_rate': (usage['ok_count'] / usage['question_count']) if usage['question_count'] else 0.0,
        'total_model_calls': usage.get('total_model_calls'),
        'avg_model_calls': usage.get('avg_model_calls'),
        'total_prompt_tokens': usage['total_prompt_tokens'],
        'total_completion_tokens': usage['total_completion_tokens'],
        'total_tokens': usage['total_tokens'],
        'avg_prompt_tokens': usage['avg_prompt_tokens'],
        'avg_completion_tokens': usage['avg_completion_tokens'],
        'avg_total_tokens': usage['avg_total_tokens'],
        'avg_elapsed_seconds': usage.get('avg_elapsed_seconds'),
        **metrics,
        'table_f1_per_1k_tokens': f1_per_1k,
        'rouge_l_f1_per_1k_tokens': rouge_per_1k,
        'chrf_per_1k_tokens': chrf_per_1k,
        'estimated_input_cost': costs['estimated_input_cost'],
        'estimated_output_cost': costs['estimated_output_cost'],
        'estimated_total_cost': costs['estimated_total_cost'],
        'avg_estimated_cost_per_question': avg_cost_per_question,
        'cost_currency': args.cost_currency,
        'stage_rows': usage['stage_rows'],
    }


def chart_grouped_metrics(rows: List[Dict[str, Any]], out_path: Path) -> None:
    labels = [r['method_label'] for r in rows]
    x = np.arange(len(labels))
    width = 0.24
    table = [r['table_f1'] * 100.0 for r in rows]
    rouge = [r['rouge_l_f1'] * 100.0 for r in rows]
    chrf = [r['chrf'] for r in rows]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, table, width, label='Table F1 %')
    ax.bar(x, rouge, width, label='ROUGE-L F1 %')
    ax.bar(x + width, chrf, width, label='CHRF')
    ax.set_ylabel('Score')
    ax.set_title('Qwen: Performance by Method')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def chart_avg_tokens(rows: List[Dict[str, Any]], out_path: Path) -> None:
    labels = [r['method_label'] for r in rows]
    prompt = [r['avg_prompt_tokens'] for r in rows]
    completion = [r['avg_completion_tokens'] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, prompt, label='Avg prompt tokens')
    ax.bar(x, completion, bottom=prompt, label='Avg completion tokens')
    ax.set_ylabel('Tokens / question')
    ax.set_title('Qwen: Average Tokens per Question')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def chart_metric_vs_tokens(
    rows: List[Dict[str, Any]],
    out_path: Path,
    *,
    metric_key: str,
    y_segments: List[Dict[str, Any]],
    y_label: str,
    title: str,
) -> None:
    x_segments = [
        {'lo': 17000.0, 'hi': 32000.0, 'ticks': [20000, 30000]},
        {'lo': 88000.0, 'hi': 90500.0, 'ticks': [90000]},
    ]
    x_gap = 500.0
    y_gap = 0.6

    label_offsets = {
        'Zero-Shot': (8, 8),
        'Few-Shot': (8, -2),
        'CoT': (8, 12),
        'TableGPT2': (8, 2),
        'SCoPE': (8, 8),
        'EhrAgent': (8, 6),
        'BlendSQL': (8, 6),
    }
    point_colors = {
        'Zero-Shot': '#1f77b4',
        'Few-Shot': '#ff7f0e',
        'CoT': '#2ca02c',
        'TableGPT2': '#9467bd',
        'SCoPE': '#d62728',
        'EhrAgent': '#8c564b',
        'BlendSQL': '#7f7f7f',
    }

    def piecewise_position(value: float, segments: List[Dict[str, Any]], gap: float) -> float:
        offset = 0.0
        for segment in segments:
            lo = float(segment['lo'])
            hi = float(segment['hi'])
            width = hi - lo
            if lo <= value <= hi:
                return offset + (value - lo)
            offset += width + gap
        raise ValueError(f'value {value} is outside configured segments')

    def segment_boundaries(segments: List[Dict[str, Any]], gap: float) -> List[float]:
        boundaries: List[float] = []
        offset = 0.0
        for idx, segment in enumerate(segments[:-1]):
            width = float(segment['hi']) - float(segment['lo'])
            boundaries.append(offset + width + gap / 2.0)
            offset += width + gap
        return boundaries

    def add_x_break(ax: Any, xpos: float, size: float = 0.015, gap_scale: float = 0.012) -> None:
        xmin, xmax = ax.get_xlim()
        xfrac = (xpos - xmin) / (xmax - xmin)
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1.3)
        ax.plot((xfrac - size, xfrac + size), (-gap_scale, +gap_scale), **kwargs)
        ax.plot((xfrac - size, xfrac + size), (-gap_scale - 0.02, +gap_scale - 0.02), **kwargs)

    def add_y_break(ax: Any, ypos: float, size: float = 0.02, gap_scale: float = 0.012) -> None:
        ymin, ymax = ax.get_ylim()
        yfrac = (ypos - ymin) / (ymax - ymin)
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1.3)
        ax.plot((-gap_scale, +gap_scale), (yfrac - size, yfrac + size), **kwargs)
        ax.plot((1 - gap_scale, 1 + gap_scale), (yfrac - size, yfrac + size), **kwargs)

    fig, ax = plt.subplots(figsize=(10.8, 7.8))

    x_tick_positions: List[float] = []
    x_tick_labels: List[str] = []
    for segment in x_segments:
        for tick in segment['ticks']:
            x_tick_positions.append(piecewise_position(float(tick), x_segments, x_gap))
            x_tick_labels.append(f"{int(tick/1000)}k")

    y_tick_positions: List[float] = []
    y_tick_labels: List[str] = []
    for segment in y_segments:
        for tick in segment['ticks']:
            y_tick_positions.append(piecewise_position(float(tick), y_segments, y_gap))
            y_tick_labels.append(str(int(tick)))

    for item in rows:
        x = piecewise_position(float(item['avg_total_tokens']), x_segments, x_gap)
        y = piecewise_position(float(item[metric_key]) * 100.0, y_segments, y_gap)
        color = point_colors.get(item['method_label'])
        ax.scatter([x], [y], s=360, color=color, edgecolors='white', linewidths=1.4, zorder=3)
        dx, dy = label_offsets.get(item['method_label'], (4, 4))
        ax.annotate(
            item['method_label'],
            (x, y),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=24,
            fontweight='bold',
        )

    total_x = sum(float(s['hi']) - float(s['lo']) for s in x_segments) + x_gap * (len(x_segments) - 1)
    total_y = sum(float(s['hi']) - float(s['lo']) for s in y_segments) + y_gap * (len(y_segments) - 1)
    ax.set_xlim(0.0, total_x)
    ax.set_ylim(0.0, total_y)
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)
    ax.set_xlabel('Average total tokens / question', fontsize=22, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=22, fontweight='bold')
    ax.set_title(title, fontsize=26, fontweight='bold', pad=20)
    ax.tick_params(axis='both', labelsize=19)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    ax.minorticks_off()
    ax.grid(False, which='both', axis='both')
    ax.xaxis.grid(False, which='both')
    ax.yaxis.grid(False, which='both')
    ax.set_axisbelow(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for xpos in segment_boundaries(x_segments, x_gap):
        add_x_break(ax, xpos)
    for ypos in segment_boundaries(y_segments, y_gap):
        add_y_break(ax, ypos)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def chart_f1_vs_tokens(rows: List[Dict[str, Any]], out_path: Path) -> None:
    chart_metric_vs_tokens(
        rows,
        out_path,
        metric_key='table_f1',
        y_segments=[
            {'lo': 11.0, 'hi': 13.0, 'ticks': [11, 12, 13]},
            {'lo': 32.0, 'hi': 34.0, 'ticks': [32, 33, 34]},
            {'lo': 43.0, 'hi': 45.0, 'ticks': [43, 44, 45]},
            {'lo': 54.0, 'hi': 64.0, 'ticks': [55, 56, 57, 58, 60, 62, 64]},
        ],
        y_label='Table F1 (%)',
        title='Qwen: Table F1 vs Token Budget',
    )


def chart_row_jaccard_vs_tokens(rows: List[Dict[str, Any]], out_path: Path) -> None:
    chart_metric_vs_tokens(
        rows,
        out_path,
        metric_key='row_jaccard',
        y_segments=[
            {'lo': 6.0, 'hi': 8.0, 'ticks': [6, 7, 8]},
            {'lo': 29.0, 'hi': 35.0, 'ticks': [29, 30, 31, 32, 33, 34, 35]},
            {'lo': 44.0, 'hi': 53.0, 'ticks': [44, 45, 46, 48, 50, 52, 53]},
        ],
        y_label='Row Jaccard (%)',
        title='Qwen: Row Jaccard vs Token Budget',
    )


def chart_f1_per_1k(rows: List[Dict[str, Any]], out_path: Path) -> None:
    ordered = sorted(rows, key=lambda r: r['table_f1_per_1k_tokens'], reverse=True)
    labels = [r['method_label'] for r in ordered]
    values = [r['table_f1_per_1k_tokens'] * 100.0 for r in ordered]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, values)
    ax.set_ylabel('Table F1 points per 1K tokens')
    ax.set_title('Qwen: Token-Normalized Efficiency')
    ax.tick_params(axis='x', rotation=25)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def chart_cost(rows: List[Dict[str, Any]], out_path: Path) -> None:
    cost_rows = [r for r in rows if r['avg_estimated_cost_per_question'] is not None]
    if not cost_rows:
        return
    labels = [r['method_label'] for r in cost_rows]
    values = [r['avg_estimated_cost_per_question'] for r in cost_rows]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, values)
    ax.set_ylabel(f"Average cost / question ({cost_rows[0]['cost_currency']})")
    ax.set_title('Qwen: Estimated Cost per Question')
    ax.tick_params(axis='x', rotation=25)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def chart_stage_breakdown(stage_rows: List[Dict[str, Any]], out_path: Path) -> None:
    # stacked total tokens/question by stage
    methods = []
    stage_names = []
    for row in stage_rows:
        if row['method_label'] not in methods:
            methods.append(row['method_label'])
        if row['stage'] not in stage_names:
            stage_names.append(row['stage'])
    stage_values = {stage: [] for stage in stage_names}
    for method in methods:
        subset = {r['stage']: r for r in stage_rows if r['method_label'] == method}
        for stage in stage_names:
            item = subset.get(stage)
            if item is None or not item['question_count']:
                stage_values[stage].append(0.0)
            else:
                stage_values[stage].append(item['total_tokens_total'] / item['question_count'])
    fig, ax = plt.subplots(figsize=(12, 6))
    bottoms = np.zeros(len(methods))
    for stage in stage_names:
        vals = np.array(stage_values[stage])
        ax.bar(methods, vals, bottom=bottoms, label=stage)
        bottoms += vals
    ax.set_ylabel('Average stage tokens / question')
    ax.set_title('Qwen: Stage-Level Token Breakdown')
    ax.tick_params(axis='x', rotation=25)
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    stage_rows: List[Dict[str, Any]] = []
    for spec in RUN_SPECS:
        summary = build_summary(spec, args)
        stage_rows.extend(summary.pop('stage_rows'))
        summaries.append(summary)

    write_csv(out_dir / 'qwen_method_cost_summary.csv', summaries)
    write_json(out_dir / 'qwen_method_cost_summary.json', summaries)
    write_csv(out_dir / 'qwen_stage_token_breakdown.csv', stage_rows)
    write_json(out_dir / 'qwen_stage_token_breakdown.json', stage_rows)

    chart_grouped_metrics(summaries, out_dir / 'qwen_performance_by_method.png')
    chart_avg_tokens(summaries, out_dir / 'qwen_avg_tokens_per_question.png')
    chart_f1_vs_tokens(summaries, out_dir / 'qwen_table_f1_vs_avg_tokens.png')
    chart_row_jaccard_vs_tokens(summaries, out_dir / 'qwen_row_jaccard_vs_avg_tokens.png')
    chart_f1_per_1k(summaries, out_dir / 'qwen_table_f1_per_1k_tokens.png')
    chart_stage_breakdown(stage_rows, out_dir / 'qwen_stage_token_breakdown.png')
    if args.input_cost_per_million_tokens >= 0 and args.output_cost_per_million_tokens >= 0:
        chart_cost(summaries, out_dir / 'qwen_avg_cost_per_question.png')

    readme = out_dir / 'README.txt'
    readme.write_text(
        'Qwen cross-method cost analysis\n\n'
        'Methods included:\n'
        '- Prompting Zero-Shot (Qwen)\n'
        '- Prompting Few-Shot (Qwen)\n'
        '- Prompting CoT (Qwen)\n'
        '- SCoPE (Qwen planner / Qwen executor)\n'
        '- BlendSQL (Qwen)\n'
        '- EhrAgent (Qwen, strict predicted-only evaluation)\n\n'
        'Outputs:\n'
        '- qwen_method_cost_summary.csv/json\n'
        '- qwen_stage_token_breakdown.csv/json\n'
        '- PNG charts for performance, tokens, Table F1 vs tokens, Row Jaccard vs tokens, efficiency, and stage breakdown\n',
        encoding='utf-8'
    )
    print(out_dir)


if __name__ == '__main__':
    main()
