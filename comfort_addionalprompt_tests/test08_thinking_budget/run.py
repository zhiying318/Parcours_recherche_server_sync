"""Replay test07 prompts with soft thinking budgets and exact generated-ID metrics."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Length-only ablations, ordered from minimal to detailed. No task-specific advice.
PROMPTS = {
    # 'baseline': '',
    'shortest': (
        'Keep your thinking as short as possible while maintaining a correct answer.'
    ),
    'approx': (
        'Keep your thinking to approximately {budget} tokens.'
    ),
    'cap': (
        'Use at most {budget} tokens for your thinking.'
    ),
    'explicit_end': (
        'Use at most {budget} tokens for your thinking. '
        'End the thinking section with </think> within this token limit.'
    ),
}


def build_prompt(original, variant, budget):
    note = PROMPTS[variant].format(budget=budget)
    if not note:
        return original
    marker = 'Choose ONE option and respond with ONLY the letter.'
    if original.count(marker) != 1:
        raise ValueError('Expected exactly one original MCQ output-format instruction')
    prefix, suffix = original.split(marker, 1)
    return prefix + 'Thinking length constraint:\n' + note + '\n\n' + marker + suffix


def token_metrics(ids, tokenizer, cap):
    # The opening <think> normally belongs to the input chat template.
    close = tokenizer.encode('</think>', add_special_tokens=False)
    if not close:
        raise ValueError('Tokenizer has no closing thinking marker')
    end = next((i for i in range(len(ids) - len(close) + 1)
                if ids[i:i + len(close)] == close), None)
    opening = tokenizer.encode('<think>', add_special_tokens=False)
    start = len(opening) if opening and ids[:len(opening)] == opening else 0
    eos = tokenizer.eos_token_id
    ended = bool(ids and ids[-1] == eos)
    return dict(thinking_tokens=(end if end is not None else len(ids)) - start,
                generated_tokens=len(ids), thinking_closed=end is not None,
                hit_generation_cap=len(ids) >= cap and not ended,
                generated_token_ids=ids)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source-csv', type=Path, default=HERE.parent / 'test07_camera_geometry_before_question/results_preciseprompt/mcq_long_qwen3_5vl_thinking.csv')
    p.add_argument('--model-id', default='Qwen/Qwen3.5-9B')
    p.add_argument('--variants', nargs='+', choices=list(PROMPTS), default=list(PROMPTS))
    p.add_argument('--budget', type=int, default=1024)
    p.add_argument('--max-new-tokens', type=int, default=4096)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--output-dir', type=Path, default=HERE / 'results')
    p.add_argument('--device-map', default='cuda:0')
    p.add_argument('--attn-implementation', default='flash_attention_2')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.budget <= 0 or args.max_new_tokens <= args.budget or args.limit < 0:
        p.error('Require budget > 0, max-new-tokens > budget, limit >= 0')
    rows = list(csv.DictReader(args.source_csv.open(encoding='utf-8', newline='')))
    random.Random(args.seed).shuffle(rows)  # Avoid a relation/view-biased prefix in smoke runs.
    if args.limit:
        rows = rows[:args.limit]
    if not rows or len({r['image_path'] for r in rows}) != len(rows):
        raise ValueError('Source must have nonempty, unique image rows')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ('dry_run', 'output_dir')}
    config['source_sha256'] = hashlib.sha256(args.source_csv.read_bytes()).hexdigest()
    config['prompts'] = PROMPTS
    config['prompt_layout'] = 'length_before_original_format_and_options_v2'
    manifest = args.output_dir / 'config.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Configuration changed: choose a different --output-dir')
    manifest.write_text(json.dumps(config, indent=2) + '\n')
    preview = {v: build_prompt(rows[0]['mcq_prompt'], v, args.budget) for v in args.variants}
    (args.output_dir / 'prompt_preview.json').write_text(json.dumps(preview, indent=2) + '\n')
    if args.dry_run:
        print(json.dumps({'samples': len(rows), 'variants': args.variants, 'preview': str(args.output_dir / 'prompt_preview.json')}, indent=2))
        return
    missing = [r['image_path'] for r in rows if not Path(r['image_path']).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} images missing, e.g. {missing[:3]}')
    from transformers import set_seed
    from spatial_eval.backends.qwen3_5vl import Qwen35VLThinkingBackend
    from spatial_eval.prompts.MCQ import _normalize_choice_thinking

    class MeasuredBackend(Qwen35VLThinkingBackend):
        def _decode(self, inputs, output):
            ids = output[0, inputs['input_ids'].shape[1]:].tolist()
            self.metrics = token_metrics(ids, self.processor.tokenizer, args.max_new_tokens)
            # Preserve thinking markers even if the tokenizer treats them as special tokens.
            return self.processor.tokenizer.decode(ids, skip_special_tokens=False).strip()

    backend = MeasuredBackend(model_id=args.model_id, device_map=args.device_map,
                              attn_implementation=args.attn_implementation)
    summaries = []
    for variant in args.variants:
        output = args.output_dir / f'{variant}.jsonl'
        records = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
        done = {r['image_path'] for r in records}
        with output.open('a', encoding='utf-8') as handle:
            for row in rows:
                if row['image_path'] in done:
                    continue
                sample_seed = (args.seed + int(hashlib.sha256(row['image_path'].encode()).hexdigest()[:8], 16)) % 2**32
                set_seed(sample_seed)  # Stable across variants and resume.
                prompt = build_prompt(row['mcq_prompt'], variant, args.budget)
                raw = backend.ask(row['image_path'], prompt, args.max_new_tokens)
                pred = _normalize_choice_thinking(raw)
                record = dict(row, mcq_prompt=prompt, model_answer=raw, pred_letter=pred,
                              correct=bool(pred and pred == row['correct_letter']),
                              **backend.metrics)
                record['complete'] = record['thinking_closed'] and bool(pred) and not record['hit_generation_cap']
                record['within_budget'] = record['thinking_tokens'] <= args.budget
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush()
                records.append(record)
                print(f'{variant} {len(records)}/{len(rows)}: correct={record["correct"]}, thinking={record["thinking_tokens"]}', flush=True)
        lengths = sorted(r['thinking_tokens'] for r in records)
        n = len(records)
        summary = dict(variant=variant, n=n, accuracy=sum(r['correct'] for r in records)/n,
                       complete_rate=sum(r['complete'] for r in records)/n,
                       within_budget_rate=sum(r['within_budget'] for r in records)/n,
                       cap_rate=sum(r['hit_generation_cap'] for r in records)/n,
                       mean_thinking_tokens=sum(lengths)/n,
                       p50=lengths[(n-1)//2], p90=lengths[min(n-1, (9*n+9)//10-1)],
                       usable_rate=sum(r['correct'] and r['complete'] and r['within_budget'] for r in records)/n)
        summaries.append(summary)
        (args.output_dir / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
