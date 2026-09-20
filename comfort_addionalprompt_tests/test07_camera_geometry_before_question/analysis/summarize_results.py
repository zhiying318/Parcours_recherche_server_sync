"""Audit saved CSVs without changing predictions or rerunning evaluations."""
import ast
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
ROOT = TEST.parent.parent
SOURCE = TEST / 'results_preciseprompt'
# Reuse the current parser without importing model/GPU dependencies.
tree = ast.parse((ROOT / 'spatial_eval/prompts/MCQ.py').read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_normalize_choice_thinking')
ns = {'re': re}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<parser>', 'exec'), ns)
parse = ns['_normalize_choice_thinking']
expected = {str(Path(p)) for p in json.loads((TEST / 'data/image_paths.json').read_text())}
summary, details, missing = [], [], []
for path in sorted(SOURCE.glob('*.csv')):
    rows = list(csv.DictReader(path.open()))
    seen = {str(Path(r['image_path'])) for r in rows}
    assert len(seen) == len(rows), f'Duplicate image rows: {path}'
    assert seen <= expected, f'Unexpected image paths: {path}'
    statuses, corrected = Counter(), []
    for row in rows:
        answer, pred = row['model_answer'].strip(), row['pred_letter']
        recovered, status = pred, 'non_thinking'
        if 'thinking' in path.name:
            if 'gpt_5' in path.name:
                status = 'api_final_answer_chain_not_observable'
            elif 'gemma4' in path.name:
                match = re.search(r'(?<![A-Za-z])([A-D])\s*$', answer)
                recovered = match.group(1) if match else ''
                status = 'apparent_final_answer_boundary_removed' if recovered else 'unverifiable'
            elif '</think>' not in answer:
                status = 'no_end_marker_suspected_truncation'
                recovered = ''
            else:
                recovered = parse(answer)
                if not recovered:
                    boxed = re.findall(r'\\boxed\{\s*([A-D])\s*\}', answer.rsplit('</think>', 1)[-1])
                    recovered = boxed[-1] if boxed else ''
                status = 'chain_closed_with_answer' if recovered else 'chain_closed_without_final_choice'
        statuses[status] += 1
        corrected.append(recovered == row['correct_letter'])
        details.append(dict(file=path.name, image_path=row['image_path'], correct_relation=row['correct_relation'], correct_letter=row['correct_letter'], stored_prediction=pred, audited_prediction=recovered, stored_correct=pred == row['correct_letter'], audited_correct=corrected[-1], completion_status=status, answer_characters=len(answer)))
    n, good = len(rows), sum(r['pred_letter'] == r['correct_letter'] for r in rows)
    summary.append(dict(file=path.name, rows=n, expected=len(expected), missing=len(expected-seen), correct=good, accuracy=good/n if n else None, valid_predictions=sum(r['pred_letter'] in 'ABCD' and len(r['pred_letter'])==1 for r in rows), audited_correct=sum(corrected), audited_accuracy=sum(corrected)/n if n else None, completion=dict(statuses), relations=dict(Counter(r['correct_relation'] for r in rows))))
    missing.extend(dict(file=path.name, image_path=p) for p in sorted(expected-seen))
for name, data in [('sample_audit.csv', details), ('missing_samples.csv', missing)]:
    with (HERE/name).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
(HERE/'summary.json').write_text(json.dumps(dict(captured_at=datetime.now().astimezone().isoformat(), results=summary), ensure_ascii=False, indent=2)+'\n')
for s in summary:
    print(s['file'], f"{s['correct']}/{s['rows']}={100*s['accuracy']:.2f}%", 'audited',s['audited_correct'], 'completion',s['completion'])
