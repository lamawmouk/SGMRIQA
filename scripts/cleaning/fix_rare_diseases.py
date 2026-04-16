"""
Remove rare diseases (HSE, PSP) from brain training classification questions.
Replace with common alternatives using modality-appropriate reasoning.

HSE → Intracranial hemorrhage (or Stroke if Hemorrhage already present)
PSP → Stroke (or Intracranial hemorrhage if Stroke already present)
"""

import json
import re
import sys

# ── Replacement reasoning by modality ──────────────────────────────────────

REPLACEMENTS = {
    'Intracranial hemorrhage': {
        'T1': 'there is no isointense to slightly hypointense signal suggesting hyperacute hemorrhage',
        'FLAIR': 'there is no heterogeneously hyperintense signal suggesting acute hemorrhage',
    },
    'Stroke': {
        'T1': 'there is no focal hypointensity conforming to a vascular territory',
        'FLAIR': 'there is no hyperintense signal in a vascular territory distribution',
    },
    'Cerebral atrophy': {
        'T1': 'there is no generalized cortical thinning or diffuse parenchymal volume loss',
        'FLAIR': 'there is no generalized cortical thinning or diffuse parenchymal volume loss',
    },
}

# ── HSE reasoning patterns ─────────────────────────────────────────────────

HSE_REASON_PATTERNS = [
    'because there is no asymmetric temporal lobe edema or hemorrhagic necrotizing encephalitis pattern',
    'because there is no unilateral temporal and insular cortex signal abnormality',
]

# ── PSP reasoning patterns ─────────────────────────────────────────────────

PSP_REASON_PATTERNS = [
    'because there is no midbrain atrophy with hummingbird sign or morning glory sign',
    'because the midbrain and superior cerebellar peduncles appear normal in size',
]


def _pick_replacement(question, primary, fallbacks):
    """Pick a replacement that doesn't conflict with existing options."""
    q = question
    for choice in [primary] + fallbacks:
        # Check if this option (or something very similar) is already present
        if choice not in q and choice.lower() not in q.lower():
            # Also check partial matches (e.g., "Hemorrhage" vs "Intracranial hemorrhage")
            if choice == 'Intracranial hemorrhage' and 'Hemorrhage' in q:
                continue
            return choice
    return fallbacks[-1]  # Last resort


def _get_modality(entry):
    """Get modality category (T1 or FLAIR) from entry."""
    mod = entry.get('modality', '').upper()
    if 'FLAIR' in mod:
        return 'FLAIR'
    return 'T1'  # T1, T1POST, T2, etc.


def fix_rare_diseases(qa, dry_run=False):
    """Replace HSE and PSP in classification questions."""
    count = 0
    details = []

    for e in qa:
        if e.get('task') != 'classification':
            continue

        q = e.get('question', '')
        r = e.get('reasoning', '')
        modality = _get_modality(e)
        changed = False

        new_q = q
        new_r = r

        # ── Replace HSE ───────────────────────────────────────────────
        if 'Herpes simplex encephalitis' in new_q:
            replacement = _pick_replacement(new_q, 'Intracranial hemorrhage',
                                            ['Stroke', 'Cerebral atrophy'])

            # Replace in question
            new_q = new_q.replace('Herpes simplex encephalitis (HSE)', replacement)
            new_q = new_q.replace('Herpes simplex encephalitis', replacement)

            # Replace in reasoning - find the letter
            for letter in ['A', 'B', 'C', 'D']:
                for old_reason in HSE_REASON_PATTERNS:
                    old_text = f'It is not ({letter}) Herpes simplex encephalitis {old_reason}'
                    if old_text in new_r:
                        new_reason = REPLACEMENTS[replacement][modality]
                        new_text = f'It is not ({letter}) {replacement} because {new_reason}'
                        new_r = new_r.replace(old_text, new_text)

                # Handle positive reasoning (HSE as correct answer - 1 entry)
                old_pos = f'suggestive of ({letter}) Herpes simplex encephalitis'
                if old_pos in new_r:
                    new_r = new_r.replace(old_pos, f'suggestive of ({letter}) {replacement}')
                old_pos2 = f'indicating ({letter}) Herpes simplex encephalitis'
                if old_pos2 in new_r:
                    new_r = new_r.replace(old_pos2, f'indicating ({letter}) {replacement}')

            changed = True

        # ── Replace PSP ───────────────────────────────────────────────
        if 'Progressive supranuclear palsy' in new_q:
            replacement = _pick_replacement(new_q, 'Stroke',
                                            ['Intracranial hemorrhage', 'Cerebral atrophy'])

            # Replace in question
            new_q = new_q.replace('Progressive supranuclear palsy (PSP)', replacement)
            new_q = new_q.replace('Progressive supranuclear palsy', replacement)

            # Replace in reasoning
            for letter in ['A', 'B', 'C', 'D']:
                for old_reason in PSP_REASON_PATTERNS:
                    old_text = f'It is not ({letter}) Progressive supranuclear palsy {old_reason}'
                    if old_text in new_r:
                        new_reason = REPLACEMENTS[replacement][modality]
                        new_text = f'It is not ({letter}) {replacement} because {new_reason}'
                        new_r = new_r.replace(old_text, new_text)

            changed = True

        if changed and (new_q != q or new_r != r):
            details.append({
                'volume_id': e.get('volume_id', ''),
                'slice_num': e.get('slice_num', 'vol'),
                'modality': modality,
                'old_q': q[:200],
                'new_q': new_q[:200],
                'reasoning_changed': new_r != r,
            })
            if not dry_run:
                e['question'] = new_q
                e['reasoning'] = new_r
            count += 1

    return count, details


def main():
    dry_run = '--dry-run' in sys.argv

    files = [
        ('brain_train_img', 'brain/gpt4o_brain_train_image_qa_image_qa_pairs.json'),
        ('brain_train_vol', 'brain/gpt4o_brain_train_volume_qa_volume_qa_pairs.json'),
    ]

    total = 0
    for name, path in files:
        print(f'\n{"="*60}')
        print(f'Processing: {name} ({path})')
        print(f'{"="*60}')

        with open(path) as f:
            data = json.load(f)

        count, details = fix_rare_diseases(data, dry_run=dry_run)
        total += count
        print(f'  Rare diseases replaced: {count}')

        # Show breakdown
        from collections import Counter
        # Count remaining HSE/PSP
        remaining_hse = sum(1 for e in data if 'Herpes simplex encephalitis' in e.get('question', ''))
        remaining_psp = sum(1 for e in data if 'Progressive supranuclear palsy' in e.get('question', ''))
        print(f'  Remaining HSE: {remaining_hse}')
        print(f'  Remaining PSP: {remaining_psp}')

        for d in details[:8]:
            print(f'\n    [{d["modality"]}] {d["volume_id"]} slice {d["slice_num"]} (reasoning_changed={d["reasoning_changed"]})')
            print(f'      OLD Q: {d["old_q"]}')
            print(f'      NEW Q: {d["new_q"]}')

        if not dry_run and count > 0:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f'\n  Saved {path}')
        elif dry_run:
            print(f'\n  [DRY RUN - not saved]')

    print(f'\n{"="*60}')
    print(f'TOTAL: {total} entries fixed')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
