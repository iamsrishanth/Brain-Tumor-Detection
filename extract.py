import json

with open('Brain Tumor Detection.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('train.py', 'w', encoding='utf-8') as f:
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            lines = source.split('\n')
            clean_lines = [line for line in lines if not line.strip().startswith('%')]
            f.write('\n'.join(clean_lines) + '\n\n')
