import json

nb = json.load(open('e:/programming/SIC/notebooks/02_model_baseline_testing.ipynb'))
cells = nb['cells']
print(f'Total cells: {len(cells)}')

target_id = 'd2f18379'
for i, c in enumerate(cells):
    if c.get('id') == target_id:
        print(f'Cell {i}: {c["cell_type"]} - {c.get("id", "no-id")}')
