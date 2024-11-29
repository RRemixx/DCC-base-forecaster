import os
from tqdm import tqdm

exp_ids = [303, 304, 310, 311, 312, 313]

for exp_id in tqdm(exp_ids):
    os.system(f'python eval_cp.py -m=e2ecp -i={exp_id} -c -u')