import os

# exp_ids = [10, 11, 12, 13, 14, 15, 16, 17]
# exp_ids = [401, 402, 411, 421, 431, 441]
exp_ids = [31, 51] # multi steps ahead

for exp_id in exp_ids:
    os.system(f'sbatch pred_gl_job.sh {exp_id}')
