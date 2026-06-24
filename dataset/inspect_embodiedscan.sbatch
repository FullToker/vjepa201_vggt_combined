#!/bin/bash
#SBATCH --job-name=inspect_escan
#SBATCH --partition=geo
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/inspect_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/inspect_%j.err

srun singularity exec \
    --bind /glob/g01-cache:/glob/g01-cache \
    /glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif \
    python3 - << 'PYEOF'
import pickle

pkl = '/glob/g01-cache/pf/Yushuo/vjepa201/source_data/embodiedscan/embodiedscan/embodiedscan_infos_train.pkl'
data = pickle.load(open(pkl, 'rb'))
s = data['data_list'][0]
print('sample keys:', list(s.keys()))
if s.get('images'):
    print('images[0] keys:', list(s['images'][0].keys()))
else:
    print('no images key')
for k, v in s.items():
    print(k, ':', str(v)[:200])
PYEOF
