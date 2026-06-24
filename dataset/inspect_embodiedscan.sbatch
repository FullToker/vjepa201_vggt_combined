#!/bin/bash
#SBATCH --job-name=inspect_escan
#SBATCH --partition=geo
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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
vis_ids = s['images'][0].get('visible_instance_ids', [])
bbox_ids = [i['bbox_id'] for i in s['instances'][:5]]
labels   = [i['bbox_label_3d'] for i in s['instances'][:5]]
print('visible_instance_ids[:10]:', vis_ids[:10])
print('instance bbox_ids[:5]:', bbox_ids)
print('instance bbox_label_3d[:5]:', labels)
print('type of vis_ids[0]:', type(vis_ids[0]) if vis_ids else 'empty')
PYEOF
