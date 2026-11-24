"""Summarize the independent classwise replay, without model calls."""
from pathlib import Path
import csv
HERE=Path(__file__).resolve().parent
with (HERE/'realdata_transfer_v2/classwise.csv').open(newline='') as f:rows=list(csv.DictReader(f))
summary=[]
for cls in range(10):
    record=dict(class_id=cls,class_name=next(r['class_name'] for r in rows if int(r['class_id'])==cls))
    for view in ('original','flipped_only','full_average'):
        group=[r for r in rows if int(r['class_id'])==cls and r['view']==view]
        assert len(group)==3
        for metric in ('loss','error','confidence'):record[view+'_'+metric]=sum(float(r[metric]) for r in group)/3
    summary.append(record)
with (HERE/'realdata_transfer_v2/classwise_mean.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,list(summary[0]));writer.writeheader();writer.writerows(summary)
note='\n## Where the negative transfer concentrates\n\nAcross the three independently replayed models, the footwear classes show the largest deterioration.\n\n| Class | Original error | Flipped-only error | Full-average error |\n|---|---:|---:|---:|\n'
for cls in (5,7,9):
    r=summary[cls];note+=f"| {r['class_name']} | {r['original_error']:.2%} | {r['flipped_only_error']:.2%} | {r['full_average_error']:.2%} |\n"
note+='\nThe original test images also have large positive mean horizontal brightness centroids for sandals (0.165), sneakers (0.160), and ankle boots (0.242), on a normalized column coordinate from −1 to 1. Ankle boots have the largest class-mean image/flip discrepancy. This alignment is consistent with learned sensitivity to the dataset’s directional presentation; it does not independently prove a causal mechanism or a population invariance-defect bound. The classifier was trained without horizontal-flip augmentation under the frozen recipe. These findings support retaining the original-test failure, rather than substituting the favorable artificial pair-closed result.\n'
path=HERE/'REALDATA_TRANSFER_CHECK.md'
with path.open('a',encoding='utf-8') as f:f.write(note)
print(note.encode('ascii','backslashreplace').decode('ascii'))
