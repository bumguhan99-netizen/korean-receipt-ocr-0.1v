"""Maximum-cardinality IoU matching for explicitly partial ground truth."""
def iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]))
    y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    intersection=x*y
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.0

def match(gt,pred,threshold=0.5):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    if not 0<threshold<=1:
        raise ValueError('Invalid IoU threshold')
    scores=np.array([[iou(g,p) for p in pred] for g in gt],dtype=float)
    pairs=[]
    if gt and pred:
        # One additional valid match outweighs the total possible IoU tie-break.
        cardinality_weight=min(len(gt),len(pred))+1
        objective=(scores>=threshold)*cardinality_weight+np.where(scores>=threshold,scores,0)
        rows,cols=linear_sum_assignment(-objective)
        pairs=[(int(r),int(c),float(scores[r,c])) for r,c in zip(rows,cols) if scores[r,c]>=threshold]
    matched_gt={r for r,_,_ in pairs}
    matched_pred={c for _,c,_ in pairs}
    return dict(pairs=pairs,missing_gt=[i for i in range(len(gt)) if i not in matched_gt],unscored_predictions=[i for i in range(len(pred)) if i not in matched_pred])
