"""No point adjustment; normal-only deploy thresholds; test ROC diagnostics flagged."""
import numpy as np
from sklearn.metrics import (roc_auc_score,average_precision_score,precision_recall_curve,
    roc_curve,auc,confusion_matrix,matthews_corrcoef)


def groups(rows):
    out={}
    for i,r in enumerate(rows):out.setdefault((r['split'],r['clip']),[]).append(i)
    return [np.array(v,dtype=int) for v in out.values()]


def ewma(score,rows,alpha=.4):
    value=np.empty_like(np.asarray(score,dtype=float))
    for indices in groups(rows):
        current=None
        for i in indices:
            current=float(score[i]) if current is None else alpha*float(score[i])+(1-alpha)*current
            value[i]=current
    return value


class Alarm:
    def __init__(self,confirm=3,release=2):self.confirm=confirm;self.release=release;self.reset()
    def reset(self):self.high=0;self.low=0;self.active=False
    def step(self,above):
        if above:self.high+=1;self.low=0
        else:self.low+=1;self.high=0
        if not self.active and self.high>=self.confirm:self.active=True
        if self.active and self.low>=self.release:self.active=False
        return self.active


def persistent_alarm(score,threshold,rows):
    output=np.zeros(len(score),dtype=bool)
    for indices in groups(rows):
        monitor=Alarm()
        for i in indices:output[i]=monitor.step(score[i]>threshold)
    return output


def spans(mask):
    value=np.asarray(mask,dtype=bool)
    edges=np.diff(np.r_[False,value,False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)-1))


def division(a,b):return float(a/b) if b else 0.


def classification(y,pred):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    precision=division(tp,tp+fp);recall=division(tp,tp+fn);specificity=division(tn,tn+fp)
    return {'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'precision':precision,
        'recall':recall,'f1':division(2*precision*recall,precision+recall),'specificity':specificity,
        'normal_fpr':division(fp,fp+tn),'accuracy':division(tp+tn,len(y)),
        'balanced_accuracy':float((recall+specificity)/2),
        'mcc':float(matthews_corrcoef(y,pred)) if len(y) else None}


def ranking(y,score):
    if len(np.unique(y))<2:
        return {'auroc':None,'average_precision':None,'pr_auc_trapezoid':None,'partial_auroc_fpr10':None,
                'diagnostic_tpr_at_fpr1':None,'diagnostic_tpr_at_fpr5':None,'diagnostic_fpr_at_tpr95':None,
                'undefined_reason':'Both label classes required; not replaced by zero or fabricated.'}
    p,r,_=precision_recall_curve(y,score);fpr,tpr,_=roc_curve(y,score)
    return {'auroc':float(roc_auc_score(y,score)),'average_precision':float(average_precision_score(y,score)),
        'pr_auc_trapezoid':float(auc(r[::-1],p[::-1])),
        'partial_auroc_fpr10':float(roc_auc_score(y,score,max_fpr=.1)),
        'diagnostic_tpr_at_fpr1':float(tpr[fpr<=.01].max()),
        'diagnostic_tpr_at_fpr5':float(tpr[fpr<=.05].max()),
        'diagnostic_fpr_at_tpr95':float(fpr[tpr>=.95].min())}

def fast_auroc_ap(y,score):
    """Tie-aware binary rank summaries for bootstrap, regression-checked vs sklearn."""
    order=np.argsort(score,kind='mergesort')[::-1];s=np.asarray(score)[order]
    label=np.asarray(y,dtype=np.int64)[order];positive=int(label.sum());negative=len(label)-positive
    if not positive or not negative:return None,None
    ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]
    tp=np.cumsum(label)[ends];fp=ends+1-tp
    tpr=np.r_[0,tp]/positive;fpr=np.r_[0,fp]/negative
    area=float(np.trapezoid(tpr,fpr))
    ap=float(np.sum(np.diff(np.r_[0,tp])/positive*tp/(ends+1)))
    return area,ap


def event_metrics(y,pred,rows):
    gt=detected=alerts=true_alerts=0;delays=[];frame_delays=[];normal_only_alerts=0;normal_only_frames=0
    for indices in groups(rows):
        labels=y[indices];alarms=pred[indices];events=spans(labels);predicted=spans(alarms)
        gt+=len(events);alerts+=len(predicted)
        for start,end in events:
            match=np.flatnonzero(alarms[start:end+1])
            if len(match):
                detected+=1;delay=int(match[0]);delays.append(delay)
                frame_delays.append(rows[int(indices[start+delay])]['frame']-rows[int(indices[start])]['frame'])
        for start,end in predicted:
            if np.any(labels[start:end+1]):true_alerts+=1
        if not np.any(labels):normal_only_alerts+=len(predicted);normal_only_frames+=len(indices)
    precision=division(true_alerts,alerts);recall=division(detected,gt)
    return {'gt_events_sampled':gt,'detected_gt_events':detected,'predicted_alert_events':alerts,
        'false_alert_events':alerts-true_alerts,'event_precision_overlap':precision,'event_recall_overlap':recall,
        'event_f1_overlap':division(2*precision*recall,precision+recall),
        'median_detected_delay_observations':float(np.median(delays)) if delays else None,
        'median_detected_delay_original_frames':float(np.median(frame_delays)) if frame_delays else None,
        'false_alerts_per_1000_normal_only_video_observations':1000*division(normal_only_alerts,normal_only_frames) if normal_only_frames else None,
        'normal_only_video_observations':normal_only_frames,
        'event_warning':'Sampled-label overlap, no point adjustment; delays among detected events only; no original-FPS seconds.'}


def all_metrics(y,score,threshold,rows):
    plain=classification(y,score>threshold)
    persistent=persistent_alarm(score,threshold,rows)
    filtered=classification(y,persistent)
    video_y=[];video_score=[]
    for indices in groups(rows):
        video_y.append(int(np.max(y[indices])))
        values=np.asarray(score)[indices];n=max(1,int(np.ceil(len(values)*.05)))
        video_score.append(float(np.partition(values,-n)[-n:].mean()))
    video_rank=ranking(np.array(video_y),np.array(video_score))
    result={**ranking(y,score),**{f'frame_{k}':v for k,v in plain.items()},
        **{f'alarm_{k}':v for k,v in filtered.items()},**event_metrics(y,persistent,rows),
        'video_auroc_top5mean':video_rank['auroc'],'video_ap_top5mean':video_rank['average_precision'],
        'video_metric_reason':video_rank.get('undefined_reason'),'threshold_normal_q99':float(threshold),
        'n_test':len(y),'n_test_anomaly':int(y.sum()),'test_diagnostic_thresholds_not_used_for_deployment':True}
    return result
