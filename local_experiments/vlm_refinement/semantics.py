"""Direct VLM evidence remains direct; gap-aware scoring/abstention are explicit."""
import numpy as np


def bucket(gap):
    return int(np.searchsorted([16,64,128,256],max(1,gap),side='right'))


def trusted_phase(state):
    return state['phase'] if state and state['phase']>=0 and state['confidence']!='low' else -1


def direct_arrays(data,variant):
    source=data['sparse'][variant]
    by_key={(r['split'],r['clip'],r['ordinal']):s for r,s in zip(source['rows'],source['states'])}
    states=[by_key.get((r['source_split'],r['clip'],r['ordinal'])) for r in data['rows']]
    return states


class GapProcess:
    def fit(self,rows,states):
        self.counts=np.zeros((5,4,4),np.int64);previous=None;key=None
        durations={p:[] for p in range(4)};start=None;last=None
        for row,state in zip(rows,states):
            if row['split']!='train':continue
            current=row['clip']
            if current!=key:previous=None;start=None;last=None;key=current
            p=trusted_phase(state)
            if p<0:previous=None;start=None;last=None;continue
            if previous is not None:
                self.counts[bucket(row['ordinal']-last),previous,p]+=1
                if p!=previous and start is not None:
                    durations[previous].append(row['ordinal']-start)
            if previous is None or p!=previous:start=row['ordinal']
            previous=p;last=row['ordinal']
        # Support across gap buckets shrinks toward pooled normal transitions.
        pooled=self.counts.sum(0)+.5
        self.pooled=pooled/pooled.sum(1,keepdims=True)
        self.prob=(self.counts+2*self.pooled[None])/(self.counts.sum(2,keepdims=True)+2)
        self.dwell={p:float(np.quantile(v,.95)) for p,v in durations.items() if len(v)>=3}
        self.dwell_counts={p:len(v) for p,v in durations.items()}
        return self
    def score(self,rows,states):
        score=np.zeros(len(rows));dwell=np.zeros(len(rows));available=np.zeros(len(rows),bool)
        key=None;previous=None;last=None;start=None
        for i,(row,state) in enumerate(zip(rows,states)):
            current=row['split'],row['clip']
            if current!=key:previous=None;last=None;start=None;key=current
            if state is None:
                # No fresh VLM evidence here. Do not label the unseen instant normal.
                continue
            available[i]=state['consistency']!='unobservable'
            p=trusted_phase(state)
            direct=len(state['violations']) if state['consistency']=='violation' else 0.
            if p<0:direct*=.5
            transition=0.
            if p>=0 and previous is not None:
                transition=-np.log(self.prob[bucket(row['ordinal']-last),previous,p])
            score[i]=direct+.25*transition
            if p>=0:
                if p!=previous or start is None:start=row['ordinal']
                bound=self.dwell.get(p)
                if bound is not None:dwell[i]=max(0.,(row['ordinal']-start)/max(bound,1)-1)
                previous=p;last=row['ordinal']
            else:previous=None;last=None;start=None
        return score,dwell,available
    def meta(self):
        return {'gap_edges_original_frames':[16,64,128,256],'counts':self.counts.tolist(),
                'transition_probabilities':self.prob.tolist(),'dwell_original_frames_q95':self.dwell,
                'dwell_normal_observations':self.dwell_counts,'dwell_human_verified':False,
                'cadence':'binned sparse VLM evidence;not fully dense semantic inference'}
