"""Causal, explicit process checks; candidates, not ground-truth semantics."""
import numpy as np


class ProcessMonitor:
    def __init__(self, grammar, dwell_limits=None, expected=None, confirm=2):
        self.g = grammar
        self.enabled = grammar.get('observable_process_order',False)
        self.order = grammar.get('required_phase_order',[])
        self.edges = {(e['from_phase'],e['to_phase']) for e in grammar['allowed_transitions']}
        self.limits = dwell_limits or {}
        self.expected = expected or {}
        self.confirm = confirm
        self.reset()

    def reset(self):
        self.current = None; self.pending = None; self.pending_n = 0; self.dwell = 0
        self.missing = {}; self.anchored = False; self.visited = set(); self.hold = []

    def step(self, phase, present):
        reasons = []
        # Unknown predictions do not fabricate process transitions or missing objects.
        if phase < 0 or not self.enabled: return 0., ['uncertain_phase'] if phase < 0 else [], self.current
        if phase == self.pending: self.pending_n += 1
        else: self.pending = phase; self.pending_n = 1
        if self.pending_n < self.confirm: return 0., [], self.current
        previous = self.current
        if previous != phase:
            self.dwell = 0; self.missing = {}
            if previous is not None:
                if (previous,phase) not in self.edges:
                    reasons.append(f'illegal_order:{previous}->{phase}')
                if previous in self.order and phase in self.order:
                    a,b = self.order.index(previous),self.order.index(phase)
                    wrap = self.g.get('cyclic_process',False) and a==len(self.order)-1 and b==0
                    if b>a+1:
                        reasons.append('skipped_required_phase:'+','.join(map(str,self.order[a+1:b])))
                    elif b<a and not wrap:
                        reasons.append(f'reversed_required_order:{previous}->{phase}')
                    if wrap: self.visited = set(); self.anchored = True
            if self.order and phase==self.order[0]: self.anchored=True
            self.current = phase; self.visited.add(phase)
        self.dwell += 1
        if self.dwell > self.limits.get(str(phase),float('inf')):
            reasons.append(f'phase_too_long:{phase}')
        for c in self.expected.get(str(phase),[]):
            self.missing[c] = 0 if c in present else self.missing.get(c,0)+1
            if self.missing[c] >= 3: reasons.append(f'missing_expected_object:{c}')
        self.hold = [(r,t-1) for r,t in self.hold if t>1]
        for r in reasons:
            if not any(a==r for a,_ in self.hold): self.hold.append((r,5))
        visible = list(dict.fromkeys(r for r,_ in self.hold))
        return float(len(visible)), visible, self.current

    def finish(self, require_complete_cycle=False):
        # Arbitrarily cropped benchmark videos are NOT guaranteed complete cycles.
        if require_complete_cycle and self.anchored:
            absent = set(self.order)-self.visited
            if absent: return ['incomplete_cycle:'+','.join(map(str,sorted(absent)))]
        return []


def causal_probabilities(prob, rows, smooth=3):
    result = np.empty_like(prob)
    key=None; history=[]
    for i,r in enumerate(rows):
        k=(r['split'],r['clip'])
        if k!=key: key=k; history=[]
        history.append(prob[i]); history=history[-smooth:]
        result[i]=np.mean(history,axis=0)
    return result
