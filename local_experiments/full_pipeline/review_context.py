"""Conservative normal-contact-sheet corrections; NOT human-certified annotations.

The initial generator mistook connector colors for process order and omitted
object removal. Preserve it and derive a smaller, visually grounded proposal.
No test recordings/labels are used in these corrections.
"""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]/'cache'/'full_pipeline_v1'

SPECS = {
 'R01': {
  'objects':['connector block','conveyor belt'],
  'phases':[
   ('connector at left','A small connector block is on the left part of a green conveyor belt.',['connector block','conveyor belt']),
   ('connector at middle','A small connector block is on the middle part of a green conveyor belt.',['connector block','conveyor belt']),
   ('connector at right','A small connector block is on the right part of a green conveyor belt.',['connector block','conveyor belt'])],
  'labels':[0,0,0,0,1,1,1,1,2,2,2,2], 'order':[0,1,2], 'edges':[(0,1),(1,2)], 'cyclic':False,
  'note':'Object color is NOT a required production sequence. Position-based stages only; empty/partial clips are not declared incomplete cycles.'},
 'R02': {
  'objects':['multimeter','scissor lift platform'],
  'phases':[
   ('empty low platform','An empty scissor lift platform is near its low position.',['scissor lift platform']),
   ('loaded low platform','A multimeter rests on the scissor lift platform near its low position.',['multimeter','scissor lift platform']),
   ('loaded raised platform','A multimeter rests on the elevated scissor lift platform.',['multimeter','scissor lift platform']),
   ('empty raised platform','The scissor lift platform is elevated with no multimeter on it.',['scissor lift platform']),
   ('empty intermediate platform','The empty scissor lift platform is at an intermediate height.',['scissor lift platform'])],
  'labels':[0,1,2,2,2,2,3,3,3,4,0,0], 'order':[0,1,2,3,4],
  'edges':[(0,1),(1,2),(2,3),(3,4),(4,0)], 'cyclic':True,
  'note':'Object removal visibly precedes platform lowering. Intermediate heights overlap; sampled labels are approximate.'},
 'R03': {
  'objects':['yellow forklift','black pallet','white rolls'],
  'phases':[
   ('separate ground load','A yellow toy forklift is separate from a black pallet carrying white rolls on the floor.',['yellow forklift','black pallet','white rolls']),
   ('forks at ground load','The yellow toy forklift forks are close to a black pallet carrying white rolls at ground level.',['yellow forklift','black pallet','white rolls']),
   ('raised load','White rolls on a black pallet are raised on the yellow toy forklift forks.',['yellow forklift','black pallet','white rolls']),
   ('lowered load near forklift','White rolls and a black pallet rest on the floor close to the yellow toy forklift.',['yellow forklift','black pallet','white rolls'])],
  'labels':[0,0,0,0,1,1,2,2,2,3,3,0], 'order':[0,1,2,3],
  'edges':[(0,1),(1,2),(2,3),(3,0)], 'cyclic':True,
  'note':'This is a toy forklift scene, not an actual factory deployment. Approach/withdrawal may look alike in still images.'},
 'R04': {
  'objects':['metal blade','cardboard sheet'],
  'phases':[
   ('empty cutter','The metal cutter has its blade raised and no cardboard in the cutting area.',['metal blade']),
   ('cardboard with blade raised','A cardboard sheet passes through a cutter whose metal blade is raised.',['metal blade','cardboard sheet']),
   ('cardboard with blade down','The metal cutter blade is down against the cardboard sheet.',['metal blade','cardboard sheet']),
   ('cut pieces accumulated','Several cut cardboard pieces are accumulated near the cutter while the blade is raised.',['metal blade','cardboard sheet'])],
  'labels':[0,0,1,2,2,1,2,2,1,2,1,3], 'order':[1,2],
  'edges':[(0,1),(1,2),(2,1),(2,3),(3,1),(3,2),(1,3)], 'cyclic':True,
  'note':'Repeated raised/down blade cycles are used; pressing vs cutting cannot reliably be separate still-image stages. Exact blade timing varies across recordings.'}
}


def main():
    for scene,spec in SPECS.items():
        p=ROOT/scene/'grammar.json'
        g=json.loads(p.read_text(encoding='utf-8'))
        if g.get('visual_review_version')==1: continue
        original=ROOT/scene/'generator_original.json'
        if not original.exists(): original.write_bytes(p.read_bytes())
        g['objects']=spec['objects']
        g['phases']=[{'id':i,'name':n,'visual_description':d,'expected_objects':o}
                     for i,(n,d,o) in enumerate(spec['phases'])]
        g['evidence_labels']=[]
        for clip in g['normal_training_clips']:
            ev=sorted([e for e in g['evidence'] if e['clip']==clip],key=lambda e:e['ordinal'])
            for i,e in enumerate(ev):
                g['evidence_labels'].append({'clip':clip,'ordinal':e['ordinal'],
                                            'phase':spec['labels'][i],'visually_confident':True})
        g['required_phase_order']=spec['order']
        g['allowed_transitions']=[{'from_phase':a,'to_phase':b} for a,b in spec['edges']]
        g['cyclic_process']=spec['cyclic']; g['observable_process_order']=True
        g['human_verified']=False; g['visual_review_version']=1
        g['review_notes']=spec['note']
        g['uncertainties']=[spec['note'],'Sparse keyframe annotations are approximate weak labels, not verified phase ground truth.',
                            'No timestamp/FPS metadata: durations are in sampled observations, not seconds.']
        g['status']='AI contact-sheet-reviewed conservative proposal; human confirmation still required'
        p.write_text(json.dumps(g,ensure_ascii=False,indent=2),encoding='utf-8')
        (p.parent/'grammar.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest(),encoding='ascii')
        print(scene, 'conservative normal-evidence grammar saved')


if __name__=='__main__': main()
