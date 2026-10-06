"""Offline contracts. No live API calls or fabricated evaluation results."""
import copy
import json
from pathlib import Path
import tempfile
import uuid
import subprocess
import sys
import unittest
from unittest.mock import patch
from PIL import Image
import numpy as np
from .semantics import schema, validate, history_entry, assess
from .api import ROOT, digest, reserve, settle, write
from .data import prepare
from ..full_pipeline.features import MAX_OBJECTS
from .model import phase, Process, fit_process, crop_features, aggregate, calibration


def state(p=0,confidence='high',consistency='consistent',violations=None):
    return {'phase':p,'confidence':confidence,'objects':[], 'consistency':consistency,
            'violations':violations or [],'reason':'visible evidence'}


class Contracts(unittest.TestCase):
    def test_standalone_entrypoints_import_without_features_first(self):
        subprocess.run([sys.executable,'-c',
            'import local_experiments.mentor_pipeline.infer; import local_experiments.mentor_pipeline.compare_runs'],
            cwd=ROOT.parent,check=True,capture_output=True)
    def test_no_mlp_phase_proxy(self):
        self.assertEqual(phase(state(1)),1)
        self.assertEqual(phase(state(1,'low')),-1)
        self.assertEqual(phase(state(-1)),-1)
    def test_direct_pixel_inputs_include_current_crops_and_only_past_context(self):
        folder=ROOT/'runs'/('mentor-input-unit-'+uuid.uuid4().hex);folder.mkdir(parents=True)
        g={'objects':['part'],'phases':[{'id':0}], 'required_phase_order':[0],
           'allowed_transitions':[],'observable_process_order':True,'cyclic_process':False}
        current=Image.new('RGB',(64,64),'red');past=Image.new('RGB',(64,64),'blue')
        objects=[{'id':1000001,'class':'part','class_id':0,'confidence':.9,
                  'bbox':[0.,0.,32.,32.],'geometry':[1.,.5,.5,.5,.5,0.,0.]}]
        captured={}
        def fake(content,sch,evidence,destination,paid,**kwargs):
            captured.update(content=content,schema=sch,evidence=evidence,paid=paid)
            write(destination/'request_meta.json',{'request_sha256':'offline-contract-test'})
            return {**state(),'objects':[{'id':1000001,'visible':True,'state':'part visible','confidence':'high'}]}
        try:
            with patch('local_experiments.mentor_pipeline.semantics.request',fake):
                result=assess(g,current,objects,[history_entry(state())],[past],folder,paid=False)
            self.assertEqual(result['phase'],0);self.assertFalse(captured['paid'])
            labels=[x['label'] for x in captured['evidence']]
            self.assertEqual(len(labels),3)
            self.assertTrue(labels[0].startswith('PAST'))
            self.assertTrue(labels[1].startswith('CURRENT full'))
            self.assertTrue(labels[2].startswith('CURRENT crop id=1000001'))
            self.assertEqual(sum(x['type']=='input_image' for x in captured['content']),3)
            self.assertFalse(json.loads((folder/'accepted.json').read_text())['mlp_phase_used'])
        finally:
            for p in folder.iterdir():
                if p.name not in ['accepted.json','request_meta.json']:raise ValueError('Unexpected unit artifact')
                p.unlink()
            folder.rmdir()
    def test_schema_is_closed_and_current_ids_only(self):
        g={'phases':[{'id':0},{'id':1}]};s=schema(g,[1001,2001])
        self.assertFalse(s['additionalProperties'])
        self.assertEqual(s['properties']['objects']['items']['properties']['id']['enum'],[1001,2001])
        validate(state(),g,[])
        with self.assertRaises(ValueError):validate({**state(),'objects':[{'id':999,'state':'x'}]},g,[])
    def test_contradictory_semantics_rejected(self):
        g={'phases':[{'id':0}]}
        with self.assertRaises(ValueError):validate(state(violations=['wrong_order']),g,[])
        with self.assertRaises(ValueError):validate(state(consistency='violation'),g,[])
    def test_process_fit_uses_only_normal_fit(self):
        rows=[{'split':'train','clip':'01'},{'split':'train','clip':'01'},
              {'split':'val','clip':'02'},{'split':'test','clip':'03'}]
        fitted=fit_process(rows,[state(0),state(1),state(1),state(0)],np.array([True,True,False,False]),2)
        self.assertEqual(fitted['transition_counts'],[[0,1],[0,0]])
        self.assertEqual(fitted['transition_observations'],1)
    def test_recording_reset_and_unknown_break_transition(self):
        m=Process({'transition_probabilities':[[.9,.1],[.2,.8]],'transition_observations':2})
        _,a=m.step(state(0));self.assertFalse(a['transition_available'])
        _,b=m.step(state(1));self.assertTrue(b['transition_available'])
        _,c=m.step(state(-1,'low','unobservable'));self.assertFalse(c['certified_normal'])
        _,d=m.step(state(0));self.assertFalse(d['transition_available'])
        m.reset();_,e=m.step(state(1));self.assertFalse(e['transition_available'])
    def test_direct_vlm_violation_changes_process_score(self):
        m=Process({'transition_probabilities':[[1.]],'transition_observations':0})
        clean,_=m.step(state());bad,_=m.step(state(consistency='violation',violations=['wrong_order']))
        self.assertGreater(bad,clean)
    def test_object_centric_features_keep_individual_crops(self):
        crops=np.zeros((MAX_OBJECTS,512),np.float32);geo=np.zeros((MAX_OBJECTS,7),np.float32)
        crops[0,0]=1.;crops[1,0]=2.
        x=crop_features(crops,geo);self.assertEqual(x.shape,(MAX_OBJECTS,518));self.assertNotEqual(x[0,0],x[1,0])
    def test_fixed_fusion(self):
        raw={k:np.array([1.,2.]) for k in ['frame_global','frame_phase','object_global','object_phase','process']}
        scales={k:{'center':0.,'scale':1.} for k in raw}
        self.assertTrue(np.allclose(aggregate(raw,scales)['mentor_full'],[1.,2.]))
    def test_calibration_refuses_nonfinite(self):
        with self.assertRaises(ValueError):calibration([np.nan])
        self.assertGreater(calibration([0.,0.])['scale'],0.)
    def test_budget_retains_failure_and_settles_only_success(self):
        tmp=ROOT/'runs'/('mentor-unit-'+uuid.uuid4().hex)
        tmp.mkdir(parents=True)
        self.assertTrue(tmp.resolve().is_relative_to((ROOT/'runs').resolve()))
        try:
            target=tmp/'ledger.json';write(target,{'limit_usd':4.,'reserved_usd':3.9,'attempts':0,
                                                       'measured_estimate_usd':0.,'successful_calls':0})
            with patch('local_experiments.mentor_pipeline.api.LEDGER',target):
                reserve(.02,'paid');settle(.02,.001,'paid')
                value=json.loads(target.read_text());self.assertAlmostEqual(value['reserved_usd'],3.901)
                reserve(.05,'unresolved')
                with self.assertRaises(RuntimeError):reserve(.1,'forbidden')
                self.assertIn('unresolved',json.loads(target.read_text())['mentor_pending'])
        finally:
            for file in tmp.iterdir():
                if file.name not in ['ledger.json','ledger.mentor.lock']:raise ValueError('Unexpected test file, preserve it')
                file.unlink()
            tmp.rmdir()


if __name__=='__main__':unittest.main()
