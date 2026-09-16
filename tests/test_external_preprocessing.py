"""Regression tests for the confirmed external-validation implementation errors."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datetime import datetime,timedelta
import tempfile
import unittest
import numpy as np
import torch
from temporal_data import FeatureScaler,SequenceExample,SequenceBundle
from temporal_models import make_sequence_model
from run_temporal_experiments import train_sequence
from run_tuned_mimic_ablation import raw_sequence,make_bundle
from numeric_feature_repairs import repaired_lab_value,repair_feature_table

def example(values,observed=None,label=0):
    x=np.asarray(values,np.float32)
    return SequenceExample(x,np.arange(len(x),dtype=np.float32),np.arange(len(x),dtype=np.float32),
        np.ones_like(x) if observed is None else np.asarray(observed,np.float32),label,'train','p','v','t')

class PreprocessingTests(unittest.TestCase):
    def test_lab_aliases_units_and_distinct_assays(self):
        self.assertEqual(repaired_lab_value({'name':'全血葡萄糖','value':'7.2','unit':'mmol/L'}),('value__lab__glucose',7.2))
        self.assertAlmostEqual(repaired_lab_value({'name':'葡萄糖','value':'180.18','unit':'mg/dL'})[1],10)
        self.assertIsNone(repaired_lab_value({'name':'尿葡萄糖','value':'10','unit':'mmol/L'}))
        self.assertIsNone(repaired_lab_value({'name':'葡萄糖-6-磷酸脱氢酶','value':'10','unit':'U/L'}))
        self.assertIsNone(repaired_lab_value({'name':'纤维蛋白及纤维蛋白原降解产物','value':'10','unit':'mg/L'}))
        self.assertEqual(repaired_lab_value({'name':'纤维蛋白原','value':'300','unit':'mg/dL'}),('value__lab__fibrinogen',3))
        self.assertIsNone(repaired_lab_value({'name':'葡萄糖','value':'10','unit':None}))

    def test_repaired_lab_carry_forward_is_causal_and_masked(self):
        from temporal_data import row_key,_normalise_time
        t=datetime(2020,1,1)
        rows=[{'visit_no':'v','split':'train','round_time':t+timedelta(hours=i),'latest_lab':[
            {'name':'全血葡萄糖','value':'8','unit':'mmol/L','latest_time':t},
            {'name':'全血葡萄糖','value':'20','unit':'mmol/L','latest_time':t+timedelta(hours=3)}]} for i in range(2)]
        names=['value__lab__glucose','mask__value__lab__glucose','value__lab__fibrinogen','mask__value__lab__fibrinogen']
        table={row_key(r):np.ones(4,np.float32)*99 for r in rows}
        repair_feature_table(rows,table,names,row_key,_normalise_time)
        np.testing.assert_array_equal(table[row_key(rows[0])],[8,1,0,0])
        np.testing.assert_array_equal(table[row_key(rows[1])],[8,0,0,0])

    def test_constant_training_channel_cannot_explode_externally(self):
        s=FeatureScaler(['value__glucose','mask__value__glucose']).fit([example([[13.47,1],[13.47,0]])])
        self.assertTrue(s.constant_features[0]); self.assertEqual(s.std[0],1)
        actual=s.transform_array(np.array([[160,1]],np.float32),np.ones((1,2),np.float32))
        self.assertEqual(actual[0,0],0)
        loaded=FeatureScaler.from_state_dict(s.state_dict())
        np.testing.assert_array_equal(loaded.transform_array([[160,1]],[[1,1]]),actual)

    def test_legacy_constant_checkpoint_and_median_imputation(self):
        state={'feature_names':['value__a','value__b','mask__value__a'],
            'mean':[13.47,10,0],'std':[.0001,2,1],'median':[13.47,8,0]}
        s=FeatureScaler.from_state_dict(state)
        a=s.transform_array([[80,0,0],[80,12,0]],[[1,0,1],[1,0,1]])
        np.testing.assert_allclose(a,[[0,-1,0],[0,1,0]])

    def test_hours_and_truncated_first_event_agree(self):
        start=datetime(2020,1,1)
        events=[(start+timedelta(hours=6*i),{'value__vital__heart_rate':110}) for i in range(30)]
        names=['value__vital__heart_rate','time__delta_hours','time__hours_from_admission','state__NewAbnormality']
        x,o,d=raw_sequence(events,(60,start,events[-1][0]),names,{})
        self.assertEqual(len(x),24); self.assertEqual(d[0],0); self.assertEqual(x[0,1],0)
        np.testing.assert_allclose(d[1:],6); np.testing.assert_allclose(x[1:,1],6/24)
        self.assertEqual(x[0,3],1)  # Same start-of-window trajectory semantics as local training.

    def test_future_or_unsorted_events_rejected(self):
        t=datetime(2020,1,1); n=['time__delta_hours']
        with self.assertRaises(ValueError):raw_sequence([(t+timedelta(hours=2),{})],(60,t,t),n,{})
        with self.assertRaises(ValueError):raw_sequence([(t+timedelta(hours=2),{}),(t,{})],(60,t,t+timedelta(hours=3)),n,{})

    def test_structured_ablation_does_not_mutate_shared_raw(self):
        raw=SequenceBundle('pre_6h','structured',['value__a','time__delta_hours','time__hours_from_admission'],[example([[10,0,0],[20,1,1]])],[])
        before=raw.examples[0].x.copy(); branch=make_bundle(raw,'structured')
        s=FeatureScaler(branch.feature_names).fit(branch.examples); s.transform(branch.examples)
        np.testing.assert_array_equal(raw.examples[0].x,before)
        self.assertFalse(np.array_equal(branch.examples[0].x,before))

class ModelTests(unittest.TestCase):
    def test_bidirectional_final_hidden_and_padding_invariance(self):
        torch.set_num_threads(1)
        for name in ['lstm','gru']:
            torch.manual_seed(7)
            model=make_sequence_model(name,3,hidden_dim=8,recurrent_pooling='final_hidden').eval()
            x=torch.randn(3,5,3); lengths=torch.tensor([2,5,3]); o=torch.ones_like(x); d=torch.zeros(3,5)
            packed=torch.nn.utils.rnn.pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
            _,state=model.encoder(packed); hidden=state[0] if name=='lstm' else state
            expected=model.head(torch.cat([hidden[-2],hidden[-1]],dim=-1)).squeeze(-1)
            actual=model(x,o,d,lengths)
            torch.testing.assert_close(actual,expected)
            padded=torch.cat([x,torch.randn(3,4,3)*1e6],dim=1)
            torch.testing.assert_close(actual,model(padded,torch.ones_like(padded),torch.zeros(3,9),lengths))
            for i,L in enumerate(lengths):
                torch.testing.assert_close(actual[i:i+1],model(x[i:i+1,:L],o[i:i+1,:L],d[i:i+1,:L],L.view(1)))

    def test_seed_controls_initialization_independently_of_previous_jobs(self):
        torch.set_num_threads(1)
        ex=[example([[i*.1,0,0],[i*.2,1,1]],label=i%2) for i in range(8)]
        kwargs=dict(model_name='gru',train_examples=ex,validation_examples=ex,test_examples=ex,
            device=torch.device('cpu'),epochs=1,batch_size=4,learning_rate=.001,
            weight_decay=.0001,hidden_dim=8,num_workers=0,seed=55)
        with tempfile.TemporaryDirectory() as tmp:
            a,_=train_sequence(**kwargs,save_path=Path(tmp)/'a.pt')
            torch.randn(1000)
            b,_=train_sequence(**kwargs,save_path=Path(tmp)/'b.pt')
            np.testing.assert_array_equal(a['test'],b['test'])
            ck=torch.load(Path(tmp)/'a.pt',weights_only=False)
            self.assertEqual(ck['recurrent_pooling'],'final_hidden'); self.assertEqual(ck['seed'],55)
            old=make_sequence_model('gru',3,8)
            self.assertEqual(old.pooling,'legacy_last_output')

if __name__=='__main__':unittest.main()
