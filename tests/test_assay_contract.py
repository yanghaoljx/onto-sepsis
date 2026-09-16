import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from datetime import datetime,timedelta
import numpy as np
from clinical_feature_contract import lab_value,parse_exact_number,rebuild_labs
from temporal_data import FeatureScaler,SequenceExample,row_key,_normalise_time
from run_tuned_mimic_ablation import raw_sequence
import tempfile
import polars as pl

class AssayContractTests(unittest.TestCase):
    def test_per_assay_asof_keeps_earlier_panels_and_excludes_future(self):
        from clinical_feature_contract import rebuild_from_all_events
        t=datetime(2020,1,1);names=['value__lab__glucose','mask__value__lab__glucose','value__lab__wbc','mask__value__lab__wbc']
        row=dict(visit_no='v',split='train',round_time=t+timedelta(hours=3),hours_from_admission=3)
        table={row_key(row):np.zeros(4,np.float32)}
        data=pl.DataFrame({'visit_no':['v']*5,'event_time':[t-timedelta(hours=1),t+timedelta(hours=1),t+timedelta(hours=2),t+timedelta(hours=3),t+timedelta(hours=4)],
            'feature':[names[0],names[0],names[2],names[0],names[0]],'value':[999.,8.,12.,50.,100.]})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events.parquet';data.write_parquet(path)
            audit,availability=rebuild_from_all_events([row],table,names,row_key,_normalise_time,path)
        np.testing.assert_array_equal(table[row_key(row)],[8,1,12,1])
        self.assertEqual(audit['canonical_events_consumed'],2)

    def test_distinct_assays_and_specimens(self):
        for name,unit,code in [('肺泡气中氧分压','mmHg','血气'),('50%氧饱和度氧分压','mmHg','血气'),
            ('活化部分凝血活酶时间比率',None,'凝血'),('凝血酶原时间参考时间','秒','凝血'),
            ('直接胆红素','umol/L','肝功'),('钙离子','mmol/L','血气'),('缓冲碱','mmol/L','血气'),
            ('白细胞','/HP','尿常规'),('高精度HBV病毒载量','IU/ml','病毒'),('血红蛋白A2','%','血常规'),
            ('钠','mmol/L','尿电解质'),('白蛋白','%','蛋白电泳')]:
            with self.subTest(name=name): self.assertIsNone(lab_value(dict(name=name,unit=unit,code=code,value='10'))[0])
        for name,unit,value,feature,expected in [('*总胆红素','μmol/L','17','bilirubin',17),
            ('钙','mmol/L','2.3','calcium',2.3),('[HR]活化部分凝血活酶时间','秒','35','aptt',35),
            ('☆*#血红蛋白','g/dL','12','hemoglobin',120),('  中性分叶核粒细胞百分率','%','65','neutrophil',65),
            ('全血碱剩余','mmol/L','0','base_excess',0)]:
            self.assertEqual(lab_value(dict(name=name,unit=unit,value=value))[0],('value__lab__'+feature,expected))

    def test_exact_numeric_parser(self):
        self.assertEqual(parse_exact_number('1.2e2'),120)
        self.assertEqual(parse_exact_number('-.5'),-.5)
        for s in ['<0.1','>500','1:900','1-2','未测','nan','inf','1,2']:
            self.assertIsNone(parse_exact_number(s))

    def test_zero_availability_and_conflicting_samples(self):
        names=['value__lab__base_excess','mask__value__lab__base_excess']
        t=datetime(2020,1,1);rows=[]
        for i in range(3):
            items=[dict(name='全血碱剩余',value='0',unit='mmol/L',latest_time=t)]
            if i==2:items=[dict(name='全血碱剩余',value=v,unit='mmol/L',latest_time=t+timedelta(hours=2)) for v in ['-2','3']]
            rows.append(dict(split='train',visit_no='v',round_time=t+timedelta(hours=i),latest_lab=items))
        table={row_key(r):np.ones(2,np.float32) for r in rows}
        audit,available=rebuild_labs(rows,table,names,row_key,_normalise_time)
        self.assertEqual(audit['snapshot_item_counts']['same_time_conflicting_values'],1)
        self.assertIn(names[0],available[row_key(rows[1])])
        np.testing.assert_array_equal(table[row_key(rows[1])],[0,0])
        s=FeatureScaler(names);s.mean[:]=[2,0];s.std[:]=1;s.median[:]=[3,0]
        self.assertEqual(s.transform_array([[0,0]],[[0,1]],[[1,1]])[0,0],-2)
        self.assertEqual(s.transform_array([[0,0]],[[0,1]],[[0,1]])[0,0],1)

    def test_external_exclusions_apply_before_ontology_and_zero_carry(self):
        t=datetime(2020,1,1); names=['value__lab__base_excess','mask__value__lab__base_excess','state__Tachypnea','time__delta_hours']
        events=[(t,{'value__lab__base_excess':0,'value__vital__respiratory_rate':40}),
                (t+timedelta(hours=12),{'value__vital__respiratory_rate':40})]
        x,o,d,a=raw_sequence(events,(60,t,t+timedelta(hours=12)),names,{},numeric_contract='v4')
        np.testing.assert_array_equal(x[:,2],0)
        np.testing.assert_array_equal(a[:,0],1)
        self.assertEqual(o[-1,0],0)

if __name__=='__main__':unittest.main()
