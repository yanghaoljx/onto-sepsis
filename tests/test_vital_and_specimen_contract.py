import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest,tempfile
import polars as pl
from datetime import datetime
from prepare_verified_vital_events import canonical_vitals
from blood_gas_specimen import load_specimen_lookup,allowed_blood_gas_feature

class VitalSpecimenTests(unittest.TestCase):
    def test_same_time_vitals_are_order_independent(self):
        data=pl.DataFrame({'visit_no':['v']*5,'time':['2020-01-01 00:00:00']*5,'code':['HR','HR','HR','RR-Set','BS'],'value':['60','120','99999','40','7']})
        a=canonical_vitals(data.lazy()).collect();b=canonical_vitals(data.reverse().lazy()).collect()
        self.assertEqual(a.height,1);self.assertEqual(a['value'][0],90)
        self.assertEqual(a['value'][0],b['value'][0])
        self.assertEqual(a['records'][0],2)
    def test_arterial_filter_rejects_ambiguous_timestamps(self):
        text='subject_id,hadm_id,charttime,specimen\n1,10,2020-01-01 00:00:00,ART.\n2,20,2020-01-01 00:00:00,ART.\n2,20,2020-01-01 00:00:00,VEN.\n'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'lookup.csv';p.write_text(text);lookup=load_specimen_lookup(p)
        t=datetime(2020,1,1)
        self.assertTrue(lookup[(1,10,t)]);self.assertFalse(lookup[(2,20,t)])
        self.assertFalse(allowed_blood_gas_feature('value__lab__pao2',False))
        self.assertTrue(allowed_blood_gas_feature('value__lab__lactate',False))

if __name__=='__main__':unittest.main()
