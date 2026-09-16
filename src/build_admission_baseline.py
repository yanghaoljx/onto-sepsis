"""First available measurements, original locked admission cohort, no resampling."""
from pathlib import Path
import pandas as pd
import polars as pl
from scipy.stats import mannwhitneyu, chi2_contingency
from clinical_feature_contract import canonical_unit

R=Path(__file__).resolve().parent
O=R/'outputs/admission_baseline'
LABS={'wbc':'白细胞计数','neutrophil':'中性粒细胞百分比','hemoglobin':'血红蛋白','platelet':'血小板计数','crp':'C反应蛋白','procalcitonin':'降钙素原','lactate':'乳酸','albumin':'白蛋白','bilirubin':'总胆红素','creatinine':'肌酐','urea':'尿素','glucose':'葡萄糖','sodium':'钠','potassium':'钾','chloride':'氯','calcium':'钙','pt':'凝血酶原时间','aptt':'活化部分凝血活酶时间','inr':'国际标准化比值','fibrinogen':'纤维蛋白原','d_dimer':'D-二聚体','ph':'动脉血pH','pao2':'动脉血氧分压','paco2':'动脉血二氧化碳分压','bicarbonate':'碳酸氢根','base_excess':'碱剩余','oxygenation_index':'氧合指数','oxygen_saturation':'血氧饱和度'}
VITALS={'temperature':('体温','°C'),'heart_rate':('心率','次/min'),'respiratory_rate':('呼吸频率','次/min'),'systolic_bp':('收缩压','mmHg'),'diastolic_bp':('舒张压','mmHg'),'spo2':('脉搏血氧饱和度','%')}

def main():
 O.mkdir(parents=True,exist_ok=True)
 ids=pl.read_parquet(R/'ProcessedData/model_dataset/timepoint_dataset.parquet',columns=['visit_no','patient_id','label']).unique().to_pandas()
 assert ids.visit_no.is_unique and ids.groupby('label').size().to_dict()=={0:4079,1:4079}
 b=pd.read_csv(R/'Dataset/baseinfo.csv',dtype=str).drop_duplicates('就诊号',keep='last').rename(columns={'就诊号':'visit_no'})
 d=ids.merge(b,on='visit_no',validate='one_to_one');assert len(d)==8158
 d['admission']=pd.to_datetime(d['入院日期']+' '+d['入院时间'])
 d['discharge']=pd.to_datetime(d['出院日期']+' '+d['出院时间'])
 d['age']=(d.admission-pd.to_datetime(d['出生日期'],errors='coerce')).dt.total_seconds()/86400/365.25
 d.loc[~d.age.between(0,120),'age']=float('nan')
 c=pd.read_csv(R/'ProcessedData/sepsis_cohort.csv',dtype=str)[['visit_no','event_time']]
 d=d.merge(c,on='visit_no',validate='one_to_one');d['onset']=pd.to_datetime(d.event_time,errors='coerce')
 assert d.loc[d.label==1,'onset'].notna().all()
 e=pl.read_parquet(R/'ProcessedData/model_dataset/verified_numeric_events_v5.parquet').to_pandas()
 e=e.merge(d[['visit_no','admission','discharge','onset','label']],on='visit_no',validate='many_to_one')
 e=e[(e.event_time>=e.admission)&(e.event_time<=e.admission+pd.Timedelta(hours=24))&(e.event_time<=e.discharge)&((e.label==0)|(e.event_time<e.onset))]
 first=e.sort_values(['visit_no','feature','event_time']).drop_duplicates(['visit_no','feature'])
 assert not first.duplicated(['visit_no','feature']).any()
 wide=first.pivot(index='visit_no',columns='feature',values='value')
 d=d.merge(wide,on='visit_no',how='left',validate='one_to_one')
 rows=[];stats=[]
 def add(name,col,unit):
  a=d.loc[d.label==1,col].dropna();b=d.loc[d.label==0,col].dropna()
  p=float(mannwhitneyu(a,b,alternative='two-sided',method='asymptotic').pvalue) if len(a) and len(b) else None
  def fmt(x):return '缺失' if not len(x) else f'{x.median():.2f}（{x.quantile(.25):.2f}，{x.quantile(.75):.2f}）'
  rows.append({'指标':name,'单位':unit,'阳性组（4079）':fmt(a),'阴性组（4079）':fmt(b),'阳性有效n':len(a),'阴性有效n':len(b),'P值':'不可计算' if p is None else '<0.001' if p<.001 else f'{p:.3f}'})
  stats.append({'feature':col,'p_raw':p,'positive_n':len(a),'negative_n':len(b),'positive_median':a.median(),'positive_q1':a.quantile(.25),'positive_q3':a.quantile(.75),'negative_median':b.median(),'negative_q1':b.quantile(.25),'negative_q3':b.quantile(.75)})
 add('入院年龄','age','岁')
 sex=pd.crosstab(d['性别'].fillna('缺失'),d.label);p=chi2_contingency(sex)[1]
 for i,k in enumerate(sex.index):
  rows.append({'指标':f'性别编码{k}（含义待字典确认）','单位':'n（%）','阳性组（4079）':f'{sex.loc[k,1]}（{sex.loc[k,1]/4079*100:.1f}%）','阴性组（4079）':f'{sex.loc[k,0]}（{sex.loc[k,0]/4079*100:.1f}%）','阳性有效n':4079,'阴性有效n':4079,'P值':('<0.001' if p<.001 else f'{p:.3f}') if i==0 else '—'})
 for key,(name,unit) in VITALS.items():
  col='value__vital__'+key
  if col in d:add(name,col,unit)
 for key,name in LABS.items():
  col='value__lab__'+key
  if col in d:add(name,col,canonical_unit(col))
 t=pd.DataFrame(rows);t.to_csv(O/'Table1_入院首次基线.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(stats).to_csv(O/'statistics_numeric.csv',index=False)
 first[['visit_no','feature','event_time','value','admission','label']].to_parquet(O/'first_measurements_audit.parquet',index=False)
 d[['visit_no','patient_id','label','age','性别']+list(wide.columns)].to_parquet(O/'admission_values.parquet',index=False)
 note=f'连续变量为中位数（第25，第75百分位数）。每次住院每项指标取入院24小时内首次有效结果，且阳性组结果须早于脓毒症事件。检验时间为已核验数据的结果可用时间max(lab_time,check_time)，不是采血时间。窗口内无值保留缺失，不插补，各项检验基于可用病例。P值为双侧Mann–Whitney U检验，性别为卡方检验，未校正多重比较。8158次住院对应{d.patient_id.nunique()}名患者，P值为住院层面的探索性比较，未校正重复住院相关性。原阴性队列按入院后病程时间分箱抽样。住院天数属于结局，不列入入院基线。旧baseline_4079x4079_table文件来自错误的重新抽样，已被本表替代，不应使用。'
 note+=' 生命体征原始数据只有批次创建时间，同一指标同一时间的多个值在现有清洗流程中取均值；表中生命体征因此是首次可用批次值，不能称为首次单次测量。'
 (O/'Table1_入院首次基线.md').write_text('# 表1 入院首次基线特征\n\n'+t.to_markdown(index=False)+'\n\n'+note+'\n',encoding='utf-8')
 print(t.to_string(index=False))

if __name__=='__main__':main()
