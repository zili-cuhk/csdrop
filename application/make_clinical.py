"""Build data/clinical.csv from the cBioPortal patient table of the TCGA PanCancer Atlas LUAD study.

Usage: python make_clinical.py data/data_clinical_patient.txt data/clinical.csv
The input file is part of the study download luad_tcga_pan_can_atlas_2018 at https://www.cbioportal.org.
The output is restricted to the 498 patients listed in data/cohort_patient_ids.txt, the
patients of the clinical table used in the paper.
"""
import os, sys
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src, sep='\t', comment='#', dtype=str)
c = pd.DataFrame({
    'patient_id': d.PATIENT_ID,
    'age': pd.to_numeric(d.AGE, errors='coerce'),
    'sex': d.SEX,
    'stage': d.AJCC_PATHOLOGIC_TUMOR_STAGE,
    'm_stage': d.PATH_M_STAGE,
    'n_stage': d.PATH_N_STAGE,
    't_stage': d.PATH_T_STAGE,
    'OS_STATUS': d.OS_STATUS,
    'os_months': pd.to_numeric(d.OS_MONTHS, errors='coerce'),
    'PFS_STATUS': d.PFS_STATUS,
    'pfs_months': pd.to_numeric(d.PFS_MONTHS, errors='coerce'),
})
c['os_event'] = c.OS_STATUS.str.startswith('1').astype('Int64')
c['pfs_event'] = c.PFS_STATUS.str.startswith('1').astype('Int64')
ids = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cohort_patient_ids.txt')).read().split()
c = c[c.patient_id.isin(ids)].reset_index(drop=True)
c.to_csv(out, index=False)
print(f'wrote {len(c)} patients to {out}')
