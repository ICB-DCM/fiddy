from pathlib import Path

import petab
import yaml2sbml


yaml2sbml_yaml = 'simple.yaml'
petab_path = Path('..') / 'petab'
petab_yaml = 'problem.yaml'
measurements_tsv = 'measurements.tsv'
model_name = 'simple'

yaml2sbml.yaml2petab(
    yaml_dir=yaml2sbml_yaml,
    output_dir=str(petab_path),
    sbml_name=model_name, 
    petab_yaml_name=petab_yaml,
    measurement_table_name=measurements_tsv,
)
