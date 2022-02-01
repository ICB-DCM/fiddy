petab_path=../petab

mkdir -p $petab_path

# Validate input yaml2sbml model
yaml2sbml_validate simple.yaml

# Copy measurements to PEtab directory
cp measurements.tsv $petab_path/measurements.tsv

# Write the PEtab problem
python writer.py

# Remove condition parameters from PEtab parameters table
for condition_parameter in parameter_k
do
	sed -i "/^${condition_parameter}/d" $petab_path/parameter*
done

# Validate the PEtab problem
petablint -vy $petab_path/problem.yaml
