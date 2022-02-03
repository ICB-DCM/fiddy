# pip install wheel twine build packaging

rm dist/*
python -m build
twine upload dist/*
