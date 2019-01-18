# Make Study Groups

Here, the scripts operate with `PROTOCOL2` data table in the processed versions of the DB.

* `run_a_make_combined_class.py` allows creating a DB column which combines several classes
* `run_b_match_by_age_gender.py` performs matching of healthy cases to cases of 
the specified class by age and gender, by this composing a balanced study group
* `run_c_calc_statistics.py` calculates mean and STD of age values for each class and each gender
* `run_d_make_study_group.py` performs train/validation split 
and puts the resulting files to `../data/`
