# Preliminary Database Trimming

Contains basics script for processing of database version, left after Alexander Misiuk.
Launch `run__all.py` for consecutive running of all scripts.

What should happen:

* Backups of `*.sqlite` files must be automatically created
* Alongside with `PROTOCOL` table, `PROTOCOL2` table must be created
* Columns from `fields_of_interest.txt` must be copied/converted to `PROTOCOL2` (for some columns data type must change)
* Column `class_healthy` must be created and filled
* Column `age` must be calculated
* Column `xray_validated` must be created and partly filled based on the results of manual corrupted X-ray selection 
(by Alexander Misiuk)

To calculated some statistics, run `run_d_calc_statistics.py`.

In case of DB files corruption, just delete the DB files, rename the backup (`*.bak`) files 
and launch `run__all.py` again.  

