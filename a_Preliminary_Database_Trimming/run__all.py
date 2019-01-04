# -*- coding: utf-8 -*-

from run_a_select_fields_of_interest import select_fields_of_interest
from run_b_mark_healthy import mark_healthy
from run_c_calc_statistics import calc_statistics
from run_d_add_validation_flag import add_validation_flag

if __name__ == '__main__':
    select_fields_of_interest()
    mark_healthy()
    calc_statistics()
    add_validation_flag()
