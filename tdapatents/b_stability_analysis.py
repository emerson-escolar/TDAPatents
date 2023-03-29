import pathlib
import re
import numpy as np

import pandas

## CURRENTLY BROKEN
# UNUSED

if False:
    data_folder = "180901_csv/"
    data_code = "D0"
else:
    data_folder = "200110_csv/"
    data_code = "D1"

folder = pathlib.Path("./cor_pca2d_logmerg/{}m0".format(data_code))
basename = "cor_pca2d_logmerg_{}m0_5wdw1shft".format(data_code)


def get_firms():
    data = pandas.read_csv(data_folder + "firm_rank_name_industry.csv",dtype=str)
    firms = list(data.loc[:,'firm_name'])
    firms = [x.replace(" ", "_") for x in firms]

    return firms

def remove_invalid_columns(data, firms):
    for firm in firms:
        if (data.loc[firm,:] == -1).all() or (data.loc[firm,:].isna()).all():
            data = data.drop([firm])
    return data


def summarize_stat_fixed_number(number, statistic='type'):
    part1 = basename + "_y1976_to_y2005_"
    part2 = 'n' + str(number) + '_'
    part3 = 'o(0\.[0-9]+)_'
    part4 = 'firstgap_flare_stats\.csv'

    pattern = re.compile(part1+part2+part3+part4)

    firms = get_firms()
    data = pandas.DataFrame(index=firms)
    for flare_file in folder.glob("*/*flare_stats.csv"):
        match = pattern.match(flare_file.name)
        if match:
            print(flare_file.name)
            overlap = float(match.group(1))
            raw = pandas.read_csv(str(flare_file),index_col=0).loc[:,statistic]
            data[overlap] = raw

    data = data.reindex(sorted(data.columns), axis=1)
    data = remove_invalid_columns(data, firms)

    print(data)

    return data

def summarize_stat_fixed_overlap(overlap, statistic='type'):
    part1 = basename + "_y1976_to_y2005_"
    part2 = 'n([0-9]+)_'
    part3 = 'o0\.' + str(overlap) + '_'
    part4 = 'firstgap_flare_stats\.csv'

    pattern = re.compile(part1+part2+part3+part4)

    firms = get_firms()
    data = pandas.DataFrame(index=firms)
    for flare_file in folder.glob("*/*flare_stats.csv"):
        match = pattern.match(flare_file.name)
        if match:
            print(flare_file.name)
            n = int(match.group(1))
            raw = pandas.read_csv(str(flare_file),index_col=0).loc[:,statistic]
            data[n] = raw

    data = data.reindex(sorted(data.columns), axis=1)
    data = remove_invalid_columns(data, firms)

    print(data)

    return data


def do_both_fixed_overlap(overlap):
    stab = summarize_stat_fixed_overlap(overlap)
    ofname = "stab_type_{}_o{:.2f}.csv".format(basename, overlap/100.)
    stab.to_csv(str(folder.joinpath(ofname)))

    stab = summarize_stat_fixed_overlap(overlap, statistic='k_index')
    ofname = "stab_index_{}_o{:.2f}.csv".format(basename, overlap/100.)
    stab.to_csv(str(folder.joinpath(ofname)))

def do_both_fixed_number(number):
    stab = summarize_stat_fixed_number(number)
    ofname = "stab_type_{}_n{}.csv".format(basename, number)
    stab.to_csv(str(folder.joinpath(ofname)))

    stab = summarize_stat_fixed_number(number, statistic="k_index")
    ofname = "stab_index_{}_n{}.csv".format(basename, number)
    stab.to_csv(str(folder.joinpath(ofname)))



if __name__ == "__main__":
    # summarize_stat_fixed_number(20, 'k_index')
    # summarize_stat_fixed_overlap(50, 'k_index')

    # stab = summarize_stat_fixed_number(15)
    # stab.to_csv(str(folder.joinpath("stab_cor_pca2d_logmerg_m0_5wdw1shft_n15.csv")))

    do_both_fixed_overlap(50)
    do_both_fixed_number(20)
