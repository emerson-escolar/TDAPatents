import pathlib
import re
import numpy as np

import pandas

folder = pathlib.Path("./cor_pca2d_logmerg/m0")

def get_firms():
    data = pandas.read_csv("180901_csv/190204_firm_rank_name_industry.csv",dtype=str)
    firms = list(data.loc[:,'firm_name'])
    firms = [x.replace(" ", "_") for x in firms]

    return firms

def remove_invalid_columns(data, firms):
    for firm in firms:
        if (data.loc[firm,:] == -1).all() or (data.loc[firm,:].isna()).all():
            data = data.drop([firm])
    return data


def summarize_stat_fixed_number(number, statistic='type'):
    part1 = 'cor_pca2d_logmerg_m0_5wdw1shft_y1976_to_y2005_'
    part2 = 'n' + str(number) + '_'
    part3 = 'o(0\.[0-9]+)_'
    part4 = 'firstgap_flare_stats\.txt'

    pattern = re.compile(part1+part2+part3+part4)

    firms = get_firms()
    data = pandas.DataFrame(index=firms)
    for flare_file in folder.glob("*/*flare_stats.txt"):
        print(flare_file.name)
        match = pattern.match(flare_file.name)
        if match:
            overlap = float(match.group(1))
            raw = pandas.read_csv(str(flare_file),index_col=0).loc[:,statistic]
            data[overlap] = raw

    data = data.reindex(sorted(data.columns), axis=1)
    data = remove_invalid_columns(data, firms)

    print(data)

    return data

def summarize_stat_fixed_overlap(overlap, statistic='type'):
    part1 = 'cor_pca2d_logmerg_m0_5wdw1shft_y1976_to_y2005_'
    part2 = 'n([0-9]+)_'
    part3 = 'o0\.' + str(overlap) + '_'
    part4 = 'firstgap_flare_stats\.txt'

    pattern = re.compile(part1+part2+part3+part4)

    firms = get_firms()
    data = pandas.DataFrame(index=firms)
    for flare_file in folder.glob("*/*flare_stats.txt"):
        print(flare_file.name)
        match = pattern.match(flare_file.name)
        if match:
            n = int(match.group(1))
            raw = pandas.read_csv(str(flare_file),index_col=0).loc[:,statistic]
            data[n] = raw

    data = data.reindex(sorted(data.columns), axis=1)
    data = remove_invalid_columns(data, firms)

    print(data)

    return data


if __name__ == "__main__":
    # summarize_stat_fixed_number(20, 'k_C')
    # summarize_stat_fixed_overlap(50, 'k_C')

    # summarize_stat_fixed_number(20)
    stab = summarize_stat_fixed_overlap(50)
    stab.to_csv(str(folder.joinpath("stab_o.50.csv")))
