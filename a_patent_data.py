import pathlib
import numpy as np
import pandas

import collections

# This file contains functions to facilitate extraction of data from data package 180901_csv.
# Format of data is assumed to be as given in those files.
# Using files with different format may (most likely) cause these functions to fail spectacularly.

class DataLocator(object):
    def __init__(self, folder_name, fname_format):
        cur_path = pathlib.Path.cwd()

        self.folder_path = cur_path.joinpath(folder_name)
        self.folder_name = folder_name
        self.fname_format = fname_format

    def get_fname_subbed(self, params):
        fname = self.fname_format.format(*params)
        return str(self.folder_path.joinpath(fname))


class DataLabels(object):
    def __init__(self):
        self.extra_desc = ""
        self.data_name = ""
        self.transforms_name = ""
        self.mode_name = ""

        self.p_sizes = None
        self.rgb_colors = None
        self.years_data = None

def sum_columns(raw_data):
    return np.sum(raw_data.values, axis=1, keepdims=True)

def str_truncate(data, char_limit):
    return data if (char_limit is None or len(data) <= char_limit) else data[:char_limit]


class PatentData(object):
    """
    Class to handle input.
    """

    def __init__(self, extra_data_desc,
                 patent_folder_name, patent_fname_format,
                 firm_translator_fname,
                 firm_translator_func=(lambda x:int(x)),
                 class_translator_fname=None,
                 char_limit=None):
        self.extra_data_desc = extra_data_desc
        self.patent_data_locator = DataLocator(patent_folder_name, patent_fname_format)

        # Translators (used for row and column label renaming)
        self.generate_firm_translator(firm_translator_fname, firm_translator_func, char_limit)
        self.generate_class_translator(class_translator_fname, char_limit)

        self.cosdis_data_locator = None


    def init_transform(self, cosdis_folder_name, cosdis_fname_format):
        if cosdis_folder_name is not None:
            self.cosdis_data_locator = DataLocator(cosdis_folder_name, cosdis_fname_format)


    def generate_class_translator(self, fname, char_limit=None):
        if fname is None:
            self.class_translator = None
            return

        dict_data = pandas.read_csv(fname, dtype=str).values

        name_func = lambda item : str_truncate((item[0] + "_" + item[1].replace(" ", "_")), char_limit)
        self.class_translator = collections.OrderedDict([(int(item[0]), name_func(item)) for item in dict_data])

        raw_colors = {}
        for item in dict_data:
            raw_colors[name_func(item)] = (0,0,0)
        self.patent_raw_colors = raw_colors

    def generate_firm_translator(self, fname, func=(lambda x:int(x)), char_limit=None):
        ## assume names (firm_rank_name_industry.csv) are given in the ff format:
        ##
        ## rank_tgt_unique | firm_name | industry | computer | pharma
        ## 1               | Name One  |          |    1     |
        ## 2               | Name 2    |          |          |   1
        ## ...             | ...       |          |          |
        ##
        ## and that the column 0 entry transformed as func(item[0]) are the labels used in the main data.
        ## industry column is currently ignored.

        dict_data = pandas.read_csv(fname,dtype=str).values
        # print(dict_data)
        name_func = lambda item : str_truncate(item[1].replace(" ", "_"), char_limit)
        translator = collections.OrderedDict([(func(item[0]), name_func(item))  for item in dict_data])

        raw_colors = {}
        for item in dict_data:
            if item[3] == "1":
                raw_colors[name_func(item)] = (255,0,0)
            elif item[4] == "1":
                raw_colors[name_func(item)] = (0,255,0)
            else:
                raw_colors[name_func(item)] = (0,0,255)

        self.firm_translator = translator
        self.firm_raw_colors = raw_colors


    def get_transform(self, year):
        fname = self.cosdis_data_locator.get_fname_subbed((year,))
        C = pandas.read_csv(fname, index_col=[0]).fillna(1)
        for i in range(C.shape[0]):
            C.iat[i,i] = 0
        # ASSUME rows and columns are in the same order:
        C.columns = C.index
        return 1-C


    def get_data(self, year,
                 drop_zero=True, do_transform=True, do_transpose=False,
                 do_log=True, sum_to_one=False):
        """
        Get data corresponding to year.
        """

        labels = DataLabels()
        labels.extra_desc = self.extra_data_desc
        labels.data_name = self.extra_data_desc + "_y{:d}".format(year)

        # Find and read data.
        fname = self.patent_data_locator.get_fname_subbed((year,))
        raw_data = pandas.read_csv(fname, index_col=[0]).T.rename(index=self.firm_translator)

        # Drop zeros?
        orig_num_firms = raw_data.shape[0]
        orig_num_patents = raw_data.shape[1]
        orig_num = orig_num_patents if do_transpose else orig_num_firms
        if drop_zero:
            raw_data = raw_data.loc[(raw_data != 0).any(axis=1), :]
            raw_data = raw_data.loc[:, (raw_data != 0).any(axis=0)]

        # Transforms?
        if do_transform and self.cosdis_data_locator is not None:
            labels.transforms_name = "trans"
            M = self.get_transform(year).loc[raw_data.columns, raw_data.columns]
            data = raw_data.dot(M)
            data.columns = raw_data.columns
        else:
            data = raw_data

        if self.class_translator:
            data.rename(columns = self.class_translator, inplace=True)

        if do_transpose:
            labels.transforms_name = "TENCHI" + labels.transforms_name
            data = data.T

        labels.p_sizes = sum_columns(data)
        # Log transform?
        if do_log:
            labels.transforms_name = "log" + labels.transforms_name
            data = np.log(data + 1)
            labels.p_sizes = np.log(labels.p_sizes + 1)

        # Sum to one?
        if sum_to_one:
            labels.transforms_name = "sumone" + labels.transforms_name
            data = data.div(data.sum(axis=1).replace(to_replace=0, value=1), axis=0)

        # rgb colors
        if do_transpose:
            labels.rgb_colors = [self.patent_raw_colors[x] for x in list(data.index)]
        else:
            labels.rgb_colors = [self.firm_raw_colors[x] for x in list(data.index)]

        # report
        if drop_zero:
            print("Processed {:s} {:s}. {:d} out of {:d} entities nonzero".format(labels.transforms_name, labels.data_name, data.shape[0], orig_num))
        else:
            print("Processed {:s} {:s}. {:d} entities, zeros retained".format(labels.transforms_name, labels.data_name, orig_num))

        return labels, data

    def get_accumulated_data(self, from_year, to_year,
                             drop_zero=True, do_transform=True, do_transpose=False,
                             do_log=True, sum_to_one=False):
        """
        Get accumulated data.
        Adds up data between from_year and to_year, inclusive.
        """

        labels = DataLabels()
        labels.extra_desc = self.extra_data_desc
        labels.data_name = self.extra_data_desc+"_y{:d}_to_y{:d}".format(from_year,to_year)

        if do_transform and self.cosdis_data_locator is not None:
            labels.transforms_name = "trans"
        labels.transforms_name = "accum" + labels.transforms_name

        # Read data
        ans = pandas.DataFrame()
        for year in range(from_year, to_year+1):
            # do not take log before summing!
            # also do not normalize to one!
            _, data = self.get_data(year, drop_zero=drop_zero, do_transform=do_transform, do_transpose=do_transpose,
                                          do_log=False, sum_to_one=False)
            ans = ans.add(data,axis='index',fill_value=0).fillna(0)

        if do_transpose:
            labels.transforms_name = "TENCHI" + labels.transforms_name

        labels.p_sizes = sum_columns(ans)
        if (do_log):
            labels.transforms_name = "log" + labels.transforms_name
            ans = np.log(ans + 1)
            labels.p_sizes = np.log(labels.p_sizes + 1)

        if sum_to_one:
            labels.transforms_name = "sumone" + labels.transforms_name
            ans = ans.div(ans.sum(axis=1).replace(to_replace=0, value=1), axis=0)

        if do_transpose:
            labels.rgb_colors = [self.patent_raw_colors[x] for x in list(ans.index)]
        else:
            labels.rgb_colors = [self.firm_raw_colors[x] for x in list(ans.index)]

        return labels, ans


    def get_merged_accumulated_data(self, from_year, to_year, accum_window, window_shift,
                                    drop_zero=True, do_transform=True, do_transpose=False,
                                    do_log=True, sum_to_one=False):
        """
        Merges accumulated data.

        Moving window sum (accumulation) of data between from_year and to_year inclusive.
        Window is of size accum_window and moves by window_shift.
        """


        labels = DataLabels()
        labels.extra_desc = self.extra_data_desc
        labels.data_name = self.extra_data_desc
        labels.data_name += "_{:d}wdw{:d}shft".format(accum_window, window_shift)
        labels.data_name += "_y{:d}_to_y{:d}".format(from_year, to_year)


        if do_transform and self.cosdis_data_locator is not None:
            labels.transforms_name = "trans"
        labels.transforms_name = "merg" + labels.transforms_name
        if do_transpose:
            labels.transforms_name = "TENCHI" + labels.transforms_name
        if (do_log):
            labels.transforms_name = "log" + labels.transforms_name
        if sum_to_one:
            labels.transforms_name = "sumone" + labels.transforms_name

        # prepare outputs:
        ans = pandas.DataFrame()
        years_data = np.array([])
        p_sizes_all = np.array([])
        rgb_colors_all = np.zeros((0,3))

        # compute:
        for year in range(from_year, to_year+1, window_shift):
            if year + accum_window-1 > to_year:
                break

            year_labels, data  = self.get_accumulated_data(year, year + accum_window-1,
                                                           drop_zero=drop_zero, do_transform=do_transform, do_transpose=do_transpose,
                                                           do_log=do_log, sum_to_one=sum_to_one)
            # append indices with year data
            data.index = data.index.map(lambda x : x + "_y" + str(year))

            # append to outputs
            ans = ans.append(data).fillna(0)
            years_data = np.append(years_data, np.repeat(year,data.shape[0]))

            p_sizes_all = np.append(p_sizes_all, year_labels.p_sizes)
            rgb_colors_all = np.append(rgb_colors_all, year_labels.rgb_colors, axis=0)


        labels.p_sizes = p_sizes_all
        labels.rgb_colors = rgb_colors_all
        labels.years_data = years_data

        return labels, ans


    def get_merged_data(self, from_year, to_year, drop_zero=True, do_transform=True, do_transpose=False,
                        do_log=True, sum_to_one=False):
        labels, ans = self.get_merged_accumulated_data(from_year, to_year,
                                                       accum_window=1, window_shift=1,
                                                       drop_zero=drop_zero, do_transform=do_transform, do_transpose=do_transpose,
                                                       do_log=do_log, sum_to_one=sum_to_one)

        labels.data_name = self.extra_data_desc+"_y{:d}_to_y{:d}".format(from_year, to_year)

        return labels, ans
