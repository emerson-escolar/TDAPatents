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


class PatentData(object):
    """
    Class to handle input.
    """

    def __init__(self, extra_data_desc,
                 patent_folder_name, patent_fname_format,
                 cosdis_folder_name, cosdis_fname_format,
                 firm_translator_fname,
                 firm_translator_func=(lambda x:int(x)),
                 class_translator_fname = None):
        self.extra_data_desc = extra_data_desc

        self.cosdis_data_locator = None
        if cosdis_folder_name is not None:
            self.cosdis_data_locator = DataLocator(cosdis_folder_name,
                                                   cosdis_fname_format)

        self.patent_data_locator = DataLocator(patent_folder_name,
                                               patent_fname_format)
        self.firm_translator, self.raw_colors = PatentData.generate_firm_translator(firm_translator_fname,
                                                                                    firm_translator_func)

        if class_translator_fname:
            raise NotImplementedError("patent class translator not implemented")
            self.class_translator = PatentData.generate_class_translator(class_translator_fname)
        else:
            self.class_translator = None

    @staticmethod
    def generate_firm_translator(fname, func=(lambda x:int(x))):
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
        translator = collections.OrderedDict([(func(item[0]), item[1].replace(" ", "_")) for item in dict_data])

        raw_colors = {}
        for item in dict_data:
            if item[3] == "1":
                raw_colors[item[1].replace(" ", "_")] = (255,0,0)
            elif item[4] == "1":
                raw_colors[item[1].replace(" ", "_")] = (0,255,0)
            else:
                raw_colors[item[1].replace(" ", "_")] = (0,0,255)

        return translator, raw_colors


    @staticmethod
    def extract_sums(raw_data):
        sums = np.sum(raw_data.values, axis=1, keepdims=True)
        return sums


    def get_transform(self, year):
        # if self.cosdis_data_locator is None:
        #     return 1

        fname = self.cosdis_data_locator.get_fname_subbed((year,))
        C = pandas.read_csv(fname, index_col=[0]).fillna(1)
        for i in range(C.shape[0]):
            C.iat[i,i] = 0
        # ASSUME rows and columns are in the same order:
        C.columns = C.index
        return 1-C


    def get_data(self, year, do_transform=True, do_log=True, sum_to_one=False, drop_zero=True):
        labels = DataLabels()
        labels.extra_desc = self.extra_data_desc
        labels.data_name = self.extra_data_desc + "_y{:d}".format(year)

        # find and read data
        fname = self.patent_data_locator.get_fname_subbed((year,))
        print(fname)
        raw_data = pandas.read_csv(fname, index_col=[0]).T.rename(index=self.firm_translator)

        if self.class_translator:
            raw_data =raw_data.rename(columns = self.class_translator)

        # drop zeros?
        orig_num_firms = raw_data.shape[0]
        if drop_zero:
            raw_data = raw_data.loc[(raw_data != 0).any(axis=1), :]

        # transforms?
        if do_transform and self.cosdis_data_locator is not None:
            labels.transforms_name = "trans"
            M = self.get_transform(year).loc[raw_data.columns, raw_data.columns]
            data = raw_data.dot(M)
            data.columns = raw_data.columns
        else:
            data = raw_data

        # patent sizes
        p_sizes = PatentData.extract_sums(data)

        # log transform?
        if do_log:
            labels.transforms_name = "log" + labels.transforms_name
            data = np.log(data + 1)
            p_sizes = np.log(p_sizes + 1)

        # sum to one?
        # if sum_to_one:
        if sum_to_one:
            labels.transforms_name = "sumone" + labels.transforms_name
            data = data.div(data.sum(axis=1).replace(to_replace=0, value=1), axis=0)

        # rgb colors
        rgb_colors = [self.raw_colors[x] for x in list(data.index)]

        # report
        if drop_zero:
            print("Processed {:s} {:s}. {:d} out of {:d} firms nonzero".format(labels.transforms_name, labels.data_name, data.shape[0], orig_num_firms))
        else:
            print("Processed {:s} {:s}. {:d} firms, zeros retained".format(labels.transforms_name, labels.data_name, orig_num_firms))


        return labels, data, p_sizes, rgb_colors

    def get_accumulated_data(self, from_year, to_year,
                             do_transform=True, do_log=True, sum_to_one=False, drop_zero=True):
        labels = DataLabels()
        labels.extra_desc = self.extra_data_desc
        labels.data_name = self.extra_data_desc+"_y{:d}_to_y{:d}".format(from_year,to_year)

        if do_transform and self.cosdis_data_locator is not None:
            labels.transforms_name = "trans"
        labels.transforms_name = "accum" + labels.transforms_name

        # read data
        ans = pandas.DataFrame()
        for year in range(from_year, to_year+1):
            # do not take log before summing!
            # also do not normalize to one!
            _, data, _, _ = self.get_data(year, do_transform=do_transform,
                                          do_log=False, sum_to_one=False, drop_zero=drop_zero)
            ans = ans.add(data,axis='index',fill_value=0)

        p_sizes = PatentData.extract_sums(ans)
        if (do_log):
            labels.transforms_name = "log" + labels.transforms_name
            ans = np.log(ans + 1)
            p_sizes = np.log(p_sizes + 1)

        if sum_to_one:
            labels.transforms_name = "sumone" + labels.transforms_name
            ans = ans.div(ans.sum(axis=1).replace(to_replace=0, value=1), axis=0)

        rgb_colors = [self.raw_colors[x] for x in list(ans.index)]

        return labels, ans, p_sizes, rgb_colors




    def get_merged_data(self, from_year, to_year,
                        do_transform=True, do_log=True, sum_to_one=False, drop_zero=True):
        labels, ans, p_sizes_all, years_data, rgb_colors_all= self.get_merged_accumulated_data(from_year, to_year, accum_window=1, window_shift=1, do_transform=do_transform, do_log=do_log, sum_to_one=sum_to_one, drop_zero=drop_zero)

        labels.data_name = self.extra_data_desc+"_y{:d}_to_y{:d}".format(from_year, to_year)

        return labels, ans, p_sizes_all, years_data, rgb_colors_all

    def get_merged_accumulated_data(self, from_year, to_year, accum_window, window_shift,
                                    do_transform=True, do_log=True, sum_to_one=False,
                                    drop_zero=True):
        labels = DataLabels()
        labels.extra_desc = self.extra_data_desc
        labels.data_name = self.extra_data_desc
        labels.data_name += "_{:d}wdw{:d}shft".format(accum_window, window_shift)
        labels.data_name += "_y{:d}_to_y{:d}".format(from_year, to_year)


        if do_transform and self.cosdis_data_locator is not None:
            labels.transforms_name = "trans"
        labels.transforms_name = "merg" + labels.transforms_name
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

            _, data, p_sizes, rgb_colors = self.get_accumulated_data(year,
                                                                     year + accum_window-1,
                                                                     do_transform=do_transform,
                                                                     do_log=do_log,
                                                                     sum_to_one=sum_to_one,
                                                                     drop_zero=drop_zero)
            # append indices with year data
            data.index = data.index.map(lambda x : x + "_y" + str(year))

            # append to outputs
            ans = ans.append(data)
            years_data = np.append(years_data, np.repeat(year,data.shape[0]))
            p_sizes_all = np.append(p_sizes_all, p_sizes )
            rgb_colors_all = np.append(rgb_colors_all, rgb_colors, axis=0)


        return labels, ans, p_sizes_all, years_data, rgb_colors_all
