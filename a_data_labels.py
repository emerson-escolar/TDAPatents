import numpy as np
import pandas
import json

class DataLabels(object):
    def __init__(self):
        self.extra_desc = ""
        self.data_name = ""
        self.transforms_name = ""
        self.mode_name = ""

        self.p_sizes = None # np.array or None
        self.rgb_colors = None  # np.array or list or None
        self.unique_members = None  # list or None

        self.years_data = None   # numpy.array or None
        self.intemporal_index = None  # pandas.Index or None

        self.sectors_data = None   # numpy.array or None



    def to_json_fname(self, fname):
        with open(fname, "w") as outfile:
            self.to_json(outfile)

    def to_json(self, fp):
        json.dump(self, fp, cls=DataLabelsEncoder)


    @classmethod
    def from_json_fname(cls, fname):
        with open(fname, "r") as infile:
            return json.load(infile, object_hook=decode_data_labels)



def decode_data_labels(dct):
    if "__data_labels__" in dct:
        ans = DataLabels()
        ans.extra_desc = dct["extra_desc"]
        ans.data_name = dct["data_name"]
        ans.transforms_name = dct["transforms_name"]
        ans.mode_name = dct["mode_name"]

        def to_array(x):
            return None if x is None else np.array(x)

        ans.p_sizes = to_array(dct["p_sizes"])
        ans.rgb_colors = to_array(dct["rgb_colors"])
        ans.years_data = to_array(dct["years_data"])

        ans.unique_members = dct["unique_members"]
        ans.intemporal_index = None if dct["intemporal_index"] is None else pandas.Index(dct["intemporal_index"])

        ans.sectors_data = to_array(dct["sectors_data"])
        return ans

    return dct


class DataLabelsEncoder(json.JSONEncoder):
    def default(self, labels):
        if isinstance(labels, DataLabels):
            data = {"__data_labels__": True,
                    "extra_desc": labels.extra_desc,
                    "data_name" : labels.data_name,
                    "transforms_name": labels.transforms_name,
                    "mode_name": labels.mode_name}
            data["p_sizes"] = self._member_to_maybe_list(labels.p_sizes)
            data["rgb_colors"] = self._member_to_maybe_list(labels.rgb_colors)
            data["unique_members"] = self._member_to_maybe_list(labels.unique_members)
            data["years_data"] = self._member_to_maybe_list(labels.years_data)
            data["intemporal_index"] = self._member_to_maybe_list(labels.intemporal_index)
            data["sectors_data"] = self._member_to_maybe_list(labels.sectors_data)
            return data

    @staticmethod
    def _member_to_maybe_list(member):
        if member is None or isinstance(member, list):
            return member
        elif isinstance(member, np.ndarray) or isinstance(member, pandas.Index):
            return member.tolist()
        return None
