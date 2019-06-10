import numpy as np

def color_averager(list_of_triples):
    ans = np.mean(np.array(list_of_triples), axis=0)
    return "rgb({:d},{:d},{:d})".format(int(ans[0]),int(ans[1]),int(ans[2]))

def is_empty_data(data, from_year, to_year):
    if (data.shape[0] == 0):
        print("Warning: year {:d} to {:d} has no nonzero data! Skipping..".format(from_year,to_year))
        return True

    if (data.shape[0] == 1):
        print("Warning: year {:d} to {:d} has only one nonzero firm data! Skipping..".format(from_year, to_year))
        return True
    return False
