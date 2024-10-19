def get_s2w_data(split):
    with open(f'../../dataset/screen2words/split/{split}_screens.txt','r') as fp:
        s = fp.read()
        data_set = s.split('\n')
    data_set.pop()
    data_set=set(data_set)
    return data_set