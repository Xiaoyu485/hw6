def reachability(numta, vi, vj, endtime):
    import pickle
    import pandas
    import scipy
    output = open('pathtime.txt', 'r')
    data = pickle.load(output)
    output.close();
    add = pandas.read_csv('addresses.csv')
    adds = add.ShortHand.tolist()
    j = add.ShortHand.tolist().index(vj)
    trans_time = data.time[data.level_0 == vi][data.level_1 == vj].values[0]
    dest_load = data.load[data.level_0 == vi][data.level_1 == vj].values[0]
    dest_reven = data.value[data.level_0 == vi][data.level_1 == vj].values[0]##mark
    F = []
    F.append(trans_time)
    F.append(-dest_load)
    F.append(1)
    temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    temp[j] = 1
    F.extend(temp)
    F.append(dest_reven)
    new_numta = [x+y for x,y in zip(F,numta)]
    reach_vec = new_numta[3:19]
    for i in scipy.arange(len(reach_vec)):
        if reach_vec[i] == 0:
            print ((data.time[data.level_0 == vj][data.level_1 == adds[i]].values[0]) <= endtime)
            if (data.time[data.level_0 == vj][data.level_1 == adds[i]].values[0]) <= endtime:
                if (new_numta[0] + data.time[data.level_0 == vj][data.level_1 == adds[i]].values[0] <=
                        add.TimeWindowEnd[add.ShortHand == adds[i]].values[0]):
                    continue
                else:
                    reach_vec[i] = 1
            else:
                reach_vec[i] = 1
    new_numta[3:19] = reach_vec
    return new_numta