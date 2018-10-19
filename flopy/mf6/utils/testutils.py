from ..data import mfdatautil


def make_int_tuple(str_list):
    int_list = []
    for item in str_list:
        int_list.append(int(item)-1)
    return tuple(int_list)


def read_vertices(vert_file):
    fd = open(vert_file, 'r')
    vertrecarray = []
    for line in fd:
        fd_spl = line.strip().split()
        vertrecarray.append((int(fd_spl[0]) - 1, float(fd_spl[1]),
                             float(fd_spl[2])))
    fd.close()
    return vertrecarray


def read_cell2d(cell2d_file):
    fd = open(cell2d_file, 'r')
    c2drecarray = []
    for line in fd:
        fd_spl = line.strip().split()
        rec_array = [int(fd_spl[0]) - 1, float(fd_spl[1]), float(fd_spl[2])]
        rec_array.append(int(fd_spl[3]))
        for item in fd_spl[4:]:
            rec_array.append(int(item) - 1)
        c2drecarray.append(tuple(rec_array))
    fd.close()
    return c2drecarray


def read_exchangedata(gwf_file, cellid_size=3):
    exgrecarray = []
    fd = open(gwf_file, 'r')
    for line in fd:
        linesp = line.strip().split()
        exgrecarray.append((make_int_tuple(linesp[0:cellid_size]),
                            make_int_tuple(linesp[cellid_size:cellid_size*2]),
                            int(linesp[cellid_size*2]),
                            float(linesp[cellid_size*2+1]),
                            float(linesp[cellid_size*2+2]),
                            float(linesp[cellid_size*2+3]),
                            float(linesp[cellid_size*2+4])))
    return exgrecarray


def read_gncrecarray(gnc_file, cellid_size=3):
    gncrecarray = []
    fd = open(gnc_file, 'r')
    for line in fd:
        linesp = line.strip().split()
        gncrecarray.append(
            (make_int_tuple(linesp[0:cellid_size]),
             make_int_tuple(linesp[cellid_size:cellid_size*2]),
             make_int_tuple(linesp[cellid_size*2:cellid_size*3]),
             float(linesp[cellid_size*3])))
    return gncrecarray


def read_chdrecarray(chd_file, cellid_size=3):
    fd = open(chd_file, 'r')
    chdrecarray = []
    for line in fd:
        fd_spl = line.strip().split()
        chdrecarray.append((make_int_tuple(fd_spl[0:cellid_size]),
                            float(fd_spl[cellid_size])))
    fd.close()
    return chdrecarray


def read_ghbrecarray(chd_file, cellid_size=3):
    fd = open(chd_file, 'r')
    ghbrecarray = []
    for line in fd:
        fd_spl = line.strip().split()
        ghbrecarray.append((make_int_tuple(fd_spl[0:cellid_size]),
                            float(fd_spl[cellid_size]),
                            float(fd_spl[cellid_size+1])))
    fd.close()
    return ghbrecarray


def read_obs(obs_file, cellid_size=3):
    fd = open(obs_file, 'r')
    obsrecarray = []
    for line in fd:
        fd_spl = line.strip().split()
        if len(fd_spl) >= 2 + cellid_size*2:
            obsrecarray.append(
                (fd_spl[0], fd_spl[1], make_int_tuple(fd_spl[2:2+cellid_size]),
                 make_int_tuple(fd_spl[2 + cellid_size:2 + 2 * cellid_size])))
        else:
            obsrecarray.append((fd_spl[0], fd_spl[1],
                                make_int_tuple(fd_spl[2:2 + cellid_size])))

    fd.close()
    return obsrecarray


def read_std_array(array_file, data_type):
    data_list = []
    fd = open(array_file, 'r')
    for current_line in fd:
        split_line = mfdatautil.ArrayUtil.split_data_line(current_line)
        for data in split_line:
            if data_type == 'float':
                data_list.append(float(data))
            elif data_type == 'int':
                data_list.append(int(data))
            else:
                data_list.append(data)
    fd.close()
    return data_list


def read_sfr_rec(sfr_file, cellid_size=3):
    fd = open(sfr_file, 'r')
    sfrrecarray = []
    for line in fd:
        fd_spl = line.strip().split()
        sfrrecarray.append((int(fd_spl[0]) - 1,
                            make_int_tuple(fd_spl[1:1+cellid_size]),
                            float(fd_spl[cellid_size+1]),
                            int(fd_spl[cellid_size+2]),
                            float(fd_spl[cellid_size+3]),
                            float(fd_spl[cellid_size+4]),
                            float(fd_spl[cellid_size+5]),
                            float(fd_spl[cellid_size+6]),
                            float(fd_spl[cellid_size+7]),
                            int(fd_spl[cellid_size+8]),
                            float(fd_spl[cellid_size+9]),
                            int(fd_spl[cellid_size+10])))
    fd.close()
    return sfrrecarray


def read_reach_con_rec(sfr_file):
    fd = open(sfr_file, 'r')
    sfrrecarray = []
    for line in fd:
        fd_spl = line.strip().split()
        con_arr = []
        for index, item in enumerate(fd_spl):
            item_val = int(item)
            if index == 0:
                item_val -= 1
            else:
                if item_val == -1:
                    item_val = -0.0
                elif item_val < 0:
                    item_val += 1
                    item_val = float(item_val)
                else:
                    item_val -= 1
                    item_val = float(item_val)
            con_arr.append(item_val)
        sfrrecarray.append(tuple(con_arr))
    fd.close()
    return sfrrecarray


def read_reach_div_rec(sfr_file):
    fd = open(sfr_file, 'r')
    sfrrecarray = []
    for line in fd:
        fd_spl = line.strip().split()
        sfrrecarray.append((int(fd_spl[0]) - 1, int(fd_spl[1]) - 1,
                            int(fd_spl[2]) - 1, fd_spl[3]))
    fd.close()
    return sfrrecarray


def read_reach_per_rec(sfr_file):
    fd = open(sfr_file, 'r')
    sfrrecarray = []
    for line in fd:
        fd_spl = line.strip().split()
        per_arr = [int(fd_spl[0]) - 1, fd_spl[1]]
        first = True
        for item in fd_spl[2:]:
            if fd_spl[1].lower() == 'diversion' and first:
                per_arr.append(str(int(item) - 1))
                first = False
            else:
                per_arr.append(item)
        sfrrecarray.append(tuple(per_arr))
    fd.close()
    return sfrrecarray


def read_wells(wel_file, cellid_size=3):
    fd = open(wel_file, 'r')
    welrecarray = []
    for line in fd:
        fd_spl = line.strip().split()
        new_wel = []
        new_wel.append(make_int_tuple(fd_spl[0:cellid_size]))
        new_wel.append(float(fd_spl[cellid_size]))
        for item in fd_spl[cellid_size+1:]:
            new_wel.append(item)
        welrecarray.append(tuple(new_wel))
    fd.close()
    return welrecarray
