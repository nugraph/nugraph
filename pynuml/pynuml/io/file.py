import sys
from abc import ABC
from typing import Any, Callable, Dict, List, Tuple
import psutil

import h5py
import numpy as np
import pandas as pd
from mpi4py import MPI

class Event:
    def __init__(self,
                 index: int,
                 event_id: np.ndarray,
                 data: Dict[str, pd.DataFrame] = {}):
        self.index = index
        self.event_id = event_id
        self.data = data.copy()

    @property
    def name(self):
        r, sr, evt = self.event_id
        return f'r{r}_sr{sr}_evt{evt}'

    def __setitem__(self, key: str, item: pd.DataFrame):
        if type(key) != str:
            raise Exception('Key must be a string!')
        if type(item) != pd.DataFrame:
            raise Exception('Value must be a pandas DataFrame!')
        self.data[key] = item

    def __getitem__(self, key: str):
        if type(key) != str:
            raise Exception('Key must be a string!')
        return self.data[key]

    def __str__(self):
        ret = f'event {self.event_id}\n'
        for group, df in self.data.items():
            ret += f'  {group} ({df.shape[0]} rows):\n'
            for key in df.keys():
                ret += f'    {key}\n'
        return ret

class File:
    def __init__(self, fname: str, parKey: str = "/event_table/event_id"):
        self._colmap = {
            "event_table": {
                "nu_dir": [ "nu_dir_x", "nu_dir_y", "nu_dir_z" ],
                "nu_vtx": [ "nu_vtx_x", "nu_vtx_y", "nu_vtx_z" ],
                "nu_vtx_corr": [ "nu_vtx_corr_x", "nu_vtx_corr_y", "nu_vtx_corr_z" ],
            },
            "particle_table": {
                "start_position": [ "start_position_x", "start_position_y", "start_position_z" ],
                "end_position": [ "end_position_x", "end_position_y", "end_position_z" ],
                "start_position_corr": [ "start_position_corr_x", "start_position_corr_y", "start_position_corr_z" ],
                "end_position_corr": [ "end_position_corr_x", "end_position_corr_y", "end_position_corr_z" ],
            },
            "spacepoint_table": {
                "hit_id": [ "hit_id_u", "hit_id_v", "hit_id_y" ],
                "position": [ "position_x", "position_y", "position_z" ],
            },
            "pandoraPrimary_table": {
                "vtx": [ "vtx_x", "vtx_y", "vtx_z" ],
            },
        }

        # open the input HDF5 file in parallel
        self._fd = h5py.File(fname, "r", driver='mpio', comm=MPI.COMM_WORLD)

        # check if data partitioning key datasets exists in the file
        if parKey not in self._fd.keys():
            raise Exception(f'Error: dataset {parKey} is not found in file {fname}!')

        # parse the name of data partitioning key
        import os.path
        self._parTable = os.path.dirname(parKey)
        # remove leading '/'
        if self._parTable[0] == '/': self._parTable = self._parTable[1:]

        # extract dataset names: partitioning key, seq, and seq_cnt
        self._par_name = os.path.basename(parKey)
        self._seq_name = self._par_name + ".seq"
        self._cnt_name = self._par_name + ".seq_cnt"

        # obtain metadata of dataset parKey, later the dataset will be read
        # into self._index as a numpy array in data_partition()
        self._index = self._fd.get(parKey)
        self._num_events = self._index.shape[0]

        # self._groups is a python list, each member is a 2-element list consisting
        # of a group name, and a python list of dataset names
        self._groups = []

        # a python dictionary storing a sequence-count dataset in each group, keys
        # are group names, values are the sequence-count dataset subarrays assigned
        # to this process
        self._seq_cnt = {}
        self._evt_seq = {}

        self._whole_seq_cnt = {}
        self._whole_seq     = {}

        self._use_seq_cnt = True

        # partition based on event amount of particle table (default)
        self._evt_part = 2

        # a python nested dictionary storing datasets of each group read from the
        # input file. keys of self._data are group names, values are python
        # dictionaries, each has names of dataset in that group as keys, and values
        # storing dataset subarrays
        self._data = {}

        # _starts: data partition start indeices of all processes
        # _counts: data cmount assigned to each process
        starts = None
        counts = None

        # starting array index of parKey assigned to this process
        self._my_start = -1

        # number of array elements of parKey assigned to this process
        self._my_count = -1

    def __del__(self):
        if hasattr(self, '_fd') and self._fd: self._fd.close()

    def __len__(self):
        # inquire the number of unique event IDs in the input file
        return self._num_events

    def __str__(self):
        ret = ""
        for k1 in self._fd.keys():
            ret += f"{k1}:\n"
            for k2 in self._fd[k1].keys():
                if self._seq_name in k2: continue
                ret += f"    {k2}\n"
        return ret

    def __getitem__(self, idx: int):
        """load a single event from file"""
        self.read_data(idx, 1)
        ret = self.build_evt(idx, 1)
        return ret[0] if len(ret) else None

    def check_shape0(self,
                     group: str,
                     keys: List[str] = []) -> None:
        # Check if shape[0] of all datasets in keys are of the same size
        shape0 = self._fd[group][keys[0]].shape[0]
        for k in keys[1:]:
            if k == self._cnt_name: continue # exception is seq_cnt dataset
            if shape0 != self._fd[group][k].shape[0]:
               raise Exception(f'Dataset "/{group}/{k}" shape[0]={self._fd[group][k].shape[0]} inconsistent with {keys[0]}.shape[0]={shape0}')

    def add_group(self,
                  group: str,
                  keys: List[str] = []) -> None:

        # if no keys specified, append all columns in HDF5 group
        if not keys:
            # retrieve all the dataset names of the group
            keys = list(self._fd[group].keys())
            # datasets seq and seq_cnt are not needed
            if group != self._parTable and self._par_name in keys: keys.remove(self._par_name)
            if self._seq_name in keys: keys.remove(self._seq_name)
            if self._cnt_name in keys: keys.remove(self._cnt_name)
        else:
            # Check if datasets in keys are available in the file
            for k in keys:
                if k not in self._fd[group].keys():
                   raise Exception(f'Dataset "/{group}/{k}" does not exist')

        # if group does not already exist, just add it
        if not self._groups or group not in self._groups[:][0]:
            self.check_shape0(group, keys)
            self._groups.append([ group, keys ])
            return

        # if group is already present, need to figure out whether any extra keys need to be added
        for g, k in self._groups:
            if g == group:
                self.check_shape0(group, keys)
                for key in keys:
                    if key not in k:
                        k.append(key)
                return
        raise Exception(f'group "{group}" not found.')

    def keys(self):
        return self._fd.keys()

    def _cols(self,
              group: str,
              key: str) -> List[str]:
        if key == self._par_name: return [ "run", "subrun", "event" ]
        if group in self._colmap and key in self._colmap[group].keys(): return self._colmap[group][key]
        elif self._fd[group][key].shape[1]==1: return [key]
        else: return [ key+"_"+str(c) for c in range(0,self._fd[group][key].shape[1])]

    def get_dataframe(self,
                      group: str,
                      keys: List[str] = []) -> pd.DataFrame:
        if not keys:
            keys = list(self._fd[group].keys())
            if self._seq_name in keys: keys.remove(self._seq_name)
            if self._cnt_name in keys: keys.remove(self._cnt_name)
        dfs = [ pd.DataFrame(np.array(self._fd[group][key]), columns=self._cols(group, key)) for key in keys ]
        return pd.concat(dfs, axis="columns").set_index(["run","subrun","event"])

    def get_dataframe_evt(self,
                          group: str,
                          keys: List[str] = []) -> pd.DataFrame:
        if not keys:
            keys = list(self._data[group].keys())
            if self._seq_name in keys: keys.remove(self._seq_name)
            if self._cnt_name in keys: keys.remove(self._cnt_name)
        dfs = [ pd.DataFrame(np.array(self._data[group][key]), columns=self._cols(group, key)) for key in keys ]
        df = pd.concat(dfs, axis="columns")
        evt_idx_col = []
        for seq in self._seq_cnt[group]:
            evt_idx_col += seq[1]*[seq[0]]
        df['evt_idx'] = evt_idx_col
        return df

    def index(self, idx: int):
        """get the index for a given row"""
        return self._my_index[idx - self._my_start]

    def read_seq(self) -> None:
        for group, datasets in self._groups:
            try:
                # read an HDF5 dataset into a numpy array
                self._whole_seq[group] = np.array(self._fd[group+"/"+self._seq_name])
            except KeyError:
                print(f"Error: dataset {group}/{self._seq_name} does not exist")
                sys.stdout.flush()
                sys.exit(1)

    def read_seq_cnt(self) -> None:
        # Dataset event_id.seq_cnt stores the event IDs sorted in an increasing
        # order. There is no duplicated values and gaps may exist between any
        # two consecutive elements. Note dataset event_id.seq_cnt in group
        # self._parTable contains all event IDs with no gap.
        for group, datasets in self._groups:
            try:
                # read an HDF5 dataset into a numpy array
                self._whole_seq_cnt[group] = np.array(self._fd[group+"/"+self._cnt_name])
            except KeyError:
                print(f"Error: dataset {group}/{self._cnt_name} does not exist")
                sys.stdout.flush()
                sys.exit(1)

    def data_partition(self) -> None:
        # Calculate the start indices and counts of evt.seq assigned to each process
        # self._starts: a numpy array of size nprocs
        # self._counts: a numpy array of size nprocs
        # Note self._starts and self._counts are matter only in root process.
        # self._my_start: (== self._starts[rank]) this process's start
        # self._my_count: (== self._counts[rank]) this process's count
        # self._my_index: partitioned dataset i.e. assigned to this process

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
        self._starts = np.zeros(nprocs, dtype=int)
        self._counts = np.zeros(nprocs, dtype=int)

        if rank == 0:
            if self._use_seq_cnt:
                self.read_seq_cnt()
            else:
                self.read_seq()

            num_events = self._num_events

            if self._evt_part == 0:
                # Below implements event ID based partitioning, which
                # calculates the start and count of evt.seq id for each process
                _count = num_events // nprocs
                for j in range(num_events % nprocs):
                    self._starts[j] = _count * j + j
                    self._counts[j] = _count + 1

                for j in range(num_events % nprocs, nprocs):
                    self._starts[j] = _count * j + num_events % nprocs
                    self._counts[j] = _count

            elif self._evt_part == 1:
                # event amount based partitioning, which calculates event sizes
                # across all groups. Note it is possible multiple consecutive rows
                # a dataset have the same event ID. It is also possible some event
                # IDs contain no data. First, we accumulate numbers of events
                # across all groups
                evt_size = np.zeros(num_events, dtype=int)
                if self._use_seq_cnt:
                    for group, datasets in self._groups:
                        seq_cnt = self._whole_seq_cnt[group]
                        num_datasets = len(datasets)
                        for i in range(seq_cnt.shape[0]):
                            evt_size[seq_cnt[i, 0]] += seq_cnt[i, 1] * num_datasets
                else:
                    for group, datasets in self._groups:
                        seq = self._whole_seq[group]
                        for i in range(seq.shape[0]):
                            evt_size[seq[i, 0]] += 1

                # now we have collected the number of events per event ID across all groups
                total_evt_num = np.sum(evt_size)
                avg_evt_num = total_evt_num // nprocs
                avg_evt = total_evt_num // num_events / 2

                # assign ranges of event IDs to individual processes
                acc_evt_num = 0
                rank_id = 0
                for j in range(num_events):
                    if rank_id == nprocs - 1: break
                    if acc_evt_num + evt_size[j] >= avg_evt_num:
                        remain_l = avg_evt_num - acc_evt_num
                        remain_r = evt_size[j] - remain_l
                        if remain_l > remain_r and remain_l > avg_evt:
                            # assign event j to rank_id
                            self._counts[rank_id] += 1
                            acc_evt_num = 0
                        else:
                            # assign event j to rank_id+1
                            self._counts[rank_id+1] = 1
                            acc_evt_num = evt_size[j]
                        # done with rank_id i
                        rank_id += 1
                        self._starts[rank_id] = self._starts[rank_id-1] + self._counts[rank_id-1]
                    else:
                        self._counts[rank_id] += 1
                        acc_evt_num += evt_size[j]
                self._counts[nprocs-1] += num_events - j

            elif self._evt_part == 2:
                # use event amounts in the particle_table only to partition events
                seq_cnt = self._whole_seq_cnt['particle_table']
                total_evt_num = np.sum(seq_cnt[:,1])
                avg_evt_num = total_evt_num // nprocs
                avg_evt = total_evt_num // seq_cnt.shape[0] / 2

                self._starts[0] = seq_cnt[0,0]
                acc_evt_num = 0
                rank_id = 0
                for j in range(seq_cnt.shape[0]):
                    if rank_id == nprocs - 1: break
                    if acc_evt_num + seq_cnt[j,1] >= avg_evt_num:
                        remain_l = avg_evt_num - acc_evt_num
                        remain_r = seq_cnt[j,1] - remain_l
                        # if remain_r > remain_l:
                        if remain_l > remain_r and remain_l > avg_evt:
                            # assign event j to rank_id
                            self._counts[rank_id] = seq_cnt[j+1, 0] - self._starts[rank_id]
                            self._starts[rank_id+1] = seq_cnt[j+1, 0]
                            acc_evt_num = 0
                        else:
                            # assign event j to rank_id+1
                            self._counts[rank_id] = seq_cnt[j, 0] - self._starts[rank_id]
                            self._starts[rank_id+1] = seq_cnt[j, 0]
                            acc_evt_num = seq_cnt[j, 1]
                        # done with rank_id
                        rank_id += 1
                    else:
                        acc_evt_num += seq_cnt[j, 1]

                self._counts[nprocs-1] = num_events - self._starts[nprocs-1]

        # All processes participate the collective communication, scatter.
        # Root distributes start and count to all processes. Note only root process
        # uses self._starts and self._counts.
        start_count = np.empty([nprocs, 2], dtype=int)
        start_count[:, 0] = self._starts[:]
        start_count[:, 1] = self._counts[:]
        recvbuf = np.empty(2, dtype=int)
        comm.Scatter(start_count, recvbuf, root=0)
        self._my_start = recvbuf[0]
        self._my_count = recvbuf[1]

        # This process is assigned event IDs of range from self._my_start to
        # (self._my_start + self._my_count - 1)

        # each process reads its share of dataset and stores it in a numpy
        # array
        self._my_index = np.array(self._index[self._my_start : self._my_start + self._my_count, :])

    def binary_search_min(self, key, base, nmemb):
        low = 0
        high = nmemb
        while low != high:
                mid = (low + high) // 2
                if base[mid] < key:
                        low = mid + 1
                else:
                        high = mid
        return low

    def binary_search_max(self, key, base, nmemb):
        low = 0
        high = nmemb
        while low != high:
                mid = (low + high) // 2
                if base[mid] <= key:
                        low = mid + 1
                else:
                        high = mid
        return (low - 1)

    def calc_bound_seq(self, group):
        # return the lower and upper array indices of subarray assigned to this
        # process, using the partition sequence dataset

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        displ  = np.zeros([nprocs], dtype=int)
        count  = np.zeros([nprocs], dtype=int)
        bounds = np.zeros([nprocs, 2], dtype=int)

        all_evt_seq = None
        if rank == 0:
            # root reads the entire dataset self._seq_name, if not already
            if not self._whole_seq: self.read_seq()

            all_evt_seq = self._whole_seq[group]
            dim = len(all_evt_seq)

            # calculate displ, count to be used in scatterV for all processes
            for i in range(nprocs):
                if self._counts[i] == 0: continue
                end = self._starts[i] + self._counts[i] - 1
                bounds[i, 0] = self.binary_search_min(self._starts[i], all_evt_seq, dim)
                bounds[i, 1] = self.binary_search_max(end,             all_evt_seq, dim)
                displ[i] = bounds[i, 0]
                count[i] = bounds[i, 1] - bounds[i, 0] + 1

        lower_upper = np.empty([2], dtype=int)

        # root distributes start and end indices to all processes
        comm.Scatter(bounds, lower_upper, root=0)

        # this process is assigned array indices from lower to upper
        lower = 0
        upper = 0
        if self._my_count > 0:
            lower = lower_upper[0]
            upper = lower_upper[1] + 1

        # root scatters the subarray of evt_seq to all processes
        self._evt_seq[group] = np.zeros(upper - lower, dtype=np.int64)
        comm.Scatterv([all_evt_seq, count, displ, MPI.LONG_LONG], self._evt_seq[group], root=0)

        return lower, upper

    def calc_bound_seq_cnt(self, group):
        # return the lower and upper array indices of subarray assigned to this
        # process, using the partition sequence-count dataset

        comm   = MPI.COMM_WORLD
        rank   = comm.Get_rank()
        nprocs = comm.Get_size()

        displ   = np.zeros([nprocs], dtype=int)
        count   = np.zeros([nprocs], dtype=int)
        seq_cnt = np.zeros([nprocs, 2], dtype=int)

        all_seq_cnt = None
        if rank == 0:
            # root reads the entire dataset self._cnt_name, if not already
            if not self._whole_seq_cnt: self.read_seq_cnt()

            all_seq_cnt = self._whole_seq_cnt[group]
            dim = len(all_seq_cnt)

            # calculate displ, count for all processes to be used in scatterV
            recv_rank = 0  # receiver rank
            displ[recv_rank] = 0
            seq_cnt[recv_rank, 0] = 0
            seq_end = self._starts[recv_rank] + self._counts[recv_rank]
            seq_id = 0
            for i in range(dim):
                if all_seq_cnt[i, 0] >= seq_end :
                    seq_cnt[recv_rank, 1] = i - displ[recv_rank]
                    recv_rank += 1  # move on to the next receiver rank
                    seq_end = self._starts[recv_rank] + self._counts[recv_rank]
                    displ[recv_rank] = i
                    seq_cnt[recv_rank, 0] = seq_id
                seq_id += all_seq_cnt[i, 1]

            # last receiver rank
            seq_cnt[recv_rank, 1] = dim - displ[recv_rank]

            displ[:] *= 2
            count[:] = seq_cnt[:, 1] * 2

        # root distributes seq_cnt to all processes
        my_seq_cnt = np.empty([2], dtype=int)
        comm.Scatter(seq_cnt, my_seq_cnt, root=0)

        # self._seq_cnt[group][:, 0] is the event ID
        # self._seq_cnt[group][:, 1] is the number of elements
        self._seq_cnt[group] = np.empty([my_seq_cnt[1], 2], dtype=np.int64)

        # root scatters the subarray of evt_seq to all processes
        comm.Scatterv([all_seq_cnt, count, displ, MPI.LONG_LONG], self._seq_cnt[group], root=0)

        lower = 0
        upper = 0
        if self._my_count > 0:
            lower = my_seq_cnt[0]
            upper = my_seq_cnt[0] + np.sum(self._seq_cnt[group][:, 1])

        # this process is assigned array indices from lower to upper

        return lower, upper

    def read_data(self,
                  start: int,
                  count: int) -> None:
        # (sequentially) read subarrays of all datasets in all groups that fall
        # in the range of self._seq_name, starting from 'start' and amount of 'count'

        for group, datasets in self._groups:
            if self._use_seq_cnt:
                # use evt_id.seq_cnt to calculate subarray boundaries
                # reads the entire dataset self._cnt_name, if not already
                if not self._whole_seq_cnt or group not in self._whole_seq_cnt.keys():
                    self.read_seq_cnt()
                all_seq_cnt = self._whole_seq_cnt[group]
                # search indices of start and end in all_seq_cnt
                # all_seq_cnt[:,0] are all unique
                ilower = np.searchsorted(all_seq_cnt[:,0], start)
                iupper = np.searchsorted(all_seq_cnt[:,0], start+count)
                self._seq_cnt[group] = np.array(all_seq_cnt[ilower:iupper], dtype=np.int64)
                lower = np.sum(all_seq_cnt[0:ilower, 1])
                upper = lower + np.sum(all_seq_cnt[ilower:iupper, 1])
            else:
                # use evt_id.seq to calculate subarray boundaries
                # root reads the entire dataset self._seq_name, if not already
                if not self._whole_seq: self.read_seq()
                all_evt_seq = self._whole_seq[group]
                dim = len(all_evt_seq)
                # search indices of start and end in all_seq
                # all_seq[:] are not unique
                end = start + count - 1
                lower = self.binary_search_min(start, all_evt_seq, dim)
                upper = self.binary_search_max(end,   all_evt_seq, dim)
                upper += 1
                self._evt_seq[group] = np.array(all_evt_seq[lower:upper], dtype=np.int64)

            # Iterate through all the datasets and read the subarray from index lower
            # to upper and store it into a dictionary with the names of group and
            # dataset as the key.
            self._data[group] = {}
            for dset in datasets:
                # read subarray into a numpy array
                self._data[group][dset] = np.array(self._fd[group][dset][lower : upper])

        self._my_start = start
        self._my_count = count
        # read assigned partitioning key dataset into a numpy array
        self._my_index = np.array(self._index[start : start + count, :])

    def read_data_all(self,
                      use_seq_cnt: bool = True,
                      evt_part: int = 2,
                      profile: bool = False) -> None:
        # use_seq_cnt: True  - use event.seq_cnt dataset to calculate partitioning
        #                      starts and counts
        #              False - use event.seq dataset to calculate starts and counts
        # evt_part: 0  - partition based on event IDs
        #           1 - partition based on event amount
        #           2 - partition based on event amount of particle table (default)
        # Parallel read dataset subarrays assigned to this process ranging from
        # array index of self._my_start to (self._my_start + self._my_count - 1)
        if profile:
            par_time = 0
            bnd_time = 0
            rds_time = 0
            time_s = MPI.Wtime()

        self._use_seq_cnt = use_seq_cnt
        self._evt_part = evt_part

        # calculate the data partitioning start indices and amounts assigned to
        # each process. Set self._starts, self._counts, self._my_start,
        # self._my_count, and self._my_index
        self.data_partition()

        if profile:
            time_e = MPI.Wtime()
            par_time = time_e - time_s
            time_s = time_e

        for group, datasets in self._groups:
            if self._use_seq_cnt:
                # use evt_id.seq_cnt to calculate subarray boundaries
                lower, upper = self.calc_bound_seq_cnt(group)
            else:
                # use evt_id.seq to calculate subarray boundaries
                lower, upper = self.calc_bound_seq(group)

            if profile:
                time_e = MPI.Wtime()
                bnd_time += time_e - time_s
                time_s = time_e

            # Iterate through all the datasets and read the subarray from index lower
            # to upper and store it into a dictionary with the names of group and
            # dataset as the key.
            self._data[group] = {}
            for dset in datasets:
                # read subarray into a numpy array
                self._data[group][dset] = np.array(self._fd[group][dset][lower : upper])

            if profile:
                time_e = MPI.Wtime()
                rds_time += time_e - time_s
                time_s = time_e

        if profile:
            rank   = MPI.COMM_WORLD.Get_rank()
            nprocs = MPI.COMM_WORLD.Get_size()

            total_t = np.array([par_time, bnd_time, rds_time])
            max_total_t = np.zeros(3)
            MPI.COMM_WORLD.Reduce(total_t, max_total_t, op=MPI.MAX, root = 0)
            min_total_t = np.zeros(3)
            MPI.COMM_WORLD.Reduce(total_t, min_total_t, op=MPI.MIN, root = 0)
            if rank == 0:
                print("---- Timing break down of the file read phase (in seconds) -------")
                if self._use_seq_cnt:
                    print(f'Use "{self._cnt_name}" to calculate subarray boundaries')
                else:
                    print(f'Use "{self._seq_name}" to calculate subarray boundaries')

                print("data partitioning           time ", end='')
                print("MAX=%8.2f  MIN=%8.2f" % (max_total_t[0], min_total_t[0]))
                print("calc boundaries             time ", end='')
                print("MAX=%8.2f  MIN=%8.2f" % (max_total_t[1], min_total_t[1]))
                print("read datasets               time ", end='')
                print("MAX=%8.2f  MIN=%8.2f" % (max_total_t[2], min_total_t[2]))
                print("(MAX and MIN timings are among %d processes)" % nprocs)

    def build_evt(self,
                  start: int = None,
                  count: int = None) -> List[Dict]:
        # This process is responsible for event IDs from start to (start+count-1).
        # All data of the same event ID will be used to create a graph.
        # This function collects all data based on self._seq_name, or
        # self._cnt_name into a python list containing Pandas DataFrames, one
        # for a unique event ID.
        if not self._groups:
            raise Exception('cannot build event without adding any HDF5 groups')

        ret_list = []

        if start is None: start = self._my_start
        if count is None: count = self._my_count

        if self._use_seq_cnt:
            # track the latest used index per group
            idx_grp = dict.fromkeys(self._data.keys(), 0)

            # accumulate starting array index per group
            idx_start = dict.fromkeys(self._data.keys(), 0)

            # whether idx is presented in a group's _seq_cnt[:,0]
            idx_found = dict.fromkeys(self._data.keys(), False)

        # Iterate through assigned event IDs
        for idx in range(int(start), int(start+count)):
            # check if idx is missing in all groups
            is_missing = True
            if self._use_seq_cnt:
                for group in self._data.keys():
                    idx_found[group] = False
                    dim = self._seq_cnt[group].shape[0]

                    # check against the max of this group's
                    if idx > self._seq_cnt[group][dim-1, 0]:
                        continue

                    # check and search for idx in _seq_cnt[group][:,0]
                    if idx == idx_grp[group]:
                        # this is most likely the case when building all graphs
                        # for all events at once
                        idx_found[group] = True
                        idx_grp[group] = idx
                    elif idx - idx_grp[group] <= 8:
                        # linear search for idx in _seq_cnt[group][:,0]
                        # if distance is less than 8, linear search is faster
                        for jj in range(idx_grp[group], dim):
                            if idx == self._seq_cnt[group][jj, 0]:
                                idx_found[group] = True
                                idx_grp[group] = jj
                                break
                            elif idx < self._seq_cnt[group][jj, 0]:
                                break
                    else:
                        # binary search for idx in _seq_cnt[group][:,0]
                        # Note there is no duplicated values in
                        # _seq_cnt[group][:,0] and the values are sorted in an
                        # increasing order
                        low = idx_grp[group]
                        high = dim
                        while low < high:
                            mid = (low + high) // 2
                            if self._seq_cnt[group][mid, 0] < idx:
                                low = mid + 1
                            elif self._seq_cnt[group][mid, 0] > idx:
                                high = mid
                            else:
                                idx_found[group] = True
                                idx_grp[group] = mid
                                break

                    if idx_found[group]:
                        if idx == start:
                            # Calculate starting array index only necessary for
                            # first idx. For 2nd and later, idx_start is
                            # accumulated later
                            idx_start[group] = self._seq_cnt[group][0:idx_grp[group], 1].sum()
                        # skip self._parTable group, as it is not used to
                        # determine whether idx is missing.
                        if group != self._parTable:
                            is_missing = False
            else:
                for group in self._data.keys():
                    dim = len(self._evt_seq[group])
                    # dataset event_id.seq may contain duplicated event IDs
                    # IDs in this dataset are sorted in a monotonically non-decreasing order
                    lower = self.binary_search_min(idx, self._evt_seq[group], dim)
                    upper = self.binary_search_max(idx, self._evt_seq[group], dim) + 1
                    if lower < upper:
                        is_missing = False
                        break

            # this idx is missing in all groups
            if is_missing:
                continue

            # for each event seq ID, create a dictionary, ret
            #   first item: key is "index" and value is the event seq ID
            #   remaining items: key is group name and value is a Pandas DataFrame
            #   containing the dataset subarray in this group with the event ID, idx
            ret = Event(idx, self.index(idx))

            # Iterate through all groups
            for group in self._data.keys():

                if self._use_seq_cnt:
                    # Note self._seq_cnt[group][:, 0] is the event ID
                    # Note self._seq_cnt[group][:, 1] is the number of elements

                    if not idx_found[group]:
                        # For idx is missing from this group but not in other
                        # groups, create an empty Pandas DataFrame
                        dfs = []
                        for dataset in self._data[group].keys():
                            data_dataframe = pd.DataFrame(columns=self._cols(group, dataset))
                            dfs.append(data_dataframe)
                        ret[group] = pd.concat(dfs, axis="columns")
                        continue

                    if group == self._parTable:
                        # Special treatment for group self._parTable, as its
                        # seq_cnt[:,1] contains all 1s and earlier increment of
                        # idx_grp[group] may be skipped due to missing idx
                        lower = idx_grp[group]
                        upper = lower + 1
                    else:
                        lower = idx_start[group]
                        upper = self._seq_cnt[group][idx_grp[group], 1] + lower

                    # The range from lower to upper (exclusively) is subarray
                    # indices of elements belonging to the same event ID, idx

                    if count > 1:
                        # increment start array indices to avoid searching the
                        # already-done data
                        idx_start[group] += self._seq_cnt[group][idx_grp[group], 1]
                        idx_grp[group] += 1

                else:
                    # Note self._evt_seq stores event ID values and is already sorted in
                    # an increasing order
                    dim = len(self._evt_seq[group])

                    # Find the local start and end row indices for this event ID, idx
                    lower = self.binary_search_min(idx, self._evt_seq[group], dim)
                    upper = self.binary_search_max(idx, self._evt_seq[group], dim) + 1

                # dfs is a python list containing Pandas DataFrame objects
                dfs = []
                for dataset in self._data[group].keys():
                    if lower >= upper:
                        # idx is missing from the dataset self._seq_name,
                        # In this case, create an empty numpy array
                        data = np.array([])
                    else:
                        # array elements from lower to upper of this dataset have the
                        # event ID == idx
                        data = self._data[group][dataset][lower : upper]

                    # create a Pandas DataFrame to store the numpy array
                    df = pd.DataFrame(data, columns=self._cols(group, dataset))
                    for col in df.columns:
                        if df[col].dtype == '|S64' or df[col].dtype == 'object':
                            df[col] = df[col].str.decode('utf-8')
                    dfs.append(df)

                # concatenate into the dictionary "ret" with group names as keys
                ret[group] = pd.concat(dfs, axis="columns")

            # Add all dictionaries "ret" into a list.
            # Each of them corresponds to the data of one single event ID
            ret_list.append(ret)

        return ret_list

    def process(self,
                processor: Callable[[Event], Tuple[str, Any]],
                out: Callable[[Any, str], None]) -> None:
        '''Process all events in this data partition'''
        xproc = psutil.Process()
        comm = MPI.COMM_WORLD
        nprocs = comm.Get_size()
        rank = comm.Get_rank()
        if rank == 0:
            out.write_metadata(processor.metadata)
        self.read_data_all()

        verbose = False

        # whether or not to build graphs one event at a time
        build_one_evt_at_a_time = True

        if build_one_evt_at_a_time == False:
            evt_list = self.build_evt()
            for evt in evt_list:
                name, data = processor(evt)
                if data is not None: out(name, data)
            if verbose:
                print("Build all events: MPI rank %3d Memory footprint = %8.1f MiB" %
                    (rank, xproc.memory_info().rss/ 1024.0 ** 2))
        else:
            # Iterate through assigned event IDs
            for idx in range(int(self._my_start), int(self._my_start+self._my_count)):
                evt = self.build_evt(idx, 1)
                if len(evt) > 0:
                    name, data = processor(evt[0])
                    if data is not None: out(name, data)
            if verbose:
                print("Build 1 event at a time: MPI rank %-3d Memory footprint = %8.1f MiB" %
                    (rank, xproc.memory_info().rss/ 1024.0 ** 2))

