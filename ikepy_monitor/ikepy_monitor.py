import sys
import os

# handle command line options
from argparse import ArgumentParser

# file management
import glob

# calculation
import numpy as np

# speeding up
import ctypes
from numba import jit, vectorize

# graphic
import matplotlib.pyplot as plt
# log scale colorbar with imshow
import matplotlib.colors as mcolors

# debug
import time
# import tqdm
# from memory_profiler import profile

######## see .v ########################
# DIN    <= {HEADER[31:0],SPLCOUNT[15:0],4'd0,BOARD_ID[3:0],48'h0123_4567_89AB}; // HEADER =  REG_HEADER[31:0]
# DIN    <= {FOOTER[31:0],SPLCOUNT[15:0],EMCOUNT[15:0],wrCnt[31:0],8'hAB}; // FOOTER = REG_FOOTER[31:0]
# DIN    <= {SIG[76:0],COUNTER[26:0]}; // 104-bits
#                                                         // {MainHodo[63:0],PMR[11:0],MR_Sync,COUNTER[26:0]}
#                                                         // COUNTER start from SPILL signal and increment with 200MHz SYSCLK
########################################

# clock
DELAY_BH1_TO_NEWHOD = 0
DELAY_BH2_TO_NEWHOD = 0
DELAY_TC1_TO_NEWHOD = 0
DELAY_TC2_TO_NEWHOD = 0
DELAY_OLDHOD_TO_NEWHOD = 0

# ns
CLOCK_TIME = 5


def get_option():
    # define command line options
    argparser = ArgumentParser()
    argparser.add_argument(
        '-d', '--dir', type=str, default=None, help='path to directory')
    argparser.add_argument(
        '-f', '--file', type=str, default=None, help='path to file')
    argparser.add_argument(
        '-k', '--kc', type=int, default=1, help='select KC705-(1 or 2)')
    argparser.add_argument(
        '-p', '--pc', type=int, default=1, help='select caolila(1) or cometdaq03(2)')

    return argparser


@jit('i8[:](i8[:],i8[:])', nopython=True)
def coincidence_p3(tdc_foo, tdc_bar):
    # obtain an coincidenced array of tdc values with respect to the p3 signal
    # input: tdc_foo (main), tdc_bar (trigger), (both must be sorted)
    # output: coincidenced array with width of 5 clocks

    # coincidence width, when n, -n, -n+1, ..., -1, 0, 1, ..., n-1, n
    back_and_forth = 5

    tdc_coincidenced = np.empty_like(tdc_foo)
    size_foo = tdc_foo.size
    size_bar = tdc_bar.size

    # bookmarks of three arrays
    i_foo = j_bar = k_coincidenced = 0
    while i_foo < size_foo and j_bar < size_bar:
        diff_bar_foo = tdc_bar[j_bar] - tdc_foo[i_foo]

        if (diff_bar_foo > back_and_forth):
            i_foo += 1
        elif (diff_bar_foo < -back_and_forth):
            j_bar += 1
        else:
            # within the coincidence width
            tdc_coincidenced[k_coincidenced] = tdc_foo[i_foo]
            k_coincidenced += 1
            i_foo += 1

    return tdc_coincidenced[:k_coincidenced]


@jit('UniTuple(i8[:],2)(UniTuple(i8[:],2),UniTuple(i8[:],2))', nopython=True)
def coincidence_mrsync(tuple_foo, tuple_bar):
    # obtain an coincidenced array of tdc values with respect to the mrsync signal
    # input: tuple_foo (tdc and mrsync)(main), tuple_bar (tdc and mrsync)(trigger), (both must be sorted)
    # output: coincidenced tuple (tdc and mrsync) with width of 5 clocks

    # coincidence width, when n, -n, -n+1, ..., -1, 0, 1, ..., n-1, n
    back_and_forth = 5

    tdc_foo = tuple_foo[0]
    mrsync_foo = tuple_foo[1]
    tdc_bar = tuple_bar[0]
    mrsync_bar = tuple_bar[1]

    tdc_coincidenced = np.empty_like(tdc_foo)
    mrsync_coincidenced = np.empty_like(mrsync_foo)
    size_foo = tdc_foo.size
    size_bar = tdc_bar.size

    # bookmarks of three arrays
    i_foo = j_bar = k_coincidenced = 0
    while i_foo < size_foo and j_bar < size_bar:

        # difference in tdc between the two with respect to p3 signal
        diff_bar_foo = tdc_bar[j_bar] - tdc_foo[i_foo]

        if (diff_bar_foo > back_and_forth):
            i_foo += 1
        elif (diff_bar_foo < -back_and_forth):
            j_bar += 1
        else:
            # within the coincidence width
            tdc_coincidenced[k_coincidenced] = tdc_foo[i_foo]
            mrsync_coincidenced[k_coincidenced] = mrsync_foo[i_foo]
            k_coincidenced += 1
            i_foo += 1

    return tdc_coincidenced[:k_coincidenced], mrsync_coincidenced[:k_coincidenced]


# ビットシフトをnjit関数中でufuncとして使うため
@vectorize(['i8(u8, i8)'])
def and_ufunc(a, b):
    return a & b


@jit('Tuple((u8[:,:],u8[:],u8[:]))(i8,i8,u2[:],i8[:],i8[:],u8[:,:],u8[:],u8[:])', nopython=True)
def get_diff_from_mrsync_map_pmt(ch_start, bit_size, sig_foo, tdc_foo, mrsync_foo, diff_from_mrsync_map, underflow, overflow):

    ch = ch_start
    for bit_i in range(bit_size):
        # chの、mrsync基準のtdc
        tdc_from_mrsync = np.extract(
            ((sig_foo >> bit_i) & 0b1) != 0, tdc_foo - mrsync_foo)
        # maybe better way, anyway,
        for diff in tdc_from_mrsync:
            if (diff >= 0) & (diff <= 1200):
                diff_from_mrsync_map[diff][ch] += 1
            elif (diff < 0):
                underflow[ch] += 1
            else:
                overflow[ch] += 1
        ch += 1

    return diff_from_mrsync_map, underflow, overflow


# a little bit different...
@jit('Tuple((u8[:,:],u8[:],u8[:]))(i8,i8,u8[:],i8[:],i8[:],u8[:,:],u8[:],u8[:])', nopython=True)
def get_diff_from_mrsync_map_mppc(ch_start, bit_size, sig_foo, tdc_foo, mrsync_foo, diff_from_mrsync_map, underflow, overflow):

    # print('ch_start:', ch_start)
    ch = ch_start
    for bit_i in range(bit_size):
        bit_i = np.uint64(bit_i)
        # chの、mrsync基準のtdc
        # tdc_from_mrsync \
        #    = np.extract(((sig_foo >> bit_i) & np.uint64(0b1)) != 0, tdc_foo) - np.extract(((sig_foo >> bit_i) & np.uint64(0b1)) != 0, mrsync_foo)
        # print('np.count_nonzero(and_ufunc(sig_foo, (1 << bit_i))):',
        #       np.count_nonzero(and_ufunc(sig_foo, (1 << bit_i))))
        # print('np.unique(tdc_from_mrsync):', np.unique(tdc_from_mrsync))
        tdc_from_mrsync = np.extract(
            ((sig_foo >> bit_i) & np.uint64(0b1)) != 0, tdc_foo - mrsync_foo)
        # maybe better way, anyway,
        for diff in tdc_from_mrsync:
            if (diff >= -1500) & (diff <= 1500):
                diff_from_mrsync_map[diff + 1500][ch] += 1
            elif (diff < -1500):
                underflow[ch] += 1
            else:
                overflow[ch] += 1
        ch += 1
    # print('ch:', ch)

    return diff_from_mrsync_map, underflow, overflow


def recursion(function, arguments):
    # make the recursion process happen
    # input: function (f), and tuples of arguments (A, B, C)
    # output: f(f(A,B),C)

    result = function(arguments[0], arguments[1])
    if (len(arguments) > 2):
        for i in range(2, len(arguments)):
            result = function(result, arguments[i])

    return result


class SomeCalcs(object):
    # some caliculations with related to .so by C

    def __init__(self):
        pass

    def make_zeroarrays(self, data_num):
        # input: number of tdc data in a spill
        # output: zero arrays

        # mppc
        sig_mppc = np.zeros(data_num, dtype='u8')
        # allocate adjacent locations in memory for use with .so
        sig_mppc = np.ascontiguousarray(sig_mppc)
        tdc_mppc = np.zeros(data_num, dtype='i8')
        tdc_mppc = np.ascontiguousarray(tdc_mppc)
        mrsync_mppc = np.zeros(data_num, dtype='i8')
        mrsync_mppc = np.ascontiguousarray(mrsync_mppc)

        # pmt
        sig_pmt = np.zeros(data_num, dtype='u2')
        sig_pmt = np.ascontiguousarray(sig_pmt)
        tdc_pmt = np.zeros(data_num, dtype='i8')
        tdc_pmt = np.ascontiguousarray(tdc_pmt)
        mrsync_pmt = np.zeros(data_num, dtype='i8')
        mrsync_pmt = np.ascontiguousarray(mrsync_pmt)

        # mrsync
        sig_mrsync = np.zeros(data_num, dtype='u1')
        sig_mrsync = np.ascontiguousarray(sig_mrsync)
        mrsync = np.zeros(data_num, dtype='i8')
        mrsync = np.ascontiguousarray(mrsync)

        bookmarks = np.zeros(3, dtype='u4')
        bookmarks = np.ascontiguousarray(bookmarks)

        return sig_mppc, tdc_mppc, mrsync_mppc, \
            sig_pmt, tdc_pmt, mrsync_pmt, \
            sig_mrsync, mrsync, \
            bookmarks

    def set_loader(self):
        # register loader.so

        c_library = np.ctypeslib.load_library("loader.so", ".")

        ######## see .c ########################
        # void find_spills(unsigned long long *spillcount, unsigned long long *offset, unsigned long long *tdcnum,
        #                  unsigned int *bookmarks, char *file_path, long *data_num)
        ########################################

        # specifies the type of the arguments of find_spills()
        c_library.find_spills.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
                                          ctypes.POINTER(ctypes.c_uint32),
                                          ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32)]
        # specifies the type of the return values of find_spills() (but nothing)
        c_library.find_spills.restype = None

        ######## see .c ########################
        # void a_spill_loader(unsigned long long *sig_mppc, signed long long *tdc_mppc, signed long long *mrsync_mppc,
        #                     unsigned short *sig_pmt, signed long long *tdc_pmt, signed long long *mrsync_pmt,
        #                     unsigned char *sig_mrsync, signed long long *mrsync,
        #                     unsigned int *bookmarks, char *file_path, long *offset_to_read, long *data_num)
        ########################################

        c_library.a_spill_loader.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
                                             ctypes.POINTER(ctypes.c_uint16),  ctypes.POINTER(
                                                 ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
                                             ctypes.POINTER(ctypes.c_uint8),  ctypes.POINTER(
                                                 ctypes.c_int64),
                                             ctypes.POINTER(ctypes.c_uint32), ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
        c_library.a_spill_loader.restype = None

        return c_library

    def cutoff_excess(self, sig_mppc, tdc_mppc, mrsync_mppc, sig_pmt, tdc_pmt, mrsync_pmt, sig_mrsync, mrsync, bookmarks):
        # trim the unused portion of an extra large array

        ######## see .c ########################
        # bookmarks[0] = tmp_index_mppc;
        # bookmarks[1] = tmp_index_pmt;
        # bookmarks[2] = tmp_index_mrsync;
        ########################################

        # mppc
        index_mppc = bookmarks[0]
        sig_mppc = sig_mppc[:index_mppc]
        tdc_mppc = tdc_mppc[:index_mppc]
        mrsync_mppc = mrsync_mppc[:index_mppc]

        # pmt
        index_pmt = bookmarks[1]
        sig_pmt = sig_pmt[:index_pmt]
        tdc_pmt = tdc_pmt[:index_pmt]
        mrsync_pmt = mrsync_pmt[:index_pmt]

        # mrsync
        index_mrsync = bookmarks[2]
        sig_mrsync = sig_mrsync[:index_mrsync]
        mrsync = mrsync[:index_mrsync]

        return sig_mppc, tdc_mppc, mrsync_mppc, sig_pmt, tdc_pmt, mrsync_pmt, sig_mrsync, mrsync

    def extrac_detectors(self, sig_mppc, tdc_mppc, mrsync_mppc, sig_pmt, tdc_pmt, mrsync_pmt):
        # extract tdc (and mrsync) for each detector

        # (all or of) new hod.
        condition_newhod_allor = (sig_mppc != 0)
        newhod = np.extract(condition_newhod_allor, tdc_mppc)
        mrsync_newhod = np.extract(condition_newhod_allor, mrsync_mppc)

        ######## see .v ########################
        # assign SIG[76:0] = {SIGNAL[63:0],NEWH[1:0],OLDH[7:3],|OLDH[2:0],TC[1:0],BH[1:0],MR_SYNC};
        ########################################

        # bh1
        condition_bh1 = ((sig_pmt & 0b000000000001) != 0)
        bh1 = np.extract(condition_bh1, tdc_pmt)
        mrsync_bh1 = np.extract(condition_bh1, mrsync_pmt)

        # bh2
        condition_bh2 = ((sig_pmt & 0b000000000010) != 0)
        bh2 = np.extract(condition_bh2, tdc_pmt)
        mrsync_bh2 = np.extract(condition_bh2, mrsync_pmt)

        # tc1
        condition_tc1 = ((sig_pmt & 0b000000000100) != 0)
        tc1 = np.extract(condition_tc1, tdc_pmt)
        mrsync_tc1 = np.extract(condition_tc1, mrsync_pmt)

        # tc2
        condition_tc2 = ((sig_pmt & 0b000000001000) != 0)
        tc2 = np.extract(condition_tc2, tdc_pmt)
        mrsync_tc2 = np.extract(condition_tc2, mrsync_pmt)

        # (all or of) old hod.
        condition_oldhod_allor = ((sig_pmt & 0b001111110000) != 0)
        oldhod = np.extract(condition_oldhod_allor, tdc_pmt)
        mrsync_oldhod = np.extract(condition_oldhod_allor, mrsync_pmt)

        # (all or of) extra pmt (of new hod.)
        condition_expmt_allor = ((sig_pmt & 0b110000000000) != 0)
        expmt = np.extract(condition_expmt_allor, tdc_pmt)
        mrsync_expmt = np.extract(condition_expmt_allor, mrsync_pmt)

        return newhod, mrsync_newhod, \
            bh1, mrsync_bh1, \
            bh2, mrsync_bh2, \
            tc1, mrsync_tc1, \
            tc2, mrsync_tc2, \
            oldhod, mrsync_oldhod, \
            expmt, mrsync_expmt

    def get_hitmap(self, sig_mppc, sig_pmt, kc705_id):
        # 要リファクタリング、特に行列をはっつけるとこ、どのアンプボードに対応しているかを明示した方が良い

        if (kc705_id == 1):

            # for KC705-1
            # ここのネーミング、la->lpc, ha->hpcが正しい、要修正

            la_top = [8, 23, 9, 18, 28, 19, 20, 22,
                      0, 7, 1, 2, 14, 3, 4, 6]
            ha_top = [0]*16
            for i in range(len(la_top)):
                ha_top[i] = la_top[i] + 32

            la_down = [24, 25, 26, 21, 29, 30, 31, 27,
                       10, 11, 12, 5, 15, 16, 17, 13]
            ha_down = [0]*16
            for i in range(len(la_down)):
                # ha_down[i] = la_down[i] + 48
                ha_down[i] = la_down[i] + 32

            # mppcch_to_bitnum[i] = correspoding bit in bit-filed to mppc's i-th ch
            # mppcch_to_bitnum[:32] : half top side of new hod. from left to right (seen by downstream)
            # mppcch_to_bitnum[32:] : half bottom side of new hod. from left to right (seen by downstream)
            mppcch_to_bitnum = []
            mppcch_to_bitnum[len(mppcch_to_bitnum):len(
                mppcch_to_bitnum)] = la_top
            mppcch_to_bitnum[len(mppcch_to_bitnum):len(
                mppcch_to_bitnum)] = ha_top
            mppcch_to_bitnum[len(mppcch_to_bitnum):len(
                mppcch_to_bitnum)] = la_down
            mppcch_to_bitnum[len(mppcch_to_bitnum):len(
                mppcch_to_bitnum)] = ha_down
            # print('[mppcch_to_bitnum[:32],mppcch_to_bitnum[32:]]',
            #       [mppcch_to_bitnum[:32], mppcch_to_bitnum[32:]])

            hitnum_ch_mppc = np.array(
                [np.count_nonzero((sig_mppc & (1 << ch_i)) != 0) for ch_i in mppcch_to_bitnum])

            # format to pass to imshow
            hitmap_newhod_2d = np.array(
                [hitnum_ch_mppc[:32], hitnum_ch_mppc[32:]])
            print('hitmap_newhod_2d(2,32):', hitmap_newhod_2d.shape)
            # add KC705-2's mppc (top and bottom of half left side)
            hitmap_newhod_2d = np.insert(
                hitmap_newhod_2d, 0, np.zeros(32, dtype='i8'), axis=0)
            hitmap_newhod_2d = np.insert(
                hitmap_newhod_2d, 3, np.zeros(32, dtype='i8'), axis=0)
            print('hitmap_newhod_2d(left):', hitmap_newhod_2d.shape)
            # add extra pmt
            top_expmt = np.full((2, 5), np.count_nonzero(
                (sig_pmt & 0b010000000000) != 0), dtype='i8')
            bottom_expmt = np.full((2, 5), np.count_nonzero(
                (sig_pmt & 0b100000000000) != 0), dtype='i8')
            hitmap_expmt_2d = np.insert(top_expmt, 2, bottom_expmt, axis=0)
            hitmap_newhod_2d = np.insert(
                hitmap_newhod_2d, [0], hitmap_expmt_2d, axis=1)
            print('hitmap_newhod_2d(pmt):', hitmap_newhod_2d.shape)

            # add half right side (brank)
            hitmap_newhod_2d = np.insert(
                hitmap_newhod_2d, [5+32], np.zeros((4, 24+5), dtype='i8'), axis=1)
            print('hitmap_newhod_2d(finally):', hitmap_newhod_2d.shape)

        if (kc705_id == 2):

            # amp. board: 5, 6, 7
            hpc_top = [0, 7, 1, 2, 14, 3, 4, 6,
                       8, 23, 9, 18, 28, 19, 20, 22]
            for i in range(len(hpc_top)):
                hpc_top[i] = hpc_top[i] + 32
            hpc_down = [10, 11, 12, 5, 15, 16, 17, 13,
                        24, 25, 26, 21, 29, 30, 31, 27]
            for i in range(len(hpc_down)):
                hpc_down[i] = hpc_down[i] + 32

            lpc_top = [0]*8
            for i in range(len(lpc_top)):
                lpc_top[i] = hpc_top[i] - 32
            lpc_down = [0]*8
            for i in range(len(lpc_down)):
                lpc_down[i] = hpc_down[i] - 32

            mppcch_to_bitnum = []
            mppcch_to_bitnum[len(mppcch_to_bitnum):len(
                mppcch_to_bitnum)] = hpc_top
            mppcch_to_bitnum[len(mppcch_to_bitnum):len(
                mppcch_to_bitnum)] = lpc_top
            mppcch_to_bitnum[len(mppcch_to_bitnum):len(
                mppcch_to_bitnum)] = hpc_down
            mppcch_to_bitnum[len(mppcch_to_bitnum):len(
                mppcch_to_bitnum)] = lpc_down
            # print('[mppcch_to_bitnum[:24],mppcch_to_bitnum[24:]]',
            #       [mppcch_to_bitnum[:24], mppcch_to_bitnum[24:]])

            hitnum_ch_mppc = np.array(
                [np.count_nonzero((sig_mppc & (1 << ch_i)) != 0) for ch_i in mppcch_to_bitnum])
            # format to pass to imshow
            hitmap_newhod_2d = np.array(
                [hitnum_ch_mppc[:24], hitnum_ch_mppc[24:]])
            # print('hitmap_newhod_2d(5,6,7):', hitmap_newhod_2d.shape)
            # print('hitmap_newhod_2d(5,6,7):', hitmap_newhod_2d)

            # add brank amp. 1, 2, 3, 4
            hitmap_newhod_2d = np.insert(
                hitmap_newhod_2d, [0], np.zeros((2, 32), dtype='i8'), axis=1)
            # print('hitmap_newhod_2d(1,2,3,4):', hitmap_newhod_2d.shape)
            # print('hitmap_newhod_2d(1,2,3,4):', hitmap_newhod_2d)

            # add amp. 8
            hitmap_newhod_2d = np.insert(
                hitmap_newhod_2d, 0, np.zeros(56, dtype='i8'), axis=0)
            hitmap_newhod_2d = np.insert(
                hitmap_newhod_2d, 3, np.zeros(56, dtype='i8'), axis=0)
            for ch_i, index in zip([8, 23, 9, 18, 28, 19, 20, 22], [0, 1, 2, 3, 4, 5, 6, 7]):
                hitmap_newhod_2d[0][index*7:index*7 +
                                    7] = np.count_nonzero((sig_mppc & (1 << ch_i)) != 0)
            for ch_i, index in zip([24, 25, 26, 21, 29, 30, 31, 27], [0, 1, 2, 3, 4, 5, 6, 7]):
                hitmap_newhod_2d[3][index*7:index*7 +
                                    7] = np.count_nonzero((sig_mppc & (1 << ch_i)) != 0)
            # print('hitmap_newhod_2d(8):', hitmap_newhod_2d.shape)
            # print('hitmap_newhod_2d(8):', hitmap_newhod_2d)

            # add expmt
            top_expmt = np.full((2, 5), np.count_nonzero(
                (sig_pmt & 0b010000000000) != 0), dtype='i8')
            bottom_expmt = np.full((2, 5), np.count_nonzero(
                (sig_pmt & 0b100000000000) != 0), dtype='i8')
            hitmap_expmt_2d = np.insert(top_expmt, 2, bottom_expmt, axis=0)
            hitmap_newhod_2d = np.insert(
                hitmap_newhod_2d, [56], hitmap_expmt_2d, axis=1)
            # print('hitmap_newhod_2d(expmt):', hitmap_newhod_2d.shape)
            # print('hitmap_newhod_2d(expmt):', hitmap_newhod_2d)

            # add brank expmt
            hitmap_newhod_2d = np.insert(
                hitmap_newhod_2d, [0], np.zeros((4, 5)), axis=1)
            # print('hitmap_newhod_2d(finally):', hitmap_newhod_2d.shape)

        # hitnum_ch_pmt[i]: hitnum in i-th bit(start with 0, to 11)
        hitnum_ch_pmt = np.array(
            [np.count_nonzero((sig_pmt & (1 << ch_i)) != 0) for ch_i in range(12)])
        hitnum_ch_pmt = np.array([hitnum_ch_pmt])

        return hitmap_newhod_2d, hitnum_ch_pmt


class plotter(object):

    def __init__(self):
        # 画面のサイズ
        self.fig = plt.figure(figsize=(10, 8))

        self.file_name = 'WAIT PLEASE ;-)'

        # カラーバーの更新のために保持する
        self.img_chmap_mppc = None
        self.img_chmap_pmt = None
        self.colorbar_chmap_mppc = None
        self.colorbar_chmap_pmt = None

        self.initializer()

    def initializer(self):
        # タイトルと文字サイズ
        self.fig.suptitle('monitor' + '\n' + 'file: ' +
                          self.file_name, size=10)
        self.fig.subplots_adjust(
            left=0.10, right=0.90, top=0.90, bottom=0.10, wspace=0.4, hspace=0.4)
        # グラフ全体としての画面端との余白(左,右,上,下), 各グラフ間の余白(左右,上下)

        ######## Coincidence (MR Sync) ########
        # 画面の分割数(行,列), そのグラフの置き場(通し番号)
        self.ax_mrsync = plt.subplot2grid((3, 2), (0, 1))

        ######## Coincidence (P3) ########
        self.ax_p3 = plt.subplot2grid((3, 2), (1, 1))

        ######## Heat Map (MPPC) ########
        self.ax_chmap_mppc = plt.subplot2grid((3, 2), (2, 0))
        # マップの表示
        self.img_chmap_mppc = self.ax_chmap_mppc.imshow(
            np.zeros((4, 66), dtype='i8'), aspect='auto', cmap='hot_r')
        # カラーバーの表示
        self.colorbar_chmap_mppc = self.fig.colorbar(
            self.img_chmap_mppc, label='Number of Hits', orientation='horizontal')

        ######## Heat Map (PMT) ########
        self.ax_chmap_pmt = plt.subplot2grid((3, 2), (2, 1))
        # マップの表示
        self.img_chmap_pmt = self.ax_chmap_pmt.imshow(
            np.zeros((1, 12)), aspect='auto', cmap='hot_r')
        # カラーバーの表示
        self.colorbar_chmap_pmt = self.fig.colorbar(
            self.img_chmap_pmt, label='Number of Hits', orientation='horizontal')

        ######## Hit Map (Bit-Field) ########
        self.ax_hitmap = plt.subplot2grid((3, 2), (0, 0))

        ######## TDC Diff from MR Sync ########
        self.ax_tdcdiff = plt.subplot2grid((3, 2), (1, 0))

    def reloader(self, tdc_mrsync_w_coinci, tdc_mrsync_wo_coinci, tdc_p3_w_coinci, tdc_p3_wo_coinci, hitmap_newhod_2d, hitnum_ch_pmt, hitmap, ChVsDiffFromMrSync, spillcount):
        self.fig.suptitle('monitor' + '\n' + 'file: ' +
                          self.file_name, size=10)

        ######## Coincidence (MR Sync) ########
        self.ax_mrsync.cla()
        self.ax_mrsync.set_title(
            'TDC from MR Sync: ' + 'Spill No.' + str(spillcount))
        self.ax_mrsync.set_xlabel('time from MR Sync [us]')
        self.ax_mrsync.set_ylabel('events/bin')
        self.ax_mrsync.grid(True)
        self.lines_mrsync = self.ax_mrsync.hist(
            tdc_mrsync_w_coinci*CLOCK_TIME*pow(10, -3),
            bins=250, histtype='step', log=True, label='w/ coincidence', alpha=0.8, color='coral')
        self.lines_mrsync = self.ax_mrsync.hist(
            tdc_mrsync_wo_coinci*CLOCK_TIME*pow(10, -3),
            bins=250, histtype='step', log=True, label='wo coincidence', alpha=0.8, color='dodgerblue')
        # y軸の表示範囲の上限に、最大値の10倍にして余裕をもたせる
        self.ax_mrsync.set_ylim(0.1,
                                self.lines_mrsync[0].max() * 10)
        self.ax_mrsync.legend(loc='upper right')

        ######## Coincidence (P3) ########
        self.ax_p3.cla()
        self.ax_p3.set_title(
            'TDC from P3: ' + 'Spill No.' + str(spillcount))
        self.ax_p3.set_xlabel('time from P3 [s]')
        self.ax_p3.set_ylabel('events/bin')
        self.ax_p3.grid(True)
        self.lines_p3 = self.ax_p3.hist(
            tdc_p3_w_coinci*CLOCK_TIME*pow(10, -9),
            bins=250, histtype='step', log=True, label='w/ coincidence', alpha=0.8, color='coral')
        self.lines_p3 = self.ax_p3.hist(
            tdc_p3_wo_coinci*CLOCK_TIME*pow(10, -9),
            bins=250, histtype='step', log=True, label='wo coincidence', alpha=0.8, color='dodgerblue')
        self.ax_p3.set_ylim(0.1,
                            self.lines_p3[0].max() * 10)
        self.ax_p3.legend(loc='upper right')

        ######## Heat Map (MPPC) ########
        self.ax_chmap_mppc.cla()
        self.ax_chmap_mppc.set_title('New Hod.')
        # hide ticks and it's label
        self.ax_chmap_mppc.set_xticks([])
        self.ax_chmap_mppc.set_yticks([])
        self.ax_chmap_mppc.xaxis.set_ticklabels([])
        self.ax_chmap_mppc.yaxis.set_ticklabels([])
        # make log scale colorbar
        norm_chmap = mcolors.SymLogNorm(
            linthresh=1, vmin=0.8, vmax=hitmap_newhod_2d.max()*10)
        cmap_chmap = plt.cm.viridis
        # entry 0 = white
        cmap_chmap.set_under('white')
        # データの表示
        self.img_chmap_mppc = self.ax_chmap_mppc.imshow(
            hitmap_newhod_2d, cmap=cmap_chmap, aspect='auto', norm=norm_chmap, interpolation='none')

        # overlay rectangular area on plot
        for x in range(5, 61):
            for y in range(2):
                center_area = plt.Rectangle(
                    # (x,y) of upper left corner, width, hight
                    (-0.5 + x, 0.5 + y), 1, 1,
                    edgecolor="Black", fill=False)
                self.ax_chmap_mppc.add_patch(center_area)

        for x in [0 - 0.5, 61 - 0.5]:
            for y in [-0.5, 1.5]:
                leftright_area = plt.Rectangle(
                    (x, y), 5, 2,
                    edgecolor="Black", fill=False)
                self.ax_chmap_mppc.add_patch(leftright_area)

        for x in range(5, 60, 7):
            for y in [-0.5, 2.5]:
                topbottom_area = plt.Rectangle(
                    (-0.5 + x, y), 7, 1,
                    edgecolor="Black", fill=False)
                self.ax_chmap_mppc.add_patch(topbottom_area)

        # カラーバーの更新
        self.colorbar_chmap_mppc.update_normal(self.img_chmap_mppc)

        ######## Heat Map (PMT) ########
        self.ax_chmap_pmt.cla()
        self.ax_chmap_pmt.set_title("PMTs")
        self.ax_chmap_pmt.xaxis.set_ticklabels([])
        self.ax_chmap_pmt.yaxis.set_ticklabels([])
        # データの表示
        self.img_chmap_pmt = self.ax_chmap_pmt.imshow(
            hitnum_ch_pmt, aspect='auto', cmap='hot_r', interpolation='none')
        # テキストラベルの表示
        textlabel = ['BH1', 'BH2', 'TC1', 'TC2', 'OldHod. AllOr', 'PMT',
                     'PMT', 'PMT', 'PMT', 'PMT', 'PMT', 'PMT', 'Ext.PMT', 'Ext.PMT']
        for i in range(0, 12):
            for j in range(0, 1):
                self.ax_chmap_pmt.text(
                    i, j, textlabel[i], ha='center', va='center')
        # カラーバーの更新
        self.colorbar_chmap_pmt.update_normal(self.img_chmap_pmt)

        ######## Hit Map (Bit-Field) ########
        self.ax_hitmap.cla()
        self.ax_hitmap.set_title('CH vs. Entry')
        self.ax_hitmap.set_xlabel('channel number')
        self.ax_hitmap.set_ylabel('entries')

        X = range(1+12+64)
        self.ax_hitmap.step(X, hitmap, where='mid')
        self.ax_hitmap.set_yscale('log')
        self.ax_hitmap.set_ylim(0.1, hitmap.max() * 10)

        self.ax_hitmap.grid(axis='x', color='0.95')

        # ケーブルごとに色塗り
        self.ax_hitmap.axvspan(-0.5, 12.5, color="lightcoral")
        self.ax_hitmap.axvspan(12.5, 20.5, color="palegreen")
        self.ax_hitmap.axvspan(22.5, 30.5, color="palegreen")
        self.ax_hitmap.axvspan(20.5, 22.5, color="plum")
        self.ax_hitmap.axvspan(30.5, 44.5, color="plum")
        self.ax_hitmap.axvspan(12.5+44.5-12.5, 20.5 +
                               44.5-12.5, color="lightskyblue")
        self.ax_hitmap.axvspan(22.5+44.5-12.5, 30.5 +
                               44.5-12.5, color="lightskyblue")
        self.ax_hitmap.axvspan(20.5+44.5-12.5, 22.5+44.5-12.5, color="khaki")
        self.ax_hitmap.axvspan(30.5+44.5-12.5, 44.5+44.5-12.5, color="khaki")

        # 描画範囲x軸を整える
        self.ax_hitmap.set_xlim(-0.5, 76.5)

        ######## TDC Diff from MR Sync ########
        self.ax_tdcdiff.cla()
        self.ax_tdcdiff.set_title('TDC Diff. from MR Sync')
        self.ax_tdcdiff.set_xlabel('channel number')
        self.ax_tdcdiff.set_ylabel('diff. from mrsync [CLK]')

        # make log scale colorbar
        norm_bit = mcolors.SymLogNorm(
            linthresh=1, vmin=0.8, vmax=ChVsDiffFromMrSync.max()*10)
        cmap_bit = plt.cm.viridis
        # entry 0 = white
        cmap_bit.set_under('white')
        self.ax_tdcdiff.imshow(
            ChVsDiffFromMrSync, cmap=cmap_bit, aspect='auto', norm=norm_bit, origin='upper', interpolation='none')

    def pauser(self, second, content_type):

        if content_type == 'file':
            plt.show()
            sys.exit()
        else:
            plt.pause(second)

    def finder(self, path_to_directory, daq_pc):
        # 最新から二番目に作成されたファイルを拾ってくる
        list_of_files = glob.glob(path_to_directory+'*')
        # print('list_of_files:', list_of_files)
        latest_two_file = sorted(
            list_of_files, key=os.path.getctime, reverse=True)[:2]
        # print('latest_two_file:', latest_two_file)

        if(len(latest_two_file[1:2]) > 0):
            # in cometdaq03(2): latest_two_file: [old, new]
            # so, path_to_file = latest_two_file[0]
            # in caolila(1): latest_two_file: [new, old]
            # so, path_to_file = latest_two_file[1]
            if daq_pc == 1:
                path_to_file = latest_two_file[1]
            else:
                path_to_file = latest_two_file[0]
            self.file_name = path_to_file
            return path_to_file
        else:
            return ''


# @profile
def main(SOMECALCS, PLOTTER, file_path, content_type, kc705_id):

    # -------- get a list of spills in a file --------
    T_START = time.time()

    # spillcounts
    spillcounts = np.zeros(1000, dtype='u8')
    spillcounts = np.ascontiguousarray(spillcounts)
    # offsets
    offsets = np.zeros(1000, dtype='u8')
    offsets = np.ascontiguousarray(offsets)
    # tdcnum
    tdcnum = np.zeros(1000, dtype='u8')
    tdcnum = np.ascontiguousarray(tdcnum)
    # bookmarks
    bookmarks = np.zeros(2, dtype='u4')
    bookmarks = np.ascontiguousarray(bookmarks)
    # data_nums
    data_nums = np.zeros(2, dtype='i4')
    data_nums[0] = int(os.path.getsize(file_path) / 13)

    c_library.find_spills(np.ctypeslib.as_ctypes(spillcounts), np.ctypeslib.as_ctypes(offsets), np.ctypeslib.as_ctypes(tdcnum),
                          np.ctypeslib.as_ctypes(bookmarks), buf, np.ctypeslib.as_ctypes(data_nums))

    spillcounts = spillcounts[:bookmarks[0]+1]
    offsets = offsets[:bookmarks[0]+1]
    tdcnum = tdcnum[:bookmarks[0]+1]

    # spillcounts_and_offsets_and_tdcnums = {spillcount : [offset, number of tdc data in the spill]}
    spillcounts_and_offsets_and_tdcnums = {}
    for i in range((bookmarks[0]+1)):
        spillcounts_and_offsets_and_tdcnums[spillcounts[i]] \
            = [offsets[i], tdcnum[i]]

    print('spillcounts:', spillcounts)
    T_END = time.time()
    print('TIME [s] to find spills: ', (T_END - T_START))
    # spillcount_and_offset_and_tdcnum = find_spills(file_path) この一行はCへと置き換わった！

    # -------- process for each spill --------
    for spill_i in spillcounts_and_offsets_and_tdcnums:
        print('spillcount: ' + str(spill_i) + ' ------------------------')
        T_START = time.time()

        # make zero arrays
        sig_mppc, tdc_mppc, mrsync_mppc, \
            sig_pmt, tdc_pmt, mrsync_pmt, \
            sig_mrsync, mrsync, \
            bookmarks\
            = SOMECALCS.make_zeroarrays(spillcounts_and_offsets_and_tdcnums[spill_i][1])

        offset = np.zeros(2, dtype='i4')
        # i don't know why, but an array with 1 element cannot be passed to .so, so i set the number of elements as 2
        offset[0] = spillcounts_and_offsets_and_tdcnums[spill_i][0]
        # for safety, an upper limit on the number of loops that can be done in a .so is given
        data_num = np.zeros(2, dtype='i4')
        data_num[0] = spillcounts_and_offsets_and_tdcnums[spill_i][1]

        ######## see .c ########################
        # void a_spill_loader(unsigned long long *sig_mppc, signed long long *tdc_mppc, signed long long *mrsync_mppc,
        #                     unsigned short *sig_pmt, signed long long *tdc_pmt, signed long long *mrsync_pmt,
        #                     unsigned char *sig_mrsync, signed long long *mrsync,
        #                     unsigned int *bookmarks, char *file_path, long *offset_to_read, long *data_num)
        ########################################

        # load data in a spill
        c_library.a_spill_loader(np.ctypeslib.as_ctypes(sig_mppc), np.ctypeslib.as_ctypes(tdc_mppc), np.ctypeslib.as_ctypes(mrsync_mppc),
                                 np.ctypeslib.as_ctypes(sig_pmt), np.ctypeslib.as_ctypes(
                                     tdc_pmt), np.ctypeslib.as_ctypes(mrsync_pmt),
                                 np.ctypeslib.as_ctypes(
                                     sig_mrsync), np.ctypeslib.as_ctypes(mrsync),
                                 np.ctypeslib.as_ctypes(bookmarks), buf, np.ctypeslib.as_ctypes(offset), np.ctypeslib.as_ctypes(data_num))

        # trim off the excess elements
        sig_mppc, tdc_mppc, mrsync_mppc, \
            sig_pmt, tdc_pmt, mrsync_pmt, \
            sig_mrsync, mrsync\
            = SOMECALCS.cutoff_excess(sig_mppc, tdc_mppc, mrsync_mppc,
                                      sig_pmt, tdc_pmt, mrsync_pmt,
                                      sig_mrsync, mrsync,
                                      bookmarks)

        # extract data of each detector
        newhod, mrsync_newhod, \
            bh1, mrsync_bh1, \
            bh2, mrsync_bh2, \
            tc1, mrsync_tc1, \
            tc2, mrsync_tc2, \
            oldhod, mrsync_oldhod, \
            expmt, mrsync_expmt\
            = SOMECALCS.extrac_detectors(sig_mppc, tdc_mppc, mrsync_mppc,
                                         sig_pmt, tdc_pmt, mrsync_pmt)

        # -------- merge extra pmt to all or of new hod. --------
        newhod = np.concatenate((newhod, expmt))
        newhod.sort(kind='mergesort')
        mrsync_newhod = np.concatenate((mrsync_newhod, mrsync_expmt))
        mrsync_newhod.sort(kind='mergesort')

        # remove duplicate elements?
        # newhod = np.unique(newhod)
        # mrsync_newhod = how()?

        # -------- coincidence (p3) --------
        newhod_coincidenced_p3\
            = recursion(coincidence_p3,
                        tuple([newhod-0,
                              bh1-DELAY_BH1_TO_NEWHOD, bh2-DELAY_BH2_TO_NEWHOD,
                              tc1-DELAY_TC1_TO_NEWHOD, tc2-DELAY_TC1_TO_NEWHOD,
                              oldhod-DELAY_OLDHOD_TO_NEWHOD]))

        # -------- coincidence (mrsync) --------
        tdc_coincidenced_tmp, mrsync_coincidenced_tmp\
            = recursion(coincidence_mrsync, tuple([tuple([newhod-0, mrsync_newhod]),
                                                  tuple(
                [bh1-DELAY_BH1_TO_NEWHOD, mrsync_bh1]), tuple([bh2-DELAY_BH2_TO_NEWHOD, mrsync_bh2]),
                tuple(
                [tc1-DELAY_TC1_TO_NEWHOD, mrsync_tc1]), tuple([tc2-DELAY_TC2_TO_NEWHOD, mrsync_tc2]),
                tuple([oldhod-DELAY_OLDHOD_TO_NEWHOD, mrsync_oldhod])]))
        newhod_coincidenced_mrsync = tdc_coincidenced_tmp - mrsync_coincidenced_tmp

        # -------- hitmap (mppc, pmt) --------
        heatmap_mppc_2d, heatmap_pmt_1d\
            = SOMECALCS.get_hitmap(sig_mppc, sig_pmt, kc705_id)

        # -------- hitmap (bit-filed) --------
        # number of hit to mppc, pmt and mrsync
        hit_mppc = np.zeros(64, dtype='u8')
        hit_pmt = np.zeros(12, dtype='u8')
        hit_mrsync = np.zeros(1, dtype='u8')

        hit_mppc = np.array(
            [np.count_nonzero((sig_mppc & (1 << bit_i)) != 0)
                for bit_i in range(64)], dtype='u8')
        hit_pmt = np.array(
            [np.count_nonzero((sig_pmt & (1 << bit_i)) != 0)
                for bit_i in range(12)], dtype='u8')
        hit_mrsync = np.array(
            [np.count_nonzero((sig_mrsync & (1 << bit_i)) != 0)
                for bit_i in range(1)], dtype='u8')

        hitmap = np.empty(0, dtype='u8')
        hitmap = np.insert(hitmap, hitmap.size, hit_mppc)
        hitmap = np.insert(hitmap, hitmap.size, hit_pmt)
        hitmap = np.insert(hitmap, hitmap.size, hit_mrsync)

        ######## TDC Diff from MR Sync ########
        # 6us(/5ns=1200)だから、MRSyncから1200CLKを見れば十分、オーバーフローもアンダーフローもないはず
        clk_range = 1200
        diff_from_mrsync_map = np.zeros((clk_range, 64+12), dtype='u8')
        diff_from_mrsync_map = np.ascontiguousarray(diff_from_mrsync_map)
        # 0,..., 1198, 1199に収まらないとき
        underflow = np.zeros(64+12, dtype='u8')
        underflow = np.ascontiguousarray(underflow)
        overflow = np.zeros(64+12, dtype='u8')
        underflow = np.ascontiguousarray(underflow)

        if mrsync_pmt.size != 0:
            diff_from_mrsync_map, underflow, overflow = get_diff_from_mrsync_map_pmt(
                0, 12, sig_pmt, tdc_pmt, mrsync_pmt, diff_from_mrsync_map, underflow, overflow)
        if mrsync_mppc.size != 0:
            diff_from_mrsync_map, underflow, overflow = get_diff_from_mrsync_map_mppc(
                12, 64, sig_mppc, tdc_mppc, mrsync_mppc, diff_from_mrsync_map, underflow, overflow)
        # print('np.count_nonzero(diff_from_mrsync_map):', np.count_nonzero(diff_from_mrsync_map))
        # print('diff_from_mrsync_map[0]: ', diff_from_mrsync_map[0])

        T_END = time.time()
        print('TIME [s] to caliculation: ' + str(T_END - T_START))
        ######## Drawing ########
        PLOTTER.reloader(newhod_coincidenced_mrsync, newhod-mrsync_newhod,
                         newhod_coincidenced_p3, newhod,
                         heatmap_mppc_2d, heatmap_pmt_1d,
                         hitmap, diff_from_mrsync_map,
                         spill_i)
        PLOTTER.pauser(0.1, content_type)

    return


if __name__ == '__main__':
    parser = get_option()
    args = parser.parse_args()

    if args.file is not None:
        path_to_file = args.file
        content_type = 'file'
    elif args.dir is not None:
        path_to_directory = args.dir
        content_type = 'directory'
    else:
        parser.print_help()
        sys.exit()

    # 召喚
    SOMECALCS = SomeCalcs()

    # cモジュールをロード
    c_library = SOMECALCS.set_loader()
    # 大きめに1024*5
    buf = ctypes.create_string_buffer(1024*5)

    # 召喚
    PLOTTER = plotter()

    if content_type == 'file':
        buf.value = bytes(path_to_file, encoding='utf-8')
        PLOTTER.file_name = path_to_file
        print("path_to_file:", path_to_file)
        main(SOMECALCS, PLOTTER, path_to_file, content_type, args.kc)
    while (True):
        while(len(PLOTTER.finder(path_to_directory, args.pc)) == 0):
            print('NO FILE ;-)')
            time.sleep(0.3)

        path_to_file = PLOTTER.finder(path_to_directory, args.pc)
        buf.value = bytes(path_to_file, encoding='utf-8')
        PLOTTER.file_name = path_to_file

        print("path_to_file:", path_to_file)
        main(SOMECALCS, PLOTTER, path_to_file, content_type, args.kc)
