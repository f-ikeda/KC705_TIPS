import sys
import os
import glob

import numpy as np
from functools import partial, reduce

from numba import jit

import matplotlib as mp
import matplotlib.pyplot as plt

import time
# for debug

# ########DESCRIPTION########
# USEAGE: $ python3 monitor.py path_to_directory
#
# INPUT: raw binary file, such as there are
#        00000000000001000000000123 <- hit on detector
#        ABB00012345670123456703272 <- Header
#        0000000000000000000800012a <- hit on MR Sync
#        00000000100000000000000450 <- hit on detector
#        00000000000010000000000541 <- hit on detector
#        00000010000000100000000711 <- hit on detector
#        00000000000000000008000957 <- hit on MR Sync
#        00000000000100000000000601 <- hit on detector
#        00000FEE00AAAAAAAAAAAAAAAA <- Footer
#        ABB00012345670123456703273 <- Header
#        00000000100000000000000123 <- hit on detector
#        00000000001000000008000340 <- hit on MR Sync and detector
#        00010000000000000000000601 <- hit on detector
#        00000FEE00AAAAAAAAAAAAAAAA <- Footer
#        01000000000000000000000111 <- hit on detector
#
# OUTPUT ARRAY: sig        = [                            sig of all the corresponding tdc data                            ]
#               tdc        = [123, header's, 12a, 450, 541, 711, 957, 601, footer's, header's, 123, 340, 601, footer's, 111]
#               mrsync     = [-1,        -1, 12a, 12a, 12a, 12a, 957, 957,      957,      957, 957, 340, 340,      340, 340]
#               spillcount = [-1,      3272,3272,3272,3272,3272,3272,3272,     3272,     3273,3273,3273,3273,     3273,3273]
#
# OVERVIEW: READING       ...reading a byte sequence
#           FORMATTING    ...converting a sequence of bytes into an array with elements of 13-byte
#           PROCCESING    ...creating an array of sig, tdc, mrsync and spillcount
#           ANALYZING     ...
#           DRAWING       ...drawing some plots
#
# ISSUES: コインシデンス、各MR Syncごとに独立にしろ
#
# 仕様: ファイル中に、HeaderとFooterとが、同数かつHeader,Footer,...,Header,Footerの順に、必ず1 組以上含まれていなければいけない
#       ファイル中に、MR Syncが必ず1 つ以上含まれていなければいけない
#       Headerよりも後、かつ、MR Syncよりも前にイベントがあった場合、mrsyncに前回のスピルの最後のMR Syncの値を割り当てる
#       spillcountの値は、HeaderとFooterとで同一であること(つまりロスやバグはないこと)を前提に、Headerでの値を使用する
#       ファイル中に含まれるHeaderとFooterとの個数が合わない場合、余分のHeaderあるいはFooter以降のデータは無視する

DATA_UNIT = 13
# bytes
# DIN    <= {HEADER[31:0],SPLCOUNT[15:0],4'd0,BOARD_ID[3:0],48'h0123_4567_89AB}; // HEADER =  REG_HEADER[31:0]
#            x08_Reg[7:0]    <= 8'h01;   // Header
#            x09_Reg[7:0]    <= 8'h23;   // Header
#            x0A_Reg[7:0]    <= 8'h45;   // Header
#            x0B_Reg[7:0]    <= 8'h67;   // Header
# DIN    <= {FOOTER[31:0],SPLCOUNT[15:0],EMCOUNT[15:0],wrCnt[31:0],8'hAB}; // FOOTER = REG_FOOTER[31:0]
#            x0C_Reg[7:0]    <= 8'hAA;   // Footer
#            x0D_Reg[7:0]    <= 8'hAA;   // Footer
#            x0E_Reg[7:0]    <= 8'hAA;   // Footer
#            x0F_Reg[7:0]    <= 8'hAA;   // Footer
# DIN    <= {SIG[76:0],COUNTER[26:0]}; // 104-bits
# {MainHodo[63:0],PMR[11:0],MR_Sync,COUNTER[26:0]}
BITS_SIZE_SIG = 77
# bits
BITS_SIZE_SIG_MRSYNC = 1
# bits
BITS_SIZE_SIG_PMT = 12
# bits
BITS_SIZE_SIG_NEWHOD = 64
# bits
BITS_SIZE_TDC = 27
# bits
BITS_SIZE_BOARDID = 4
# bits
BITS_SIZE_SPILLCOUNT = 16
# bits
BITS_SIZE_EMCOUNT = 16
# bits
BITS_SIZE_WRITECOUNT = 32
# bits
BITS_SIZE_HEADER_UPPER = 32
# bits
BITS_SIZE_HEADER_LOWER = 48
# bits
BITS_SIZE_FOOTER_UPPER = 32
# bits
BITS_SIZE_FOOTER_LOWER = 8
# bits

BITS_WORD_HEADER_UPPER = (0x01234567 << (
    BITS_SIZE_SPILLCOUNT + 4 + BITS_SIZE_BOARDID + BITS_SIZE_HEADER_LOWER))
# use with just ==, on raw(104 bits) data
BITS_WORD_HEADER_LOWER = 0x0123456789AB
# use with just ==, on raw(104 bits) data
BITS_WORD_FOOTER_UPPER = (0xAAAAAAAA << (
    BITS_SIZE_SPILLCOUNT + BITS_SIZE_EMCOUNT + BITS_SIZE_WRITECOUNT + BITS_SIZE_FOOTER_LOWER))
# use with just ==, on raw(104 bits) data
BITS_WORD_FOOTER_LOWER = 0xAB
# use with just ==, on raw(104 bits) data

BITS_MASK_HEADER_UPPER = ((2 ** BITS_SIZE_HEADER_UPPER - 1) <<
                          (BITS_SIZE_SPILLCOUNT + 4 + BITS_SIZE_BOARDID + BITS_SIZE_HEADER_LOWER))
# 104 bits, only the upper BITS_SIZE_HEADER_UPPER bit is filled with 1
BITS_MASK_HEADER_LOWER = 2 ** BITS_SIZE_HEADER_LOWER - 1
# 104 bits, only the lower BITS_SIZE_HEADER_LOWER bit is filled with 1
BITS_MASK_FOOTER_UPPER = ((2 ** BITS_SIZE_FOOTER_UPPER - 1) << (BITS_SIZE_SPILLCOUNT +
                                                                BITS_SIZE_EMCOUNT + BITS_SIZE_WRITECOUNT + BITS_SIZE_FOOTER_LOWER))
# 104 bits, only the upper BITS_SIZE_FOOTER_UPPER bit is filled with 1
BITS_MASK_FOOTER_LOWER = 2 ** BITS_SIZE_FOOTER_LOWER - 1
# 104 bits, only the lower BITS_SIZE_FOOTER_LOWER bit is filled with 1
BITS_MASK_SPILLCOUNT_HEADER = (2 ** BITS_SIZE_SPILLCOUNT - 1 <<
                               (4 + BITS_SIZE_BOARDID + BITS_SIZE_HEADER_LOWER))
# 104 bits, only the corresponding BITS_SIZE_SPILLCOUNT bit is filled with 1
BITS_MASK_SPILLCOUNT_FOOTER = (2 ** BITS_SIZE_SPILLCOUNT - 1 <<
                               (BITS_SIZE_EMCOUNT + BITS_SIZE_WRITECOUNT + BITS_SIZE_FOOTER_LOWER))
# 104 bits, only the corresponding BITS_SIZE_SPILLCOUNT bit is filled with 1
BITS_MASK_SIG = (2 ** BITS_SIZE_SIG - 1) << BITS_SIZE_TDC
# 104 bits, only the upper BITS_SIZE_SIG bit is filled with 1
BITS_MASK_TDC = 2 ** BITS_SIZE_TDC - 1
# 104 bits, only the lower BITS_SIZE_TDC bit is filled with 1

BITS_MASK_SIG_NEWHOD_ALLOR = (2 ** BITS_SIZE_SIG_NEWHOD - 1) - \
    (2 ** (BITS_SIZE_SIG_MRSYNC + BITS_SIZE_SIG_PMT) - 1)
# 77 bits, only the upper BIT_SIZE_SIG_NEWHOD bit is filled with 1
BITS_MASK_SIG_BH1 = 0b0010
BITS_MASK_SIG_BH2 = 0b0100
BITS_MASK_SIG_OLDHOD_ALLOR = 0b1000
BITS_MASK_SIG_MRSYNC = 2 ** BITS_SIZE_SIG_MRSYNC - 1
# 77 bits, only the lower BIT_SIZE_SIG_MRSYNC bit is filled with 1

DELAY_BH1_TO_NEWHOD = 0
# clock
DELAY_BH2_TO_NEWHOD = 0
# clock
DELAY_OLDHOD_TO_NEWHOD = 0
# clock

DELAY_WIDTH = (-1, 0, 1)
# 3 clock, +-1 and 0

CLOCK_TIME = 5
# ns

DATA_TYPE = np.dtype((np.void, DATA_UNIT))


def bytes_to_int(DEADBEEF):
    return int.from_bytes(DEADBEEF, 'big')
    # 遅い！


def formatting_data(data_bytes):
    data = np.frombuffer(data_bytes, DATA_TYPE)
    data = bytes_to_int_universal(data)

    return data


def takein_data(data, condition_header, condition_footer, index_header, index_footer):
    # in case the sizes of index_header[0] and index_footer[0] are different

    if (index_header.size > index_footer.size):
        index_header = index_header[:index_footer.size]
        # まじで？
    else:
        index_footer = index_footer[:index_header.size]

    data = data[:index_footer[-1] + 1]
    condition_header = condition_header[:index_footer[-1] + 1]
    condition_footer = condition_footer[:index_footer[-1] + 1]
    # +1 is to include the footer itself

    return data, condition_header, condition_footer, index_header, index_footer


def processing_spillcount(data):
    # ----SPILLCOUNT----
    condition_header = ((data & (BITS_MASK_HEADER_UPPER | BITS_MASK_HEADER_LOWER)) == (
        BITS_WORD_HEADER_UPPER | BITS_WORD_HEADER_LOWER))
    # making the boolian mask
    condition_footer = ((data & (BITS_MASK_FOOTER_UPPER | BITS_MASK_FOOTER_LOWER)) == (
        BITS_WORD_FOOTER_UPPER | BITS_WORD_FOOTER_LOWER))
    # making the boolian mask
    index_header = np.where(condition_header)
    index_header = index_header[0]
    # getting the position of the Header
    index_footer = np.where(condition_footer)
    index_footer = index_footer[0]

    data, condition_header, condition_footer, index_header, index_footer = takein_data(
        data, condition_header, condition_footer, index_header, index_footer)

    # getting the position of the Footer
    list_spillcount = (np.extract(condition_header, (data)
                                  & BITS_MASK_SPILLCOUNT_HEADER) >> (4 + BITS_SIZE_BOARDID + BITS_SIZE_HEADER_LOWER))
    # getting the list of the Spillcount
    spillcount = np.concatenate([np.full(index_header[0], -1), np.repeat(
        list_spillcount, np.diff(index_header, append=data.size))])
    # using Header's Spillcount
    # when there are no Header in file, index_header[0] causes an error

    print(list_spillcount)

    return data, spillcount, index_header, index_footer, condition_header, condition_footer, list_spillcount


def processing_sig(data):
    # ----SIG----
    sig = (data & BITS_MASK_SIG) >> BITS_SIZE_TDC

    return sig


@jit('i8[:](i8[:],i8[:],i8[:])', nopython=True)
# i8[:] means np.iint64's array
def processing_tdc_overflow(tdc, index_header, index_footer):
    # ----TDC CLOCK COUNT OVERFLOW----
    index_header_and_footer = np.dstack((index_header, index_footer))

    index_within_a_spill = [tdc[array_i[0]+1:array_i[1]]
                            for array_i in index_header_and_footer[0]]
    # +1 for the first TDC data from Header
    index_tdcdiff_within_a_spill = [np.where(np.diff(np.ascontiguousarray(array_i)) < 0)[
        0] for array_i in index_within_a_spill]
    # np.diff() returns local index in input
    # np.ascontiguousarray() makes array_i as contiguous array in memory

    index_overflow_and_footer = [np.zeros(
        len(array_i)+1, dtype=np.int64) for array_i in index_tdcdiff_within_a_spill]
    # np.insert() is not supported by numba, so preparing an array of the size to need in advance
    for i in range(len(index_tdcdiff_within_a_spill)):
        index_overflow_and_footer[i][: len(
            index_tdcdiff_within_a_spill[i])] = index_tdcdiff_within_a_spill[i] + index_header[i]+1+1
        index_overflow_and_footer[i][-1] = index_footer[i]
    # rewrite local index as global index in all data
    # opverwriting the values of a pre-defined array

    for array_i in index_overflow_and_footer:
        for index_k in range(len(array_i)-1):
            tdc[array_i[index_k]: array_i[index_k+1]] = tdc[array_i[index_k]                                                            : array_i[index_k+1]] + (index_k+1) * 2 ** 27

    return tdc


def processing_tdc(data, index_header, index_footer):
    # ----TDC----
    tdc = data & BITS_MASK_TDC

    index_header = index_header.astype(np.int64)
    index_footer = index_footer.astype(np.int64)
    tdc = tdc.astype(np.int64)

    tdc = processing_tdc_overflow(tdc, index_header, index_footer)

    return tdc


def processing_mrsync(sig, condition_header, condition_footer, tdc):
    # ----MR SYNC----
    condition_mrsync = ((sig & BITS_MASK_SIG_MRSYNC) != 0) & (
        ~condition_header) & (~condition_footer)

    index_mrsync = np.where(condition_mrsync)
    list_mrsync = np.extract(condition_mrsync, tdc)

    mrsync = np.concatenate([np.full(index_mrsync[0][0], -1), np.repeat(
        list_mrsync, np.diff(index_mrsync[0], append=sig.size))])
    # when there are no MR Sync sig in file, index_mrsync[0][0] causes an error

    return mrsync, condition_mrsync, list_mrsync


def removing_header_and_footer():
    # this function may not be necessary, because there are boolian masks, such as conditon_header and condition_footer
    # ----REMOVING HEADER AND FOOTER----
    spillcount = np.delete(spillcount, np.concatenate(
        [index_header[0], index_footer[0]]))
    sig = np.delete(sig, np.concatenate([index_header[0], index_footer[0]]))
    tdc = np.delete(tdc, np.concatenate([index_header[0], index_footer[0]]))
    mrsync = np.delete(mrsync, np.concatenate(
        [index_header[0], index_footer[0]]))


@jit('i8[:](i8[:],i8[:])', nopython=True)
# i8[:] means np.iint64's array
def intersect1d_sorted(array_foo, array_bar):
    # both array of array_foo and array_bar must be sorted
    array_intersected = np.empty_like(array_foo)

    index_i = index_j = index_k = 0
    while index_i < array_foo.size and index_j < array_bar.size:
        if array_foo[index_i] == array_bar[index_j]:
            array_intersected[index_k] = array_foo[index_i]
            index_i += 1
            index_j += 1
            index_k += 1
        elif array_foo[index_i] < array_bar[index_j]:
            index_i += 1
        else:
            index_j += 1

    return array_intersected[:index_k]


def coincidence(spillcount, condition_header, condition_footer, list_spillcount, tdc, mrsync, conditions, delays_to_newhod, delay_width):
    # ----COINCIDENCE----
    tdc_coincidenced_p3 = np.empty(0, dtype=np.int64)
    tdc_coincidenced_mrsync = np.empty(0, dtype=np.int64)

    for i in delay_width:
        delays_to_newhod = (i,) + delays_to_newhod[1:]

        tdc_delayed = [(tdc - delay_i)
                       for delay_i in delays_to_newhod]
        # pre-calculate in order to avoid unnecessary repetition in the for statement

        for spill_k in list_spillcount:
            # coincidence has to be considered for each spill independently
            condition_spill_k = ((spillcount == spill_k) & ~
                                 condition_header & ~condition_footer)

            tdc_coincidenced_p3 = np.insert(tdc_coincidenced_p3, tdc_coincidenced_p3.size, reduce(
                intersect1d_sorted, tuple([np.extract(condition_i_and_tdc_delayed_i[0] & condition_spill_k, condition_i_and_tdc_delayed_i[1]) for condition_i_and_tdc_delayed_i in zip(conditions, tdc_delayed)])))
            tdc_coincidenced_mrsync = np.insert(tdc_coincidenced_mrsync, tdc_coincidenced_mrsync.size, reduce(
                np.intersect1d, tuple([np.extract(condition_i_and_tdc_delayed_i[0] & condition_spill_k, condition_i_and_tdc_delayed_i[1] - mrsync) for condition_i_and_tdc_delayed_i in zip(conditions, tdc_delayed)])))
            # この書き方では、MR Syncごとに独立にコインシデンスを取れていない。要修正

    return tdc_coincidenced_p3, tdc_coincidenced_mrsync


def analyzing(spillcount, condition_header, condition_footer, list_spillcount, sig, tdc, mrsync):
    # --------ANALYZING--------
    # ########Write the analysis code here using sig, tdc, mrsync and spillcount########
    # condition_somedetector = bit-calc.(sig)
    condition_newhod_allor = ((sig & BITS_MASK_SIG_NEWHOD_ALLOR)
                              != 0) & ~condition_header & ~condition_footer
    condition_bh1 = ((sig & BITS_MASK_SIG_BH1)
                     != 0 & ~condition_header & ~condition_footer)
    condition_bh2 = ((sig & BITS_MASK_SIG_BH2)
                     != 0 & ~condition_header & ~condition_footer)
    condition_oldhod_allor = ((sig & BITS_MASK_SIG_OLDHOD_ALLOR)
                              != 0 & ~condition_header & ~condition_footer)

    conditions = (condition_newhod_allor, condition_bh1,
                  condition_bh2, condition_oldhod_allor)
    delays_to_newhod = (0, DELAY_BH1_TO_NEWHOD,
                        DELAY_BH2_TO_NEWHOD, DELAY_OLDHOD_TO_NEWHOD)

    # tdc_somedetector_p3 = np.extract(condition_somedetector, tdc)
    # tdc_somedetector_mrsync = np.extract(condition_somedetector, tdc - mrsync)
    tdc_coincidenced_p3, tdc_coincidenced_mrsync = coincidence(spillcount, condition_header, condition_footer, list_spillcount, tdc, mrsync,
                                                               conditions, delays_to_newhod, DELAY_WIDTH)
    # ##################################################################################

    return tdc_coincidenced_p3, tdc_coincidenced_mrsync


def decoding(path_to_file):
    # --------READING--------
    print(path_to_file)
    with open(path_to_file, 'rb') as file:
        data_bytes = file.read()

    data = formatting_data(data_bytes)

    # --------PROCESSING--------
    # ----SPILLCOUNT----
    data, spillcount, index_header, index_footer, condition_header, condition_footer, list_spillcount = processing_spillcount(
        data)
    # ----SIG----
    sig = processing_sig(data)
    # ----TDC----
    tdc = processing_tdc(data, index_header, index_footer)
    # ----MR SYNC----
    mrsync, condition_mrsync, list_mrsync = processing_mrsync(
        sig, condition_header, condition_footer, tdc)

    # --------ANALYZING--------
    tdc_coincidenced_p3, tdc_coincidenced_mrsync = analyzing(
        spillcount, condition_header, condition_footer, list_spillcount, sig, tdc, mrsync)

    return tdc_coincidenced_p3, tdc_coincidenced_mrsync


class plotter(object):

    def __init__(self):
        self.fig = plt.figure(figsize=(10, 8))
        # 画面サイズ
        self.file_name = 'WAIT PLEASE ;-)'

        self.initializer()

    def initializer(self):
        self.fig.suptitle('monitor', size=10)
        # タイトルと文字サイズ
        self.fig.subplots_adjust(
            left=0.10, right=0.90, top=0.90, bottom=0.1, wspace=0.4, hspace=0.4)
        # グラフ全体としての画面端との余白(左,右,上,下), 各グラフ間の余白(左右,上下)
        self.ax_mrsync = plt.subplot2grid((2, 1), (0, 0))
        # 画面の分割数(行,列), そのグラフの置き場(通し番号)
        self.ax_p3 = plt.subplot2grid((2, 1), (1, 0))

        self.ax_mrsync.grid(True)
        # プロット中のマス目の有無
        self.ax_p3.grid(True)

        self.ax_mrsync.set_title(
            "New Hod.'s All Or & BH1 & BH2 & Old Hod.'s All Or" + ': ' + self.file_name)
        # タイトル
        self.ax_p3.set_title(
            "New Hod.'s All Or & BH1 & BH2 & Old Hod.'s All Or" + ': ' + self.file_name)

        self.ax_mrsync.set_xlabel('time from MR Sync [us]')
        # x軸のラベル
        self.ax_mrsync.set_ylabel('events/bin')
        self.ax_p3.set_xlabel('time from P3 [s]')
        self.ax_p3.set_ylabel('events/bin')

        self.lines_mrsync = self.ax_mrsync.hist([-1, -1], label='from MR Sync')
        self.lines_p3 = self.ax_p3.hist([-1, -1], label='from P3')
        # プロットの初期化

    def reloader(self, data_array_mrsync, data_array_p3):
        # 更新ごとにあれこれ設定

        self.ax_mrsync.cla()
        self.ax_mrsync.set_title(
            "New Hod.'s All Or & BH1 & BH2 & Old Hod.'s All Or" + ': ' + self.file_name)
        self.ax_mrsync.set_xlabel('time from MR Sync [us]')
        self.ax_mrsync.set_ylabel('events/bin')
        self.ax_mrsync.grid(True)
        self.lines_mrsync = self.ax_mrsync.hist(data_array_mrsync*CLOCK_TIME*pow(
            10, -3), bins=250, histtype='step', log=True, label='from MR Sync')
        self.ax_mrsync.set_ylim(0.1, None)
        self.ax_mrsync.legend(loc='upper right')

        self.ax_p3.cla()
        self.ax_p3.set_title(
            "New Hod.'s All Or & BH1 & BH2 & Old Hod.'s All Or" + ': ' + self.file_name)
        self.ax_p3.set_xlabel('time from P3 [s]')
        self.ax_p3.set_ylabel('events/bin')
        self.ax_p3.grid(True)
        self.lines_p3 = self.ax_p3.hist(
            data_array_p3*CLOCK_TIME*pow(10, -9), bins=250, histtype='step', log=True, label='from P3')
        self.ax_p3.set_ylim(0.1, None)
        self.ax_p3.legend(loc='upper right')

    def pause(self, second):
        plt.pause(second)

    def finder(self, path_to_directory):
        # 最新から二番目を拾ってくる
        list_of_files = glob.glob(path_to_directory+'*')
        latest_two_file = sorted(
            list_of_files, key=os.path.getctime, reverse=True)[:2]

        if(len(latest_two_file[1:2]) > 0):
            path_to_file = latest_two_file[1]
            self.file_name = path_to_file
            return path_to_file
        else:
            return ''


argument = sys.argv
if((len(argument) < 2) & (argument[-1] != 'b')):
    print('USEAGE: $ python3 monitor.py path_to_directory b(option_debug)')
    # argument[0]: monitor.py
    # argument[1]: path_to_directory
    sys.exit()

path_to_directory = argument[1]

bytes_to_int_universal = np.frompyfunc(bytes_to_int, 1, 1)
# converting function to universal function

PLOTTER = plotter()
while (True):
    if (argument[-1] != 'b'):
        while(len(PLOTTER.finder(path_to_directory)) == 0):
            print('NO FILE ;-)')
            time.sleep(0.3)

        path_to_file = PLOTTER.finder(path_to_directory)
    else:
        path_to_file = 'testdata/1MHz.dat'
        PLOTTER.file_name = path_to_file

    # 仮にP3周期よりも早く描き切った場合、同じ処理が回るので、よくない書き方(残念ながらそんなことないと思うが)
    tdc_coincidenced_p3, tdc_coincidenced_mrsync = decoding(path_to_file)

    PLOTTER.reloader(tdc_coincidenced_mrsync, tdc_coincidenced_p3)
    PLOTTER.pause(0.1)
