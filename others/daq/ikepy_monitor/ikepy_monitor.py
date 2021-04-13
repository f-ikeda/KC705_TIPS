import os
import sys
import glob

import numpy as np

import ctypes
from numba import jit

import matplotlib as mp
import matplotlib.pyplot as plt
import tkinter
mp.use('tkagg')


# debug
#from memory_profiler import profile
import time

################ISSUE################
# プロットの軸とかを綺麗に
# ファイルの中身を全部描き切ると、プロット画面も消える
#####################################

# bytes
DATA_UNIT = 13

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

# bits
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


# use with just ==, on raw(104 bits) data
BITS_WORD_HEADER_UPPER = (0x01234567 << (
    BITS_SIZE_SPILLCOUNT + 4 + BITS_SIZE_BOARDID + BITS_SIZE_HEADER_LOWER))

# use with just ==, on raw(104 bits) data
BITS_WORD_HEADER_LOWER = 0x0123456789AB

# use with just ==, on raw(104 bits) data
BITS_WORD_FOOTER_UPPER = (0xAAAAAAAA << (
    BITS_SIZE_SPILLCOUNT + BITS_SIZE_EMCOUNT + BITS_SIZE_WRITECOUNT + BITS_SIZE_FOOTER_LOWER))

# use with just ==, on raw(104 bits) data
BITS_WORD_FOOTER_LOWER = 0xAB


# 104 bits, only the upper BITS_SIZE_HEADER_UPPER bit is filled with 1
BITS_MASK_HEADER_UPPER = ((2 ** BITS_SIZE_HEADER_UPPER - 1) <<
                          (BITS_SIZE_SPILLCOUNT + 4 + BITS_SIZE_BOARDID + BITS_SIZE_HEADER_LOWER))

# 104 bits, only the lower BITS_SIZE_HEADER_LOWER bit is filled with 1
BITS_MASK_HEADER_LOWER = 2 ** BITS_SIZE_HEADER_LOWER - 1

# 104 bits, only the upper BITS_SIZE_FOOTER_UPPER bit is filled with 1
BITS_MASK_FOOTER_UPPER = ((2 ** BITS_SIZE_FOOTER_UPPER - 1) << (BITS_SIZE_SPILLCOUNT +
                                                                BITS_SIZE_EMCOUNT + BITS_SIZE_WRITECOUNT + BITS_SIZE_FOOTER_LOWER))

# 104 bits, only the lower BITS_SIZE_FOOTER_LOWER bit is filled with 1
BITS_MASK_FOOTER_LOWER = 2 ** BITS_SIZE_FOOTER_LOWER - 1

# 104 bits, only the corresponding BITS_SIZE_SPILLCOUNT bit is filled with 1
BITS_MASK_SPILLCOUNT_HEADER = (2 ** BITS_SIZE_SPILLCOUNT - 1 <<
                               (4 + BITS_SIZE_BOARDID + BITS_SIZE_HEADER_LOWER))

# 104 bits, only the corresponding BITS_SIZE_SPILLCOUNT bit is filled with 1
BITS_MASK_SPILLCOUNT_FOOTER = (2 ** BITS_SIZE_SPILLCOUNT - 1 <<
                               (BITS_SIZE_EMCOUNT + BITS_SIZE_WRITECOUNT + BITS_SIZE_FOOTER_LOWER))


# clock
DELAY_BH1_TO_NEWHOD = 0

# clock
DELAY_BH2_TO_NEWHOD = 0

# clock
DELAY_OLDHOD_TO_NEWHOD = 0


# ns
CLOCK_TIME = 5


def header_or_not(data):
    if((data & (BITS_MASK_HEADER_UPPER | BITS_MASK_HEADER_LOWER)) == (BITS_WORD_HEADER_UPPER | BITS_WORD_HEADER_LOWER)):
        return True
    else:
        return False


def footer_or_not(data):
    if((data & (BITS_MASK_FOOTER_UPPER | BITS_MASK_FOOTER_LOWER)) == (BITS_WORD_FOOTER_UPPER | BITS_WORD_FOOTER_LOWER)):
        return True
    else:
        return False


def get_spillcount_header(data):
    return ((data & BITS_MASK_SPILLCOUNT_HEADER) >> (4 + BITS_SIZE_BOARDID + BITS_SIZE_HEADER_LOWER))


def get_spillcount_footer(data):
    return ((data & BITS_MASK_SPILLCOUNT_FOOTER) >> (BITS_SIZE_EMCOUNT + BITS_SIZE_WRITECOUNT + BITS_SIZE_FOOTER_LOWER))


def get_spill_num(file_path):
    # スピルの総数を知る
    # 要はmallocの代わり、気楽なPythonに任せてるだけ

    file_size = os.path.getsize(file_path)
    # [spillcount] : [offset, number of data in it]
    spillcount_and_offset_and_datanum = {}
    # offset (in the unit of 13bytes)
    offset = 0
    # number of readed data (in the unit of 13bytes)
    count_readed = 0

    header_flag = False
    with open(file_path, "rb") as f:
        while (data := f.read(DATA_UNIT)):
            data = int.from_bytes(data, 'big')

            if(header_or_not(data)):

                offset = count_readed

                # Headerが来たら
                spillcount_header = get_spillcount_header(data)
                # print(spillcount_header)

                header_flag = True

            elif(footer_or_not(data)):
                # Footerが来たら
                spillcount_footer = get_spillcount_footer(data)
                # print(spillcount_footer)

                if (header_flag):

                    # ヘッダーを既に読んでいれば
                    # 0 1 2 3 4
                    # t h t t f
                    # offset = 1
                    # data_num = 1 (= 4-1-1)
                    spillcount_and_offset_and_datanum[spillcount_header] = [
                        offset, count_readed - offset]
                    header_flag = False

            else:
                pass

            count_readed += 1

    print('len(spillcount_and_offset_and_datanum): ' + str(len(spillcount_and_offset_and_datanum)) +
          ' spills in file: ' + str(file_path))
    print('spillcount_and_offset_and_datanum: ' +
          str(spillcount_and_offset_and_datanum))
    return spillcount_and_offset_and_datanum


@jit('i8[:](i8[:],i8[:])', nopython=True)
def coincidence_p3(array_foo, array_bar):
    # both array of array_foo and array_bar must be sorted
    # array_fooをメインの検出器とする(fooとbarについて非対称なコードだから)
    # 二つの配列大きさが違っていても良い

    # コインシデンス幅、メインの検出器に対して0,+-1
    # nのとき、-n,-n+1,...,-1,0,1,...n-1,n
    # 0のとき、0
    back_and_forth = 1

    # fooのサイズに合わせれば良い
    array_coincidenced = np.empty_like(array_foo)

    size_foo = array_foo.size
    size_bar = array_bar.size

    i_foo = j_bar = k_coincidenced = 0
    while i_foo < size_foo and j_bar < size_bar:
        # 差分
        diff_bar_foo = array_bar[j_bar] - array_foo[i_foo]

        if (diff_bar_foo > back_and_forth):
            # fooは少し先にbarの気配を感じた！
            # fooは駆け足
            i_foo += 1
        elif (diff_bar_foo < -back_and_forth):
            # barは少し先にfooの気配を感じた！
            # barは駆け足
            j_bar += 1
        else:
            # fooのちかくにbarがいた！
            array_coincidenced[k_coincidenced] = array_foo[i_foo]
            # コインシデンス幅があるために、j_barをインクリメントしない、次のi_foo+1とも仲良いかもしれないから
            k_coincidenced += 1
            i_foo += 1

    return array_coincidenced[:k_coincidenced]


@jit('UniTuple(i8[:],2)(UniTuple(i8[:],2),UniTuple(i8[:],2))', nopython=True)
def coincidence_mrsync(tuple_tdc_foo_and_mrsync_foo, tuple_tdc_bar_and_mrsync_bar):
    # ここでも、fooをメインの検出器とする(コードの結果は非対称だから)
    # 重要なのは、インデックスの対応はあくまでAとBとで独立に満たせば十分ということ
    # 従って、入力: (tdc(foo),mrsync(foo)),(tdc(bar),mrsync(bar))
    # とし、出力: (tdc(coincidenced),mrsync(coincidenced))
    # と、2入力1出力のすべての形式を揃えることで、coincidence(C,coincidene(A,B))のように書くことができる
    # ここで、tdc(A)をあらわに書くと、np.extract((sig & A != 0), tdc)、mrsyncについても同様
    # 引数はタプルのタプル、再帰に使うのはrecursionで良い

    # コインシデンス幅、メインの検出器に対して0,+-1
    # nのとき、-n,-n+1,...,-1,0,1,...n-1,n
    # 0のとき、0
    back_and_forth = 1

    tdc_foo = tuple_tdc_foo_and_mrsync_foo[0]
    mrsync_foo = tuple_tdc_foo_and_mrsync_foo[1]
    tdc_bar = tuple_tdc_bar_and_mrsync_bar[0]
    mrsync_bar = tuple_tdc_bar_and_mrsync_bar[1]

    # fooのサイズに合わせれば良い
    tdc_coincidenced = np.empty_like(tdc_foo)
    mrsync_coincidenced = np.empty_like(mrsync_foo)

    # tdcとmrsyncとでサイズは同じなのでどちらでも構わないが
    size_foo = tdc_foo.size
    size_bar = tdc_bar.size

    i_foo = j_bar = k_coincidenced = 0
    # どちらか一方の配列を走査し終えた時点で、他方の配列の残りを見てもしょうがないから
    # Notes: while X and Y: と書いたとき、XあるいはYのいずれか一方が満たされなくなったらループを抜ける
    while i_foo < len(tdc_foo) and j_bar < len(tdc_bar):

        # p3基準ならtdcの値は狭義の単調増加なため一意に定まるから、比較がしやすい
        # 従って、p3基準でコインシデンス幅におさまるかを見てから、mrsyncの分を差っ引く作戦
        diff_bar_foo = tdc_bar[j_bar] - tdc_foo[i_foo]

        if (diff_bar_foo > back_and_forth):
            # fooは少し先にbarの気配を感じた！
            # fooは駆け足
            i_foo += 1
        elif (diff_bar_foo < -back_and_forth):
            # barは少し先にfooの気配を感じた！
            # barは駆け足
            j_bar += 1
        else:
            # fooのちかくにbarがいた！
            tdc_coincidenced[k_coincidenced] = tdc_foo[i_foo]
            mrsync_coincidenced[k_coincidenced] = mrsync_foo[i_foo]
            # コインシデンス幅があるために、j_barをインクリメントしない、次のi_foo+1とも仲良いかもしれないから
            k_coincidenced += 1
            i_foo += 1

    return tdc_coincidenced[:k_coincidenced], mrsync_coincidenced[:k_coincidenced]


def recursion(function, tuple_argument):
    # 再帰処理をさせるため
    # 関数fと引数の集合A,B,Cに対して、f(f(A,B),C)、fは対称律を満たしていないから、この順序は重要
    result = function(tuple_argument[0], tuple_argument[1])
    # tuple_argumentに2個しかない場合、iに空が入るのを防ぐため
    if (len(tuple_argument) > 2):
        for i in range(2, len(tuple_argument)):
            result = function(result, tuple_argument[i])
    return result

class caliculation(object):

    def __init__(self):
        pass

    def initialization(self, data_num):
        # input: number of data (in the unit of 13 bytes)

        sig_mppc = np.zeros(data_num, dtype='u8')
        sig_mppc = np.ascontiguousarray(sig_mppc)
        print('sig_mppc.dtype: ' + str(sig_mppc.dtype))
        tdc_mppc = np.zeros(data_num, dtype='i8')
        tdc_mppc = np.ascontiguousarray(tdc_mppc)
        print('tdc_mppc.dtype: ' + str(tdc_mppc.dtype))
        mrsync_mppc = np.zeros(data_num, dtype='i8')
        mrsync_mppc = np.ascontiguousarray(mrsync_mppc)
        print('mrsync_mppc.dtype: ' + str(mrsync_mppc.dtype))

        sig_pmt = np.zeros(data_num, dtype='u2')
        sig_pmt = np.ascontiguousarray(sig_pmt)
        print('sig_pmt.dtype: ' + str(sig_pmt.dtype))
        tdc_pmt = np.zeros(data_num, dtype='i8')
        tdc_pmt = np.ascontiguousarray(tdc_pmt)
        print('tdc_pmt.dtype: ' + str(tdc_pmt.dtype))
        mrsync_pmt = np.zeros(data_num, dtype='i8')
        mrsync_pmt = np.ascontiguousarray(mrsync_pmt)
        print('mrsync_pmt.dtype: ' + str(mrsync_pmt.dtype))

        sig_mrsync = np.zeros(data_num, dtype='u1')
        sig_mrsync = np.ascontiguousarray(sig_mrsync)
        print('sig_mrsync.dtype: ' + str(sig_mrsync.dtype))
        mrsync = np.zeros(data_num, dtype='i8')
        mrsync = np.ascontiguousarray(mrsync)
        print('mrsync.dtype: ' + str(mrsync.dtype))

        # indexes[0] = tmp_index_mppc;
        # indexes[1] = tmp_index_pmt;
        # indexes[2] = tmp_index_mrsync;

        indexes = np.zeros(3, dtype='u4')
        indexes = np.ascontiguousarray(indexes)
        print('indexes.dtype: ' + str(indexes.dtype))

        return sig_mppc, tdc_mppc, mrsync_mppc, \
            sig_pmt, tdc_pmt, mrsync_pmt, \
            sig_mrsync, mrsync, indexes

    def setting(self):
        # loader.soを登録
        c_library = np.ctypeslib.load_library("loader.so", ".")

        # void a_spill_loader(unsigned long long *sig_mppc, signed long long *tdc_mppc, signed long long *mrsync_mppc,
        #                     unsigned short *sig_pmt, signed long long *tdc_pmt, signed long long *mrsync_pmt,
        #                     unsigned char *sig_mrsync, signed long long *mrsync, unsigned int *indexes,
        #                     char *file_path, long offset_to_read, long data_num)

        # a_spill_loader()関数の引数の型を指定(ctypes)　
        c_library.a_spill_loader.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
                                             ctypes.POINTER(ctypes.c_uint16),  ctypes.POINTER(
                                                 ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
                                             ctypes.POINTER(ctypes.c_uint8),  ctypes.POINTER(
                                                 ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32),
                                             ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]

        # a_spill_loader()関数が返す値の型を指定(今回は返り値なし)
        c_library.a_spill_loader.restype = None

        return c_library

    def shortening(self, sig_mppc, tdc_mppc, mrsync_mppc, sig_pmt, tdc_pmt, mrsync_pmt, sig_mrsync, mrsync, indexes):
        # 必要な配列を抜き出して
        # indexes[0] = tmp_index_mppc;
        # indexes[1] = tmp_index_pmt;
        # indexes[2] = tmp_index_mrsync;

        index_mppc = indexes[0]
        index_pmt = indexes[1]
        index_mrsync = indexes[2]

        sig_mppc = sig_mppc[:index_mppc]
        tdc_mppc = tdc_mppc[:index_mppc]
        mrsync_mppc = mrsync_mppc[:index_mppc]

        sig_pmt = sig_pmt[:index_pmt]
        tdc_pmt = tdc_pmt[:index_pmt]
        mrsync_pmt = mrsync_pmt[:index_pmt]

        sig_mrsync = sig_mrsync[:index_mrsync]
        mrsync = mrsync[:index_mrsync]

        return sig_mppc, tdc_mppc, mrsync_mppc, sig_pmt, tdc_pmt, mrsync_pmt, sig_mrsync, mrsync

    def extracting(self, sig_mppc, tdc_mppc, mrsync_mppc, sig_pmt, tdc_pmt, mrsync_pmt):
        condition_newhod = (sig_mppc != 0)
        # 0bに注意(0xでない、一敗)
        condition_bh1 = ((sig_pmt & 0b001) != 0)
        condition_bh2 = ((sig_pmt & 0b010) != 0)
        condition_oldhod = ((sig_pmt & 0b100) != 0)

        # extractして
        newhod = np.extract(condition_newhod, tdc_mppc)
        mrsync_newhod = np.extract(condition_newhod, mrsync_mppc)

        bh1 = np.extract(condition_bh1, tdc_pmt)
        mrsync_bh1 = np.extract(condition_bh1, mrsync_pmt)

        bh2 = np.extract(condition_bh2, tdc_pmt)
        mrsync_bh2 = np.extract(condition_bh2, mrsync_pmt)

        oldhod = np.extract(condition_oldhod, tdc_pmt)
        mrsync_oldhod = np.extract(condition_oldhod, mrsync_pmt)

        return newhod, mrsync_newhod, bh1, mrsync_bh1, bh2, mrsync_bh2, oldhod, mrsync_oldhod

    def chmapping(self, sig_mppc, sig_pmt):
        chmap_mppc32ch = np.zeros((2, 32), dtype='u4')
        for ch_i in range(2):
            for ch_j in range(32):
                # 実際の配置に合わせるには、ここにch-sig_mppcを対応させる辞書を一冊かませる
                condition_jch = ((sig_mppc & (0b1 << (ch_j + ch_i * 32)) != 0))
                chmap_mppc32ch[ch_i][ch_j] = np.count_nonzero(condition_jch)

        # print(chmap_mppc32ch)

        chmap_pmt16ch = np.zeros((1, 16), dtype='u4')
        for ch_i in range(1):
            for ch_j in range(16):
                # 実際の配置に合わせるには、ここにch-sig_mppcを対応させる辞書を一冊かませる
                condition_jch = ((sig_pmt & (0b1 << (ch_j + ch_i * 16)) != 0))
                chmap_pmt16ch[ch_i][ch_j] = np.count_nonzero(condition_jch)

        return chmap_mppc32ch, chmap_pmt16ch


class plotter(object):

    def __init__(self):
        self.fig = plt.figure(figsize=(10, 8))
        # 画面サイズ
        #self.file_name = 'WAIT PLEASE ;-)'
        self.file_name = "/Users/f-ikeda/COMET_TOOLS/daq-py/monitor_for/1MHz.dat"

        self.initializer()

    def initializer(self):
        self.fig.suptitle('monitor' + '\n' + 'file: ' +
                          self.file_name, size=10)
        # タイトルと文字サイズ
        self.fig.subplots_adjust(
            left=0.10, right=0.90, top=0.90, bottom=0.1, wspace=0.4, hspace=0.4)
        # グラフ全体としての画面端との余白(左,右,上,下), 各グラフ間の余白(左右,上下)
        self.ax_mrsync = plt.subplot2grid((2, 2), (0, 1))
        # 画面の分割数(行,列), そのグラフの置き場(通し番号)
        self.ax_p3 = plt.subplot2grid((2, 2), (1, 1))
        self.ax_chmap_mppc = plt.subplot2grid((2, 2), (0, 0))
        self.ax_chmap_pmt = plt.subplot2grid((2, 2), (1, 0))

        self.ax_mrsync.grid(True)
        # プロット中のマス目の有無
        self.ax_p3.grid(True)
        self.ax_chmap_mppc.grid(True)
        self.ax_chmap_pmt.grid(True)

        self.ax_mrsync.set_title(
            "New Hod.'s All Or & BH1 & BH2 & Old Hod.'s All Or")
        # タイトル
        self.ax_p3.set_title(
            "New Hod.'s All Or & BH1 & BH2 & Old Hod.'s All Or")
        self.ax_chmap_mppc.set_title("New Hod. 32CH MAP")
        self.ax_chmap_pmt.set_title("Old Hod. 12CH MAP")

        self.ax_mrsync.set_xlabel('time from MR Sync [us]')
        # x軸のラベル
        self.ax_mrsync.set_ylabel('events/bin')
        self.ax_p3.set_xlabel('time from P3 [s]')
        self.ax_p3.set_ylabel('events/bin')

        self.lines_mrsync = self.ax_mrsync.hist([-1, -1], label='from MR Sync')
        self.lines_p3 = self.ax_p3.hist([-1, -1], label='from P3')
        self.ax_chmap_mppc.imshow(
            np.zeros((2, 32)), interpolation='nearest', vmin=0, vmax=1, cmap='jet', aspect=4.00)
        self.ax_chmap_pmt.imshow(
            np.zeros((2, 12)), interpolation='nearest', vmin=0, vmax=1, cmap='jet', aspect=4.00)
        # プロットの初期化

    def reloader(self, data_array_mrsync, data_array_p3, chmap_mppc32ch, chmap_pmt16ch, spillcount):
        # 更新ごとにあれこれ設定
        self.fig.suptitle('monitor' + '\n' + 'file: ' +
                          self.file_name, size=10)

        self.ax_mrsync.cla()
        self.ax_mrsync.set_title(
            "New Hod.'s All Or & BH1 & BH2 & Old Hod.'s All Or" + '\n' + 'Spill No.' + str(spillcount))
        self.ax_mrsync.set_xlabel('time from MR Sync [us]')
        self.ax_mrsync.set_ylabel('events/bin')
        self.ax_mrsync.grid(True)
        self.lines_mrsync = self.ax_mrsync.hist(data_array_mrsync*CLOCK_TIME*pow(
            10, -3), bins=250, histtype='step', log=True, label='from MR Sync')
        self.ax_mrsync.set_ylim(0.1, None)
        self.ax_mrsync.legend(loc='upper right')

        self.ax_p3.cla()
        self.ax_p3.set_title(
            "New Hod.'s All Or & BH1 & BH2 & Old Hod.'s All Or" + '\n' + 'Spill No.' + str(spillcount))
        self.ax_p3.set_xlabel('time from P3 [s]')
        self.ax_p3.set_ylabel('events/bin')
        self.ax_p3.grid(True)
        self.lines_p3 = self.ax_p3.hist(
            data_array_p3*CLOCK_TIME*pow(10, -9), bins=250, histtype='step', log=True, label='from P3')
        self.ax_p3.set_ylim(0.1, None)
        self.ax_p3.legend(loc='upper right')

        self.ax_chmap_mppc.cla()
        self.ax_chmap_mppc.set_title("New Hod. 32CH MAP")
        self.ax_chmap_mppc.grid(True)
        self.ax_chmap_mppc.imshow(chmap_mppc32ch, interpolation='nearest',
                                  vmin=0, vmax=1, cmap='jet', aspect=4.00)

        self.ax_chmap_pmt.cla()
        self.ax_chmap_pmt.set_title("Old Hod. 12CH MAP")
        self.ax_chmap_pmt.grid(True)
        self.ax_chmap_pmt.imshow(chmap_pmt16ch, interpolation='nearest',
                                 vmin=0, vmax=1, cmap='jet', aspect=3.00)

    def pauser(self, second):
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


# @profile
def main(CALICUTAION, PLOTTER, file_path):

    # スピルカウントとオフセットを取得、mallocの代わり、せこい！
    T_START = time.time()
    spillcount_and_offset_and_datanum = get_spill_num(file_path)
    T_END = time.time()
    print('TIME [s]: ' + str(T_END - T_START))

    # [spillcount] : [offset, number of data in it]
    for spill_i in spillcount_and_offset_and_datanum:
        T_START = time.time()

        # 空の配列を生成
        sig_mppc, tdc_mppc, mrsync_mppc, \
            sig_pmt, tdc_pmt, mrsync_pmt, \
            sig_mrsync, mrsync, indexes = CALICULATION.initialization(
                spillcount_and_offset_and_datanum[spill_i][1])

        offset = np.zeros(2,dtype='i4')
        # 謎の仕様により、要素１の配列の受け渡しはなぜかできない
        offset[0] = spillcount_and_offset_and_datanum[spill_i][0]
        # Cでのループ回数の上限を念のため与えておく
        data_num = np.zeros(2,dtype='i4')
        data_num[0] = spillcount_and_offset_and_datanum[spill_i][1]

        # void a_spill_loader(unsigned long long *sig_mppc, signed long long *tdc_mppc, signed long long *mrsync_mppc, unsigned int index_mppc,
        #                     unsigned short *sig_pmt, signed long long *tdc_pmt, signed long long *mrsync_pmt, unsigned int index_pmt,
        #                     unsigned char *sig_mrsync, signed long long *mrsync, unsigned int index_mrsync,
        #                     char *file_path, unsigned long offset_to_read)
        # 1 spill分を読み込み
        c_library.a_spill_loader(np.ctypeslib.as_ctypes(sig_mppc), np.ctypeslib.as_ctypes(tdc_mppc), np.ctypeslib.as_ctypes(mrsync_mppc),
                                 np.ctypeslib.as_ctypes(sig_pmt), np.ctypeslib.as_ctypes(
                                     tdc_pmt), np.ctypeslib.as_ctypes(mrsync_pmt),
                                 np.ctypeslib.as_ctypes(sig_mrsync), np.ctypeslib.as_ctypes(
                                     mrsync), np.ctypeslib.as_ctypes(indexes),
                                 buf, np.ctypeslib.as_ctypes(offset), np.ctypeslib.as_ctypes(data_num))

        print('spillcount: ' + str(spill_i) + '------------------------')

        sig_mppc, tdc_mppc, mrsync_mppc, sig_pmt, tdc_pmt, mrsync_pmt, sig_mrsync, mrsync = CALICULATION.shortening(
            sig_mppc, tdc_mppc, mrsync_mppc, sig_pmt, tdc_pmt, mrsync_pmt, sig_mrsync, mrsync, indexes)
        '''
        print("after sig_mppc: ", sig_mppc)
        print("after tdc_mppc: ", tdc_mppc)
        print("after mrsync_mppc: ", mrsync_mppc)

        print("after sig_pmt:", sig_pmt)
        print("after tdc_pmt:", tdc_pmt)
        print("after mrsync_pmt:", mrsync_pmt)

        print("after sig_mrsync:", sig_mrsync)
        print("after mrsync:", mrsync)
        print("after indexes: ", indexes)
        #'''

        newhod, mrsync_newhod, bh1, mrsync_bh1, bh2, mrsync_bh2, oldhod, mrsync_oldhod = CALICULATION.extracting(
            sig_mppc, tdc_mppc, mrsync_mppc, sig_pmt, tdc_pmt, mrsync_pmt)

        '''
        print('newhod: ', newhod)
        print('mrsync_newhod: ', mrsync_newhod)
        print('bh1: ', bh1)
        print('mrsync_bh1: ', mrsync_bh1)
        print('bh2: ', bh2)
        print('mrsync_bh2: ', mrsync_bh2)
        print('oldhod: ', oldhod)
        print('mrsync_oldhod: ', mrsync_oldhod)
        #'''

        # コインシデンスとって
        coinci_p3 = recursion(coincidence_p3, tuple(
            [newhod-0, bh1-DELAY_BH1_TO_NEWHOD, bh2-DELAY_BH2_TO_NEWHOD, oldhod-DELAY_OLDHOD_TO_NEWHOD]))

        tmp_coinci_tdc, tmp_coinci_mrsync = recursion(coincidence_mrsync, tuple(
            [tuple([newhod-0, mrsync_newhod]),
             tuple([bh1-DELAY_BH1_TO_NEWHOD,
                    mrsync_bh1]),
             tuple([bh2-DELAY_BH2_TO_NEWHOD,
                    mrsync_bh2]),
             tuple([oldhod-DELAY_OLDHOD_TO_NEWHOD, mrsync_oldhod])]))
        coinci_mrsync = tmp_coinci_tdc - tmp_coinci_mrsync

        # chmapつくって
        chmap_mppc32ch, chmap_pmt16ch = CALICULATION.chmapping(
            sig_mppc, sig_pmt)

        T_END = time.time()
        print('TIME [s]: ' + str(T_END - T_START))

        # ここでプロッターに渡す
        PLOTTER.reloader(coinci_mrsync, coinci_p3,
                         chmap_mppc32ch, chmap_pmt16ch, spill_i)
        PLOTTER.pauser(0.1)

    return


if __name__ == '__main__':

    argument = sys.argv
    path_to_directory = argument[1]
    if (argument != 1):
        print("USEAGE: $ python3 monitor_v2.py path_to_directory" )

    # なんか
    CALICULATION = caliculation()

    # cモジュールをロード
    c_library = CALICULATION.setting()
    # 大きめに1024*5
    buf = ctypes.create_string_buffer(1024*5)


    # お絵かき
    PLOTTER = plotter()
    while (True):
        while(len(PLOTTER.finder(path_to_directory)) == 0):
            print('NO FILE ;-)')
            time.sleep(0.3)

        path_to_file = PLOTTER.finder(path_to_directory)
        buf.value = bytes(path_to_file, encoding='utf-8')
        PLOTTER.file_name = path_to_file

        main(CALICULATION, PLOTTER, path_to_file)



