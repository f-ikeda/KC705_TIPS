import sys

import numpy as np
from functools import partial, reduce

import matplotlib.pyplot as plt

import time
# for debug

# ########DESCRIPTION########
# USEAGE: $ python3 moniter.py how_many_spill_to_read path_to_beamOn-10shoot-20201217.dat
#         $ python3 monitor.py 1 beamOn-10shoot-20201217.dat (1 spill)
#         $ python3 monitor.py 2 beamOn-10shoot-20201217.dat (2 spill)
#                                       :
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
# ISSUES: コインシデンスの処理が未実装(indexがクロックカウントに等しい配列を、コインシデンスを撮りたい分だけ用意して、各要素のminをとる)
#         データの読み込み、定期的に(how?)ディレクトリの中にある最新のものを探して(globでおk)、勝手に読むようにする
#         12月のデータで動作確認をしているため、HeaderとFooterの仕様が古い
#
# 仕様: ファイル中に、HeaderとFooterとが、同数かつHeader,Footer,...,Header,Footerの順に、必ず1 組以上含まれていなければいけない
#       ファイル中に、MR Syncが必ず1 つ以上含まれていなければいけない
#       Headerよりも後、かつ、MR Syncよりも前にイベントがあった場合、mrsyncに前回のスピルの最後のMR Syncの値を割り当てる

DATA_UNIT = 13
# bytes
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
BITS_SIZE_HEADER = 84
# bits
BITS_SIZE_FOOTER = 84
# bits

BITS_WORD_HEADER = 0xABB000123456701234567 << (
    BITS_SIZE_BOARDID + BITS_SIZE_SPILLCOUNT)
# use with just ==, on raw(104 bits) data
BITS_WORD_FOOTER = 0xFEE00AAAAAAAAAAAAAAAA
# use with just ==, on raw(104 bits) data

BITS_MASK_HEADER = (2 ** BITS_SIZE_HEADER -
                    1) << (BITS_SIZE_BOARDID + BITS_SIZE_SPILLCOUNT)
# 104 bits, only the upper BITS_SIZE_HEADER bit is filled with 1
BITS_MASK_FOOTER = 2 ** BITS_SIZE_FOOTER - 1
# 104 bits, only the lower BITS_SIZE_FOOTER bit is filled with 1
BITS_MASK_SPILLCOUNT = 2 ** BITS_SIZE_SPILLCOUNT - 1
# 104 bits, only the lower BITS_SIZE_SPILLCOUNT bit is filled with 1
BITS_MASK_SIG = (2 ** BITS_SIZE_SIG - 1) << BITS_SIZE_TDC
# 104 bits, only the upper BITS_SIZE_SIG bit is filled with 1
BITS_MASK_TDC = 2 ** BITS_SIZE_TDC - 1
# 104 bits, only the lower BITS_SIZE_TDC bit is filled with 1

BITS_MASK_SIG_NEWHOD_ALLOR = (2 ** BITS_SIZE_SIG_NEWHOD - 1) - \
    (2 ** (BITS_SIZE_SIG_MRSYNC + BITS_SIZE_SIG_PMT) - 1)
# 77 bits, only the upper BIT_SIZE_SIG_NEWHOD bit is filled with 1
BITS_MASK_SIG_MRSYNC = 2 ** BITS_SIZE_SIG_MRSYNC - 1
# 77 bits, only the lower BIT_SIZE_SIG_MRSYNC bit is filled with 1

CLOCK_TIME = 5
# ns

DATA_TYPE = np.dtype((np.void, DATA_UNIT))


def bytes_to_int(DEADBEEF):
    return int.from_bytes(DEADBEEF, 'big')


def formatting_data(data_bytes):
    data = np.frombuffer(data_bytes, DATA_TYPE)
    data = bytes_to_int_universal(data)

    return data


def processing_spillcount():
    # ----SPILLCOUNT----
    condition_header = ((data & BITS_MASK_HEADER) == BITS_WORD_HEADER)
    # making the boolian mask
    condition_footer = ((data & BITS_MASK_FOOTER) == BITS_WORD_FOOTER)
    # making the boolian mask
    index_header = np.where(condition_header)
    # getting the position of the Header
    index_footer = np.where(condition_footer)
    # getting the position of the Footer
    list_spillcount = (np.extract(condition_header, data)
                       & BITS_MASK_SPILLCOUNT)
    # getting the list of the Spillcount
    spillcount = np.concatenate([np.full(index_header[0][0], -1), np.repeat(
        list_spillcount, np.diff(index_header[0], append=data.size))])
    # when there are no Header in file, index_header[0][0] causes an error

    return spillcount, index_header, index_footer, condition_header, condition_footer, list_spillcount


def processing_sig():
    # ----SIG----
    sig = (data & BITS_MASK_SIG) >> BITS_SIZE_TDC

    return sig


def processing_tdc():
    # ----TDC----
    tdc = data & BITS_MASK_TDC

    # ----TDC CLOCK COUNT OVERFLOW----
    iter_index_header = iter(index_header[0])
    iter_index_fotter = iter(index_footer[0])
    index_header_and_footer = np.array(
        [i for i in zip(iter_index_header, iter_index_fotter)])

    index_within_a_spill = [tdc[array_i[0]+1:array_i[1]]
                            for array_i in index_header_and_footer]
    # +1 for the first TDC data from Header
    index_tdcdiff_within_a_spill = [np.where(np.diff(array_i) < 0)[
        0] for array_i in index_within_a_spill]
    # np.diff() returns local index in input

    index_overflow_and_footer = [np.insert(array_i+index_header[0][i]+1+1, array_i.size, index_footer[0][index_k])
                                 for array_i, i, index_k in zip(index_tdcdiff_within_a_spill, range(index_header[0].size), range(index_footer[0].size))]
    # rewrite local index as global index in all data
    for array_i in index_overflow_and_footer:
        for index_k in range(len(array_i)-1):
            tdc[array_i[index_k]:array_i[index_k+1]] = tdc[array_i[index_k]:array_i[index_k+1]] + (index_k+1) * 2 ** 27
    # mapか何かで書き直せるはず

    return tdc


def processing_mrsync():
    # ----MR SYNC----
    condition_mrsync = ((sig & BITS_MASK_SIG_MRSYNC) == BITS_MASK_SIG_MRSYNC) & (
        ~condition_header) & (~condition_footer)
    # making the boolian mask
    index_mrsync = np.where(condition_mrsync)
    # getting the position of the MR Sync
    list_mrsync = np.extract(condition_mrsync, tdc)
    # getting the list of the MR Sync's TDC
    mrsync = np.concatenate([np.full(index_mrsync[0][0], -1), np.repeat(
        list_mrsync, np.diff(index_mrsync[0], append=data.size))])
    # when there are no MR Sync data in file, index_mrsync[0][0] causes an error

    return mrsync, condition_mrsync, list_mrsync


def removing_header_and_footer():
    # this function may not be necessary, because there are boolian masks, such as conditon_header and condition_footer

    # ----REMOVING HEADER AND FOOTER----
    spillcount = np.delete(spillcount, np.concatenate(
        [index_header[0], index_footer[0]]))
    # removing Header and Footer
    sig = np.delete(sig, np.concatenate([index_header[0], index_footer[0]]))
    # removing Header and Footer
    tdc = np.delete(tdc, np.concatenate([index_header[0], index_footer[0]]))
    # removing Header and Footer
    mrsync = np.delete(mrsync, np.concatenate(
        [index_header[0], index_footer[0]]))
    # removing Header and Footer


def coincidence(conditions):
    # ----COINCIDENCE----
    intersect1d_merge = partial(np.intersect1d, assume_unique=True)
    # in a given spill, the set of values of TDC is considered unique because it is monotonically increasing in a narrow sense

    for spill_k in list_spillcount:
        # coincidence has to be considered for each spill independently
        condition_spill_k = ((spillcount == spill_k) & ~
                             condition_header & ~condition_footer)
        tdc_coincidenced = reduce(intersect1d_merge, tuple(
            [np.extract(condition_i & condition_spill_k, tdc) for condition_i in conditions]))
        # ここで等号を用いるのはおかしい、スライスして代入するようにしろ
        # spillcount配列から、あるスピルについて[先頭:末尾]するべき先頭と末尾は決まるはず

    return tdc_coincidenced


argument = sys.argv
if(len(argument) != 3):
    print(argument)
    print('$ python3 moniter.py how_many_spill_to_read path_to_beamOn-10shoot-20201217.dat')
    sys.exit()

file_path = argument[2]
file = open(file_path, 'rb')

file.read(DATA_UNIT)
# beamOn-10shoot-20201217.datを用いるために、ファイル冒頭のフッターを読み飛ばす
# $ python3 monitor.py 3580598 beamOn-10shoot-20201217.dat (1 spill)
# $ python3 monitor.py 7177025 beamOn-10shoot-20201217.dat (2 spill)
#                               :
# header: 0       3580598 7177025  10808540 14407270 18044246 21633446 25225710 28865318 32431605
# footer: 3580597 7177024 10808539 14407269 18044245 21633445 25225709 28865318 32431604 欠損？

data_num = {1: 3580598, 2: 7177205, 3: 10808540, 4: 14407270,
            5: 18044246, 6: 21633446, 7: 25225710, 8: 28865318}
spill_num = int(argument[1])

print('-------- TIME --------')
# --------READING--------
TIME_READ_S = time.time()
data_bytes = file.read(DATA_UNIT * data_num[spill_num])
TIME_READ_F = time.time()
print("READ TIME [s]: " + str(TIME_READ_F - TIME_READ_S))

# --------FORMATTING--------
bytes_to_int_universal = np.frompyfunc(bytes_to_int, 1, 1)
# converting function to universal function

TIME_FORMAT_S = time.time()
data = formatting_data(data_bytes)
TIME_FORMAT_F = time.time()
print("FORMAT TIME [s]: " + str(TIME_FORMAT_F - TIME_FORMAT_S))

# --------PROCESSING--------
TIME_PROCESS_S = time.time()
# ----SPILLCOUNT----
spillcount, index_header, index_footer, condition_header, condition_footer, list_spillcount = processing_spillcount()
# ----SIG----
sig = processing_sig()
# ----TDC----
tdc = processing_tdc()
# ----MR SYNC----
mrsync, condition_mrsync, list_mrsync = processing_mrsync()
TIME_PROCESS_F = time.time()
print("PROCESS TIME [s]: " + str(TIME_PROCESS_F - TIME_PROCESS_S))

# --------ANALYZING--------
TIME_ANALYZE_S = time.time()
# ----COINCIDENCE----
# ここで、コインシデンスをとるような処理を書く
# array_foo = fromnumpyfunc(foo array_foo,2,1)でdef array_foo[foo]みたいに

# ########Write the analysis code here using sig, tdc, mrsync and spillcount########
condition_newhod_allor = ((sig & BITS_MASK_SIG_NEWHOD_ALLOR)
                          != 0) & ~condition_header & ~condition_footer
# making the boolian mask
newhod_allor = np.extract(condition_newhod_allor, tdc)
newhod_allor = coincidence((condition_newhod_allor, condition_newhod_allor))
# ##################################################################################
TIME_ANALYZE_F = time.time()
print("ANALYZE TIME [s]: " + str(TIME_ANALYZE_F - TIME_ANALYZE_S))

# --------DRAWING--------
TIME_DRAW_S = time.time()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.hist(newhod_allor*CLOCK_TIME*pow(10, -9),
        bins=250, histtype='step', log=True)
ax.set_ylim(0.1, None)
TIME_DRAW_F = time.time()
print("DRAW TIME [s]: " + str(TIME_DRAW_F - TIME_DRAW_S))
plt.show()

# ここに、chマップをみるための二次元ヒストグラムを書く
# fig_bar =

'''
print('-------- DEBUG --------')
print('data.dtype: ' + str(data.dtype))
print('sig.dtype: ' + str(sig.dtype))
print('tdc.dtype: ' + str(tdc.dtype))

print('sig.size: ' + str(sig.size))
print('tdc.size: ' + str(tdc.size))
print('mrsync.size: ' + str(mrsync.size))
print('spillcount.size: ' + str(spillcount.size))

print('mrsync: ' + str(mrsync))
print('spillcount: ' + str(spillcount))

print('list_mrsync: ' + str(list_mrsync))
print('list_spillcount: ' + str(list_spillcount))

print('np.unique(mrsync).size: ' +
      str(np.unique(mrsync).size))
print('np.unique(spillcount).size: ' +
      str(np.unique(spillcount).size))

print('np.unique(mrsync, return_index=True)[1]: ' + str(
    np.unique(mrsync, return_index=True)[1]))
print('np.unique(spillcount, return_index=True)[1]: ' + str(
    np.unique(spillcount, return_index=True)[1]))

print('index_header[0]: ' + str(index_header[0]))
print('index_footer[0]: ' + str(index_footer[0]))
print('index_mrsync[0]: ' + str(index_mrsync[0]))
'''

file.close()
