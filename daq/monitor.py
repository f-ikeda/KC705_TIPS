import sys

import numpy as np

import matplotlib.pyplot as plt

import time
# for debug

# ########DESCRIPTION########
# USEAGE: $ python3 moniter.py how_many_13bytes_chunk_to_read _path_to_data b(option: for loading December's data)
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
# OUTPUT ARRAY: sig        = [sig of all the corresponding tdc data]
#               tdc        = [123, 12a, 450, 541, 711, 957, 601, 123, 340, 601, 111]
#               mrsync     = [-1, 12a, 12a, 12a, 12a, 957, 957, 957, 340, 340, 340]
#               spillcount = [-1, 3272,3272,3272,3272,3272,3272,3273,3273,3273]
#
# ISSUES: コインシデンスの処理が未実装(indexがクロックカウントに等しい配列を、コインシデンスを撮りたい分だけ用意して、各要素のminをとる)
#         データの読み込み、定期的に(how?)ディレクトリの中にある最新のものを探して(globでおk)、勝手に読むようにする
#
# 仕様: ファイル中に、HeaderとFooterとが、同数かつHeader,Footer,...,Header,Footerの順に、必ず1 組以上含まれていなければいけない
#       ファイル中に、MR Syncが必ず1 つ含まれていなければいけない
#       Headerよりも後、かつ、MR Syncよりも

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

PLOT_LENGTH_FROM_MRSYNC = 6
# us

DATA_TYPE = np.dtype((np.void, DATA_UNIT))


def bytes_to_int(DEADBEEF):
    return int.from_bytes(DEADBEEF, 'big')


argument = sys.argv
if(len(argument) != 3):
    print(argument)
    print('USEAGE: $ python3 monitor.py number_of_13bytesdata_to_read path_to_datafile')
    sys.exit()

file_path = argument[2]
file = open(file_path, 'rb')

file.read(DATA_UNIT)
# beamOn-10shoot-20201217.datを用いるために、ファイル冒頭のフッターを読み飛ばす
# header [0]
# footer [3580597]
# 読み込むデータは3580598個
# header: 0       3580598 7177025  10808540 14407270 18044246 21633446 25225710 28865318 32431605
# footer: 3580597 7177024 10808539 14407269 18044245 21633445 25225709 28865318 32431604 欠損？


print('-------- TIME --------')
# --------READING--------
data_num = int(argument[1])
# number of tdc data to read
TIME_READ_S = time.time()
data_bytes = file.read(DATA_UNIT * data_num)
# str.hex(), only in python3
TIME_READ_F = time.time()
print("READ TIME [s]: " + str(TIME_READ_F - TIME_READ_S))

bytes_to_int_universal = np.frompyfunc(bytes_to_int, 1, 1)
# converting function to universal function

# --------FORMATTING--------
TIME_FORMAT_S = time.time()
data = np.frombuffer(data_bytes, DATA_TYPE)
data = bytes_to_int_universal(data)
TIME_FORMAT_F = time.time()
print("FORMAT TIME [s]: " + str(TIME_FORMAT_F - TIME_FORMAT_S))

# --------PROCESSING--------
TIME_PROCESS_S = time.time()
# ----SPILLCOUNT----
condition_header = ((data & BITS_MASK_HEADER) == BITS_WORD_HEADER)
# making the boolian mask
condition_footer = ((data & BITS_MASK_FOOTER) == BITS_WORD_FOOTER)
# making the boolian mask
header_index = np.where(condition_header)
# getting the position of the Header
footer_index = np.where(condition_footer)
# getting the position of the Footer
spillcount_list = (np.extract(condition_header, data)
                   & BITS_MASK_SPILLCOUNT)
# getting the list of the Spillcount
spillcount = np.concatenate([np.full(header_index[0][0], -1), np.repeat(
    spillcount_list, np.diff(header_index[0], append=data.size))])
# when there are no Header data in file, header_index[0][0] causes an error

# ----SIG----
sig = (data & BITS_MASK_SIG) >> BITS_SIZE_TDC

# ----TDC----
tdc = data & BITS_MASK_TDC

# ----TDC CLOCK COUNT OVERFLOW----

# h&fがセットでいることを前提とする
header_index_iter = iter(header_index[0])
footer_index_iter = iter(footer_index[0])

array_foo = np.array([i for i in zip(header_index_iter, footer_index_iter)])
print('array_foo: ' + str(array_foo))
print('array_foo[0]: ' + str(array_foo[0]))

array_bar = [tdc[i_array[0]+1:i_array[1]] for i_array in array_foo]
# headerの次から、のために+1。後ろはフッター含まずなのでそのまま
print('array_bar: ' + str(array_bar))

array_bar = [np.where(np.diff(i_array) < 0)[0] for i_array in array_bar]
print('array_bar: ' + str(array_bar))

array_bar = [np.insert(i_array+header_index[0][i]+1+1, i_array.size, footer_index[0][k])
             for i_array, i, k in zip(array_bar, range(header_index[0].size), range(footer_index[0].size))]
print('array_bar: ' + str(array_bar))

# array_bar = [tdc[j:j+1] + 2 **
#             27 for i_array in array_bar for j in range(len(i_array) - 1)]

for i_array in array_bar:
    for j in range(len(i_array)-1):
        tdc[i_array[j]:i_array[j+1]] = tdc[i_array[j]:i_array[j+1]] + (j+1) * 2 ** 27

# print('tdc: ' + str(tdc))

# ----MR SYNC----
condition_mrsync = ((sig & BITS_MASK_SIG_MRSYNC) == BITS_MASK_SIG_MRSYNC) & (
    ~condition_header) & (~condition_footer)
# making the boolian mask
mrsync_index = np.where(condition_mrsync)
# getting the position of the MR Sync
mrsync_list = np.extract(condition_mrsync, tdc)
# getting the list of the MR Sync's TDC
mrsync = np.concatenate([np.full(mrsync_index[0][0], -1),
                         np.repeat(mrsync_list, np.diff(mrsync_index[0], append=data.size))])
# when there are no MR Sync data in file, mrsync_index[0][0] causes an error
# P3より後かつMR Syncより前のイベントについて、一つ前のスピルでの最後のMR Syncを割り当ててしまう(そのようなイベントはないはず(？))

'''
# ----REMOVING HEADER AND FOOTER----
spillcount = np.delete(spillcount, np.concatenate(
    [header_index[0], footer_index[0]]))
# removing Header and Footer
sig = np.delete(sig, np.concatenate([header_index[0], footer_index[0]]))
# removing Header and Footer
tdc = np.delete(tdc, np.concatenate([header_index[0], footer_index[0]]))
# removing Header and Footer
mrsync = np.delete(mrsync, np.concatenate([header_index[0], footer_index[0]]))
# removing Header and Footer
'''
# these processes may not be necessary, because there are boolian masks, such as conditon_header and condition_footer

# ----COINCIDENCE----
# ここで、コインシデンスをとるような処理を書く
# array_foo = fromnumpyfunc(foo array_foo,2,1)でdef array_foo[foo]みたいに


# ########Write the analysis code here using sig, tdc, mrsync and spillcount########
condition_newhod_allor = ((sig & BITS_MASK_SIG_NEWHOD_ALLOR)
                          != 0) & ~condition_header & ~condition_footer
newhod_allor = np.extract(condition_newhod_allor, tdc - mrsync)
# ##################################################################################
TIME_PROCESS_F = time.time()
print("PROCESS TIME [s]: " + str(TIME_PROCESS_F - TIME_PROCESS_S))

# --------DRAWING--------
TIME_DRAW_S = time.time()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.hist(newhod_allor*5*pow(10, -3), bins=250, histtype='step', log=True)
ax.set_ylim(0.1, None)
TIME_DRAW_F = time.time()
print("DRAW TIME [s]: " + str(TIME_DRAW_F - TIME_DRAW_S))
plt.show()

# ここに、chマップをみるための二次元ヒストグラムを書く
# fig_bar =


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

print('mrsync_list: ' + str(mrsync_list))
print('spillcount_list: ' + str(spillcount_list))

print('np.unique(mrsync).size: ' +
      str(np.unique(mrsync).size))
print('np.unique(spillcount).size: ' +
      str(np.unique(spillcount).size))

print('np.unique(mrsync, return_index=True)[1]: ' + str(
    np.unique(mrsync, return_index=True)[1]))
print('np.unique(spillcount, return_index=True)[1]: ' + str(
    np.unique(spillcount, return_index=True)[1]))

print('header_index[0]: ' + str(header_index[0]))
print('footer_index[0]: ' + str(footer_index[0]))
print('mrsync_index[0]: ' + str(mrsync_index[0]))


file.close()
