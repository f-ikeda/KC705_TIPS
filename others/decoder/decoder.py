import sys
import os

import numpy as np

from numba import jit

import matplotlib as mp
import matplotlib.pyplot as plt

# ########DESCRIPTION########
# USEAGE: $ python3 moniter.py path_to_directory
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
# ISSUES:
#
# 仕様: ファイル中に、HeaderとFooterとが、同数かつHeader,Footer,...,Header,Footerの順に、必ず1 組以上含まれていなければいけない
#       ファイル中に、MR Syncが必ず1 つ以上含まれていなければいけない
#       Headerよりも後、かつ、MR Syncよりも前にイベントがあった場合、mrsyncに前回のスピルの最後のMR Syncの値を割り当てる

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

DELAY_BH1_TO_NEWHOD = 20
# clock
DELAY_BH2_TO_NEWHOD = 21
# clock
DELAY_OLDHOD_TO_NEWHOD = 29
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


def processing_spillcount(data):
    # ----SPILLCOUNT----
    condition_header = ((data & (BITS_MASK_HEADER_UPPER | BITS_MASK_HEADER_LOWER)) == (
        BITS_WORD_HEADER_UPPER | BITS_WORD_HEADER_LOWER))
    # making the boolian mask
    condition_footer = ((data & (BITS_MASK_FOOTER_UPPER | BITS_MASK_FOOTER_LOWER)) == (
        BITS_WORD_FOOTER_UPPER | BITS_WORD_FOOTER_LOWER))
    # making the boolian mask
    index_header = np.where(condition_header)
    # getting the position of the Header
    index_footer = np.where(condition_footer)
    # getting the position of the Footer
    list_spillcount = (np.extract(condition_header, data)
                       & BITS_MASK_SPILLCOUNT_HEADER)
    # getting the list of the Spillcount
    spillcount = np.concatenate([np.full(index_header[0][0], -1), np.repeat(
        list_spillcount, np.diff(index_header[0], append=data.size))])
    # when there are no Header in file, index_header[0][0] causes an error

    return spillcount, index_header[0], index_footer[0], condition_header, condition_footer, list_spillcount


def processing_sig(data):
    # ----SIG----
    sig = (data & BITS_MASK_SIG) >> BITS_SIZE_TDC

    return sig


@ jit('i8[:](i8[:],i8[:],i8[:])', nopython=True)
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
            tdc[array_i[index_k]: array_i[index_k+1]] = tdc[array_i[index_k]
                : array_i[index_k+1]] + (index_k+1) * 2 ** 27

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


argument = sys.argv
if(len(argument) != 2):
    print(argument)
    print('$ python3 moniter.py path_to_file')
    sys.exit()

path_to_file = argument[1]

bytes_to_int_universal = np.frompyfunc(bytes_to_int, 1, 1)
# converting function to universal function

# --------READING--------
file_size = os.path.getsize(path_to_file)
file_size = (int)(file_size/DATA_UNIT)
with open(path_to_file, 'rb') as file:
    data_bytes = file.read(file_size*DATA_UNIT)

data = formatting_data(data_bytes)

# --------PROCESSING--------
# ----SPILLCOUNT----
spillcount, index_header, index_footer, condition_header, condition_footer, list_spillcount = processing_spillcount(
    data)
print('index_header.size: ' + str(index_header.size))
print('index_header: ' + str(index_header))
print('index_footer.size: ' + str(index_footer.size))
print('index_footer: ' + str(index_footer))
# print('list_spillcount.size: ' + str(list_spillcount.size))
# print('list_spillcount: ' + str(list_spillcount))
# ----SIG----
sig = processing_sig(data)
# ----TDC----
#tdc = processing_tdc(data, index_header, index_footer)
# ----MR SYNC----
# mrsync, condition_mrsync, list_mrsync = processing_mrsync(
#    sig, condition_header, condition_footer, tdc)

# --------ANALYZING--------
# ########Write the analysis code here using sig, tdc, mrsync and spillcount########
# condition_somedetector = bit-calc.(sig)
# tdc_somedetector_p3 = np.extract(condition_somedetector, tdc)
# tdc_somedetector_mrsync = np.extract(condition_somedetector, tdc - mrsync)

spillcount_header = np.extract(condition_header, (data & BITS_MASK_SPILLCOUNT_HEADER) >> (
    4 + BITS_SIZE_BOARDID + BITS_SIZE_HEADER_LOWER))
spillcount_footer = np.extract(condition_footer, (data & BITS_MASK_SPILLCOUNT_FOOTER) >> (
    BITS_SIZE_EMCOUNT + BITS_SIZE_WRITECOUNT + BITS_SIZE_FOOTER_LOWER))

if (spillcount_header.size > spillcount_footer.size):
    spillcount_header = spillcount_header[:spillcount_footer.size]
else:
    spillcount_footer = spillcount_footer[:spillcount_header.size]

print('spillcount_header.size: ' + str(spillcount_header))
print('spillcount_footer.size: ' + str(spillcount_footer))

print('spillcount_header: ' + str(spillcount_header))
print('spillcount_footer: ' + str(spillcount_footer))
spillcount_check = ((spillcount_header - spillcount_footer) == 0)
print('spillcount_header == spillcount_footer ?: ' + str(spillcount_check))
print('np.where(spillcount_check == False): ' +
      str(np.where(spillcount_check == False)))
print('np.where(np.diff(spillcount_header) != 1))' +
      str(np.where((np.diff(spillcount_header) != 1) & (-np.diff(spillcount_header) != 0xFFFF))))
print('np.where(np.diff(spillcount_footer) != 1))' +
      str(np.where((np.diff(spillcount_footer) != 1) & (-np.diff(spillcount_footer) != 0xFFFF))))
# ##################################################################################

'''
# --------DRAWING--------
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.hist(np.diff(np.extract(condition_mrsync, mrsync)),
        bins=250, histtype='step', log=True)
ax.set_ylim(0.1, None)
plt.show()
'''
