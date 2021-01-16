import sys
import struct

import numpy as np

import matplotlib.pyplot as plt

import time
# for debug

# ########DESCRIPTION########
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
# OUTPUT: sig        = [sig of all the corresponding tdc data]
#         tdc        = [123, 12a, 450, 541, 711, 957, 601, 123, 340, 601, 111]
#         mrsync     = [-1, 12a, 12a, 12a, 12a, 957, 957, 957, 340, 340, 340] !!!ヘッダーと1つ目のMR Syncの間のイベントにも-1を対応させろ
#         spillcount = [-1, 3272,3272,3272,3272,3272,3272,3273,3273,3273]

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

BITS_MASK_HEADER = 0xABB000123456701234567
# use with >> (BITS_SIZE_BOARDID + BITS_SIZE_SPILLCOUNT), then &, on raw(104 bits) data
BITS_MASK_FOOTER = 0x0FEE00AAAAAAAAAAAAAAAA
# use with just &, on raw(104 bits) data
BITS_MASK_SPILLCOUNT = 2 ** BITS_SIZE_SPILLCOUNT - 1
# 104 bits, only the lower BITS_SIZE_SPILLCOUNT bit is filled with 1, use with just &, on raw(104 bits) data

BITS_MASK_SIG = (2 ** BITS_SIZE_SIG - 1) << BITS_SIZE_TDC
# 104 bits, only the upper BITS_SIZE_SIG bit is filled with 1
BITS_MASK_TDC = 2 ** BITS_SIZE_TDC - 1
# 104 bits, only the lower BITS_SIZE_TDC bit is filled with 1

BITS_MASK_SIG_NEWHOD_ALLOR = (2 ** BITS_SIZE_SIG_NEWHOD - 1) - \
    (2 ** (BITS_SIZE_SIG_MRSYNC + BITS_SIZE_SIG_PMT) - 1)
# 77 bits, only the upper BIT_SIZE_SIG_NEWHOD bit is filled with 1
BITS_MASK_SIG_MRSYNC = 2 ** BITS_SIZE_SIG_MRSYNC - 1
# 77 bits, only the lower BIT_SIZE_SIG_MRSYNC bit is filled with 1


argument = sys.argv
if(len(argument) != 3):
    print('USEAGE: $ python3 monitor.py number_of_13bytesdata_to_read path_to_datafile')
    sys.exit()

file_path = argument[2]
file = open(file_path, 'rb')

print('-------- TIME --------')
# --------READING--------
data_num = int(argument[1])
# number of tdc data to read
TIME_READ_S = time.time()
data_str = file.read(DATA_UNIT * data_num).hex()
# str.hex(), only in python3
TIME_READ_F = time.time()
print("READ TIME [s]: " + str(TIME_READ_F - TIME_READ_S))

# --------FORMATTING--------
TIME_FORMAT_S = time.time()
data = np.array([int('0x' + data_str[i:i+26], 16)
                 for i in range(0, len(data_str), 26)])
# あるいは、dtypeを自作してnp.frombuffer()を使う
TIME_FORMAT_F = time.time()
print("FORMAT TIME [s]: " + str(TIME_FORMAT_F - TIME_FORMAT_S))

# --------PROCESSING--------
TIME_PROCESS_S = time.time()

# ----SPILLCOUNT----
header_index = np.where(((data >> (
    BITS_SIZE_BOARDID + BITS_SIZE_SPILLCOUNT)) & BITS_MASK_HEADER) == BITS_MASK_HEADER)
# getting the position of the Header
footer_index = np.where(((data & BITS_MASK_FOOTER) == BITS_MASK_FOOTER))
# getting the position of the Footer
spillcount_list = np.array(data[header_index[0]] & BITS_MASK_SPILLCOUNT)
# getting the list of the Spillcount
spillcount = np.concatenate([np.full(header_index[0][0], -1), np.repeat(
    spillcount_list, np.diff(header_index[0], append=data.size))])
# when there are no Header data in file, header_index[0][0] causes an error
spillcount = np.delete(spillcount, np.concatenate(
    [header_index[0], footer_index[0]]))
# removing Header and Footer

# ----SIG----
sig = (data & BITS_MASK_SIG) >> BITS_SIZE_TDC
sig = np.delete(sig, np.concatenate([header_index[0], footer_index[0]]))
# removing Header and Footer

# ----TDC----
tdc = data & BITS_MASK_TDC
tdc = np.delete(tdc, np.concatenate([header_index[0], footer_index[0]]))
# removing Header and Footer

# ----MR Sync----
mrsync_index = np.where((sig & BITS_MASK_SIG_MRSYNC) == BITS_MASK_SIG_MRSYNC)
# getting the position of the MR Sync
mrsync_list = np.array(tdc[mrsync_index[0]])
# getting the list of the MR Sync
mrsync = np.concatenate([np.full(mrsync_index[0][0], -1),
                         np.repeat(mrsync_list, np.diff(mrsync_index[0], append=tdc.size))])
# when there are no Header data in file, header_index[0][0] causes an error

# ########Write the analysis code here using sig, tdc, mrsync and spillcount########
# something =
# ##################################################################################

TIME_PROCESS_F = time.time()
print("PROCESS TIME [s]: " + str(TIME_PROCESS_F - TIME_PROCESS_S))

'''
# --------DRAWING--------
TIME_DRAW_S = time.time()
plt.hist(something, histtype='step', log=True)
TIME_DRAW_F = time.time()
print("DRAW TIME [s]: " + str(TIME_DRAW_F - TIME_DRAW_S))
plt.show()
'''


print('-------- DEBUG --------')
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

print('header_index[0] in raw data: ' + str(header_index[0]))
print('footer_index[0] in raw data: ' + str(footer_index[0]))
print('mrsync_index[0] in data subtracted header&footer: ' +
      str(mrsync_index[0]))

file.close()
