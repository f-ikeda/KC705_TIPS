import sys

import numpy as np

import matplotlib.pyplot as plt

import time
# for debug

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

BITS_MASK_SIG = 2 ** BITS_SIZE_SIG - 1
# a sequence of BITS_SIZE_SIG 1s
BITS_MASK_TDC = 2 ** BITS_SIZE_TDC - 1
# a sequence of BITS_SIZE_TDC 1s

BITS_MASK_SIG_NEWHOD_ALLOR = (2 ** BITS_SIZE_SIG_NEWHOD - 1) - \
    (2 ** (BITS_SIZE_SIG_MRSYNC + BITS_SIZE_SIG_PMT) - 1)
# only the upper BIT_SIZE_SIG_NEWHOD bit is filled with 1
BITS_MASK_SIG_MRSYNC = 2 ** BITS_SIZE_SIG_MRSYNC - 1
# only the lower BIT_SIZE_SIG_MRSYNC bit is filled with 1


def get_sig_tdc(data, i, n):
    # get the i-th (start with 0) sig and tdc from the beginning of the array with n tdc data
    sig = BITS_MASK_SIG & (int.from_bytes(data, 'big') >> (
        ((n - 1) - i) * DATA_UNIT * 8 + BITS_SIZE_TDC))
    tdc = BITS_MASK_TDC & (int.from_bytes(data, 'big') >>
                           (((n - 1) - i) * DATA_UNIT * 8))
    return sig, tdc


def hit_newhod_allor(sig):
    # return 1 if there is a hit on mppc's allor
    if (BITS_MASK_SIG_NEWHOD_ALLOR & sig != 0):
        return 1
    else:
        return 0


def hit_mrsync(sig):
    # return 1 if there is a hit on mrsync
    if (BITS_MASK_SIG_MRSYNC & sig != 0):
        return 1
    else:
        return 0


argument = sys.argv
if(len(argument) != 3):
    print('USEAGE: $ python3 monitor.py number_of_tdcdata_to_read path_to_datafile')
    sys.exit()

file_path = argument[2]
file = open(file_path, 'rb')

data_num = int(argument[1])
# number of tdc data to read
TIME_READ_S = time.time()
data_flagment = file.read(DATA_UNIT * data_num)
TIME_READ_F = time.time()
print("READ TIME [s]: " + str(TIME_READ_F - TIME_READ_S))

data = np.empty(0, dtype=np.int8)

TIME_PROCESS_S = time.time()
for i in range(data_num):
    sig, tdc = get_sig_tdc(data_flagment, i, data_num)
    if(hit_newhod_allor(sig)):
        data = np.append(data, tdc)
TIME_PROCESS_F = time.time()
print("PROCESS TIME [s]: " + str(TIME_PROCESS_F - TIME_PROCESS_S))

TIME_DRAW_S = time.time()
plt.hist(data, histtype='step', log=True)
TIME_DRAW_F = time.time()
print("DRAW TIME [s]: " + str(TIME_DRAW_F - TIME_DRAW_S))
plt.show()
