import sys
import struct

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

BITS_MASK_SIG = (2 ** BITS_SIZE_SIG - 1) << BITS_SIZE_TDC
# 104 bits, only the upper BITS_SIZE_SIG bit is filled with 1
BITS_MASK_TDC = 2 ** BITS_SIZE_TDC - 1
# 104 bits, only the lower BITS_SIZE_TDC bit is filled with 1

BITS_MASK_SIG_NEWHOD_ALLOR = (2 ** BITS_SIZE_SIG_NEWHOD - 1) - \
    (2 ** (BITS_SIZE_SIG_MRSYNC + BITS_SIZE_SIG_PMT) - 1)
# only the upper BIT_SIZE_SIG_NEWHOD bit is filled with 1
BITS_MASK_SIG_MRSYNC = 2 ** BITS_SIZE_SIG_MRSYNC - 1
# only the lower BIT_SIZE_SIG_MRSYNC bit is filled with 1


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

# --------READING DATA--------
data_num = int(argument[1])
# number of tdc data to read
TIME_READ_S = time.time()
data_str = file.read(DATA_UNIT * data_num).hex()
# str.hex(), only in python3
TIME_READ_F = time.time()
print("READ TIME [s]: " + str(TIME_READ_F - TIME_READ_S))

# --------FORMATTING DATA--------
TIME_FORMAT_S = time.time()
data = np.array([int('0x' + data_str[i:i+26], 16)
                 for i in range(0, len(data_str), 26)])
# あるいは、dtypeを自作してnp.frombuffer()を使う
TIME_FORMAT_F = time.time()
print("FORMAT TIME [s]: " + str(TIME_FORMAT_F - TIME_FORMAT_S))

TIME_PROCESS_S = time.time()
data_sig = (data & BITS_MASK_SIG) >> BITS_SIZE_TDC
data_tdc = data & BITS_MASK_TDC
TIME_PROCESS_F = time.time()
print("PROCESS TIME [s]: " + str(TIME_PROCESS_F - TIME_PROCESS_S))

'''
TIME_DRAW_S = time.time()
plt.hist(data, histtype='step', log=True)
TIME_DRAW_F = time.time()
print("DRAW TIME [s]: " + str(TIME_DRAW_F - TIME_DRAW_S))
plt.show()
'''

file.close()
