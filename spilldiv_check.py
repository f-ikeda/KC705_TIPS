# purpose: to check the behaviour of spill division function in MCS-mode

for mrsync in range(500000):
    if ((mrsync & 0b11111111111) == 0b00000000000) \
            & (((mrsync >> 11) & 0b1) == 0b1):
        print('mrsync:', mrsync)
