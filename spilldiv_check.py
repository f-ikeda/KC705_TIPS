# purpose: to check the behaviour of spill division function in MCS-mode

count_old = 0
for i in range(500000):
    mrsync = i
    if ((mrsync & 0b11111111111) == 0b00000000000) \
            & (((mrsync >> 11) & 0b1) == 0b1):
        print('mrsync:', mrsync,
              'diff:', (mrsync - count_old))
        count_old = mrsync
