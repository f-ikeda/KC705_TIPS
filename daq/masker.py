# This file should be placed in rbcp/script directory

import os

IP = ' 192.168.10.16'
PORT = ' 4660'
SETPARM_COMMAND = '../Setparm_command'
WRB = ' wrb'

REGISTER = {0x13: '0<---->7 HPC(New Hod.)', 0x12: '8<---->15 HPC(New Hod.)', 0x11: '16<---->23 HPC(New Hod.)', 0x10: '24<---->31 HPC(New Hod.)',
            0x17: '0<---->7 LPC(New Hod.)', 0x16: '8<---->15 LPC(New Hod.)', 0x15: '16<---->23 LPC(New Hod.)', 0x14: '24<---->31 LPC(New Hod.)',
            0x19: 'P3 MrSync Ev.M. BH1 BH2 HOD_ALL_OR  HOD11 HOD10', 0x18: '9<---->2 (HOD)'}

for key in REGISTER:
    # os.system('ls')
    print('In reg: ' + hex(key))
    print(REGISTER[key] + '    (2: all 0: none)')

    binary_str_reversed = input()
    if (binary_str_reversed == str(2)):
        mask_value = ' 0x' + str('FF')
    else:
        binary_str = binary_str_reversed[::-1]
        mask_value = ' 0x' + format(int(binary_str, 2), '02X')

    os.system(SETPARM_COMMAND + IP + PORT + WRB +
              ' 0x' + format(key, '02X') + mask_value)
    # print(SETPARM_COMMAND + IP + PORT + WRB + ' 0x' + format(key, '02X') + mask_value)
