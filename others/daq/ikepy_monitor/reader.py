import sys

DATA_UNIT = 13

argument = sys.argv
if(len(argument) != 4):
    print('USEAGE: $ python3 monitor.py start(13 bytes) end(13 bytes) path_to_datafile')
    sys.exit()
file_path = argument[3]
file = open(file_path, "rb")

file.seek(DATA_UNIT * int(argument[1]))

while file.tell() <= DATA_UNIT * int(argument[2]):
    # while not file.tell() == file_size:

    data_flagment = file.read(DATA_UNIT)
    # str.hex(), only in python3
    print(#'No.' + str(int(file.tell() / 13) - 1) + ': ' +
          'MPPC: '     + format(((int.from_bytes(data_flagment, 'big') & 0xFFFFFFFFFFFFFFFF0000000000) >> 12+1+27), '016x') +
          ' PMT: '     + format(((int.from_bytes(data_flagment, 'big') & 0x0000000000000000FFF0000000) >>    1+27),  '03x') +
          ' MR SYNC: ' + format(((int.from_bytes(data_flagment, 'big') & 0x00000000000000000008000000) >>      27),  '01x') +
          ' TDC: '     + format(((int.from_bytes(data_flagment, 'big') & 0x00000000000000000007FFFFFF)           ),  '04x'))

file.close()
