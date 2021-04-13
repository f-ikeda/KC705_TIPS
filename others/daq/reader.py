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

    data_flagment = file.read(DATA_UNIT).hex()
    # str.hex(), only in python3
    print('No.' + str(int(file.tell() / 13) - 1) + ': ' + data_flagment)

file.close()
