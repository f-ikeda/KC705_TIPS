#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 01:12:35 2020

@author: f-ikeda
"""

import os

"path to binary file"
path = "test_data.log"

"open binary file as read only mode"
file = open(path,"rb")

"get file size which should be multiples of data_unit"
file_size = os.path.getsize(path)

"size of file which already read"
readed_size = 0

"data unit of KC705 in terms of byte"
data_unit = 13

"footer as str"
footer = "fee00aaaaaaaaaaaaaaaa"

"number of spills in file"
spill_count = 0

while not readed_size == file_size:
    "whether footer is read or not"
    footer_flag = 0
    "how many times data_flagment is update"
    read_count = 0

    "initialization of a spill data"
    spill_data = ""
    "initialization of event data"
    event_data = ""

    "read data of data_unit byte until footer comes"
    while not footer_flag:
        data_flagment = file.read(data_unit).hex()
        spill_data += data_flagment
        read_count += 1
    
        if data_flagment[5:] == footer:
            footer_flag = 1

        "strとして扱っている以上、ここはwhileの後にread_countだけで書き換え可能"
        "read_countと文字数で見てスライスでいける"
        if read_count > 1 and footer_flag == 0:
            event_data += data_flagment
    readed_size += read_count * data_unit
    spill_count += 1

    print("No."+str(spill_count)+" spill_data: "+spill_data)
    print("event_data: "+event_data)
    
    "data format"
    #each of data is 13 byte(104 bit)
    
    #about header
    #20 bit: A_BB_00
    #32 bit:HEADER 
    #32 bit: HEADER
    #4 bit: BOARD_ID
    #16 bit: SPLCOUNT
    
    #in terms of byte
    #01[AB]23[B0]45[00]67[12]89[34]1011[56]1213[70]1415[12]1617[34]1819[56]2021[7BOARD_ID]2223[SPLCOUNT]2425[SPLCOUNT]
    
    #about footer
    #16 bit: EMCOUNT
    #4 bit: 0
    #20 bit: F_EE_00
    #32 bit: FOOTER
    #32 bit: FOOTER
    
    #in terms of byte
    #01[EMCOUNT]23[EMCOUNT]45[0F]67[EE]89[00]1011[AA]1213[AA]1415[AA]1617[AA]1819[AA]2021[AA]2223[AA]2425[AA]
    
    "extract information from spill data"
    print("BOARD_ID: "+spill_data[21])
    print("SPILCOUNT: "+spill_data[22:26])
    
    print("EMCOUNT: "+spill_data[len(spill_data)-1-25:len(spill_data)-1-21])

"you can use hex binary as str"
#data_str = file.read(data_unit).hex()
#print(type(data_str))
#print(data_str)

#data_bin = bytes.fromhex(data_str)
#print(type(data_bin))
#print(data_bin)

"close binary file"
file.close()