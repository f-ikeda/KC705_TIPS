#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 01:12:35 2020

@author: f-ikeda
"""
"""
#TODO
高速化
"""
"""
#データ
spillcount_data: ndarray, スピルカウントの値(int)のつまった配列
eventmatching_data: ndarray, イベントマッチングの値(int)のつまった配列
tdc_data: ndarray, TDCの値(int)のつまった配列
sig_data: ndarray, SIGの値(int)のつまった配列
eventnum_data: ndarray, スピルごとのTDCした回数(int)のつまった配列
"""


import os
import numpy as np

#for debug
import matplotlib.pyplot as plt
import matplotlib as mpl


#path to binary file
path = "daq-test.dat"
#open binary file as read only mode
file = open(path,"rb")
#get file size which should be multiples of data_unit
file_size = os.path.getsize(path)
print("file_size: "+str(file_size))
print("file.tell(): "+str(file.tell()))
#size of file which already read
readed_size = 0
#data unit of KC705 in terms of byte
data_unit = 13

#header as str
#header = "abb000123456701234567"
#footer as str
#footer = "0fee00aaaaaaaaaaaaaaaa"

#header as str
header = "abb0"
header = bytes.fromhex(header)
#footer as str
footer = "aaaaaaaa"
footer = bytes.fromhex(footer)

#initialization of a header data
spillcount_data = np.empty(0,dtype=np.int8)
#initialization of a footer data
eventmatching_data = np.empty(0,dtype=np.int8)
#initialization of tdc data
tdc_data = np.empty(0,dtype=np.int8)
#initialization of sig data
sig_data = np.empty(0,dtype=np.int16)
#initialization of event number data
eventnum_data = np.empty(0,dtype=np.int8)

while not file.tell() == file_size:
    
    #whether footer is read or not
    footer_flag = 0
    #initialization of event number
    eventnum = 0
    
    #read data of data_unit byte until footer comes
    while not footer_flag:
        #要素数13で、1 byteごとに格納
        data_flagment = file.read(data_unit)
        readed_size += data_unit
        print("file.tell(): "+str(file.tell()))
        
        #上位2 byte分で比較
        if data_flagment[:2] == header:
            #下位4 byte分が欲しい
            spillcount_data = np.append(spillcount_data,int.from_bytes(data_flagment[9:],"big"))
        
        #下位4 byte分で比較
        elif data_flagment[9:] == footer:
            #上位2 byte分が欲しい
            eventmatching_data = np.append(eventmatching_data,int.from_bytes(data_flagment[:2],"big"))
            footer_flag = 1
         
        else:
            #下位27 bit分が欲しいから、下位32 bit(4 byte)分をとり、その中の上位5 bit分を捨てるため、上位5 bitが0で下位27 bitが1の2^27-1と&
            tdc_data = np.append(tdc_data,int.from_bytes(data_flagment[9:],"big")&(pow(2,27)-1))
            #上位77 bit分が欲しいから、上位80 bit(10 byte)分をとり、その中の下位3 bit分を捨てるため右シフト
            sig_data = np.append(sig_data,int.from_bytes(data_flagment[10:],"big")>>3)
            eventnum += 1
        
    eventnum_data = np.append(eventnum_data,eventnum)
         
    #print(str(spillcount_data.size)+":"+str(eventmatching_data.size)+":"+str(tdc_data.size))
    
    #data format
    #each of data is 13 byte(104 bit)
    
    #about header
    #20 bit: A_BB_00
    #32 bit:HEADER
    #32 bit: HEADER
    #4 bit: BOARD_ID
    #16 bit: SPLCOUNT
    
    #in terms of byte
    #01[AB]23[B0]45[00]67[12]89[34]1011[56]1213[70]1415[12]1617[34]1819[56]2021[7BOARD_ID]2223[SPLCOUNT]2425[SPLCOUNT]
    
    #about TDC data
    #77 bit: SIG
    #27 bit: COUNTER
    
    #about footer
    #16 bit: EMCOUNT
    #4 bit: 0
    #20 bit: F_EE_00
    #32 bit: FOOTER
    #32 bit: FOOTER
    
    #in terms of byte
    #01[EMCOUNT]23[EMCOUNT]45[0F]67[EE]89[00]1011[AA]1213[AA]1415[AA]1617[AA]1819[AA]2021[AA]2223[AA]2425[AA]

#close binary file
file.close()

"""
#解析コードは以下に書く
i番目のスピルに属するTDCのデータ数: eventnum_data[i]
i-1番目までのスピルに属するTDCのデータ数: np.sum(eventnum_data[:i])
i番目のスピルに属するTDCのデータ: tdc_data[np.sum(eventnum_data[:i])-1:np.sum(eventnum_data[:i])-1+eventnum_data[i]]
n番目のフラグだけ立っているビット: 1<<(n-1)
n番目のフラグが立っているか: if bit & (1<<(n-1)):
"""
#1 clock needs 5 ns
clock_time = 5e-9
#all 77 channels
all_ch = 77

"""
#スピルごとのイベント数
plt.title("Events Number")
plt.xlabel("Spill Number")
plt.ylabel("Entry")
plt.scatter(spillcount_data,eventnum_data)
"""

"""
#i=2番目のスピルについて、TDCのクロックの値
i=2
plt.title("TDC Values")
plt.xlabel("Clock Counts")
plt.ylabel("Entry")
plt.hist(tdc_data[np.sum(eventnum_data[:i])-1:np.sum(eventnum_data[:i])-1+eventnum_data[i]])
"""

"""
#i=2番目のスピルについて、TDCのクロックの値の間隔
i=2
plt.title("TDC diff")
plt.xlabel("Clock Counts")
plt.ylabel("Entry")
plt.hist(np.diff(tdc_data[np.sum(eventnum_data[:i])-1:np.sum(eventnum_data[:i])-1+eventnum_data[i]],n=1))
"""


#i=2番目のスピルについて、チャンネルごとのヒット数
i=2
plt.title("Channel Hits")
plt.xlabel("Channel Number")
plt.ylabel("Entry")
#ビットシフトを行うために十分な型に変更
sig_data_uint64 = sig_data[np.sum(eventnum_data[:i])-1:np.sum(eventnum_data[:i])-1+eventnum_data[i]].astype(np.uint64)
#all_chの、ヒット数のつまった配列、その初期化
hit_ch = np.empty(0,dtype=np.int8)
#all_ch桁目のフラグが立っていれば+1
for n in range(1,all_ch+1):
    #n番目のフラグの立っているchannelがあるイベントの、インデックスのリストを取得
    hit_list = np.where(((sig_data_uint64)&(1<<(n-1)))==(1<<(n-1)))
    #n番目のchannelにはhit_list個のヒットがあるため、hit_listの要素数を加算
    hit_ch = np.append(hit_ch,hit_ch.size)
#全部でall_ch
plt.scatter(np.arange(all_ch), hit_ch)

"""
#you can use hex binary as str
data_str = file.read(data_unit).hex()
print(type(data_str))
print(data_str)

data_bin = bytes.fromhex(data_str)
print(type(data_bin))
print(data_bin)
"""
