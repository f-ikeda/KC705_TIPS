#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 01:12:35 2020

@author: f-ikeda
"""
"""
#TODO
データの型は問題ないか、特にsig_dataに関して
CHを見やすくする！
高速化
"""
"""
#データ
spillcount_data: ndarray, スピルカウントの値(int)のつまった配列
eventmatching_data: ndarray, イベントマッチングの値(int)のつまった配列
tdc_data: ndarray, TDCの値(int)のつまった配列
sig_data: ndarray, SIGの値(int)のつまった配列
eventnum_data: ndarray, スピルごとのTDCした回数(int)のつまった配列

i,nともに0始まりであることに注意！
"""


import os
import numpy as np

#for debug
import matplotlib.pyplot as plt


def data_in_spill_i(data,eventnum_data,i):
    #input: data = tdc_data or sig_data, i
    #output: tdc_data or sig_dataのうち、i番目のスピルに属するものだけを取り出した部分列ndarray
    #i番目のスピルに属するtdc_data or sig_dataの要素数: eventnum_data[i]
    #i番目まで(i番目含む)のスピルに属するtdc_data or sig_dataの要素数: np.sum(eventnum_data[:i+1])
    
    #tdc_data = [1,2,|4,3,5,|1,4,3,3|2,1]
    #eventnum_data = [2,3,4,2]
    #このとき、2番目のスピルに属するtdc_dataの値は4個(1,4,3,3)
    #初めの値1と終わりの値3を指定するインデックスは、
    #初めの値1について、eventnum_data[0]+eventnum_data[1]=2+3=5
    #終わりの値3について、eventnum_data[0]+eventnum_data[1]+eventnum_data[2]-1=2+3+4-1=9-1(=8)
    
    return data[np.sum(eventnum_data[:i]):np.sum(eventnum_data[:i+1])]

def hitlist_data_in_spill_i_ch_n(sig_data,eventnum_data,i,n):
    #input: sig_data, i, n
    #output: i番目のスピルに属するsig_dataのうち、n番目のCHにヒットのある要素のインデックス一覧
    #i番目のスピルに属するsig_dataの部分列を、ビットシフトを行うために十分な型に変更
    sig_data_in_spill_i_uint64 = data_in_spill_i(sig_data,eventnum_data,i).astype(np.uint64)
    #i番目のスピルに属するsig_dataの部分列のうち、n桁目のフラグが立っている要素のインデックスを取得
    #n桁目のフラグだけ立っているビット: 1<<(n-1)
    #n桁目のフラグが立っているか: if bit & (1<<(n-1)):
    #0番目のchについては1桁目を、n番目のchについてはn+1桁目を見る必要がある
    output = np.where(((sig_data_in_spill_i_uint64)&(1<<(n+1-1)))==(1<<(n+1-1)))
    
    return output[0]

def hittimes_data_in_spill_i(sig_data,eventnum_data,i):
    #input: sig_data, eventnum_data, i
    #output: i番目のスピルに関して、インデックスnにn番目のCHへのヒット数が格納されたndarray
    #outputとなるndarray
    output = np.empty(0,dtype=np.int8)
    for n in range(0,all_ch):
        #i番目のスピルに属するデータのうち、n番目のCHにはhitlist_data_in_spill_i_ch_nの全要素数個のヒットがあるためappend
        #(インデックス0番目の要素は、0 CHにヒットした個数となる)
        output = np.append(output,hitlist_data_in_spill_i_ch_n(sig_data,eventnum_data,i,n).size)
        
    return output

def tdcdata_in_spill_i_ch_n(tdc_data,sig_data,eventnum_data,i,n):
    #input: tdc_data, sig_data, eventnum_data, i, n
    #output: i番目のスピルに属するtdc_dataの部分列のうち、n番目のCHへのヒットがあったものだけを取り出した部分列ndarray
    output = np.empty(0,dtype=np.int8)
        #i番目のスピルに属するsig_dataのうち、n番目のCHにヒットのある要素のインデックス一覧
        #i番目のスピルに属するtdc_dataのうち、このインデックスで指定される要素が、output
    for k in hitlist_data_in_spill_i_ch_n(sig_data,eventnum_data,i,n):
        output = np.append(output,data_in_spill_i(tdc_data,eventnum_data,i)[k])
    
    return output


#path to binary file
path = "/Users/f-ikeda/EXdata/KC705/chain_gray_cable.data"
#open binary file as read only mode
file = open(path,"rb")
#get file size which should be multiples of data_unit
file_size = os.path.getsize(path)
print("file_size: "+str(file_size))
#size of file which already read
readed_size = 0
#data unit of KC705 in terms of byte
data_unit = 13

#1 clock needs 5 ns
clock_time = 5e-9
#all 77 channels
all_ch = 77

#header as str (10 byte)
header = "abb00012345670123456"
header = bytes.fromhex(header)
#footer as str (11 byte)
footer = "0fee00aaaaaaaaaaaaaaaa"
footer = bytes.fromhex(footer)

#initialization of a header data
spillcount_data = np.empty(0,dtype=np.int8)
#initialization of a footer data
eventmatching_data = np.empty(0,dtype=np.int8)
#initialization of sig data
sig_data = np.empty(0,dtype=np.int64)
#initialization of tdc data
tdc_data = np.empty(0,dtype=np.int32)
#initialization of event number data
eventnum_data = np.empty(0,dtype=np.int64)

#最初の13 byteを読み飛ばす
#a = file.read(data_unit)

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
        
        #上位10 byte分で比較
        if data_flagment[:len(header)] == header:
            #下位2 byte分が欲しい
            spillcount_data = np.append(spillcount_data,int.from_bytes(data_flagment[11:],"big"))
        
        #下位11 byte分で比較
        elif data_flagment[-len(footer):] == footer:
            #上位2 byte分が欲しい
            eventmatching_data = np.append(eventmatching_data,int.from_bytes(data_flagment[:2],"big"))
            footer_flag = 1
            
            #フッターの抜けや、ファイルサイズが13バイトの倍数じゃないとき
            #break
         
        else:
            #上位77 bit分が欲しいから、上位80 bit(10 byte)分をとり、その中の下位3 bit分を捨てるため右シフト
            sig_data = np.append(sig_data,int.from_bytes(data_flagment[:10],"big")>>3)
            #下位27 bit分が欲しいから、下位32 bit(4 byte)分をとり、その中の上位5 bit分を捨てるため、上位5 bitが0で下位27 bitが1の2^27-1と&
            tdc_data = np.append(tdc_data,int.from_bytes(data_flagment[9:],"big")&(pow(2,27)-1))

            eventnum += 1
        
    eventnum_data = np.append(eventnum_data,eventnum)
    
    #フッターの抜けや、ファイルサイズが13バイトの倍数じゃないとき
    #break
    
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
        #64 bit: MainHodo
        #12 bit: PMT
        #1 bit: MR_Sync
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
"""

"""
#スピルごとのイベント数
plt.title("Events Number")
plt.xlabel("Spill Number")
plt.ylabel("Entry")
plt.scatter(spillcount_data,eventnum_data)
#"""

"""
#i=2番目のスピルについて、TDCのクロックの値
i=2
plt.title("TDC Values")
plt.xlabel("Clock Counts")
plt.ylabel("Entry")
plt.hist(data_in_spill_i(tdc_data,eventnum_data,i))
#"""

"""
#i=2番目のスピルについて、TDCのクロックの値の間隔
i=2
plt.title("TDC diff")
plt.xlabel("Clock Counts")
plt.ylabel("Entry")
edges = np.arange(np.diff(data_in_spill_i(tdc_data,eventnum_data,i),n=1).min(),np.diff(data_in_spill_i(tdc_data,eventnum_data,i),n=1).max())
plt.hist(np.diff(data_in_spill_i(tdc_data,eventnum_data,i),n=1),bins=edges,histtype="step",log=True)
#"""

"""
#全てのスピルについて、TDCのクロックの値の間隔
plt.title("TDC diff")
plt.xlabel("Clock Counts")
plt.ylabel("Entry")
y = np.empty(0,dtype=np.int32)
#スピルの数は全部でspillcount_dataの要素数個
for i in range(spillcount_data.size):
    y = np.append(y,np.diff(data_in_spill_i(tdc_data,eventnum_data,i),n=1))
edges = np.arange(y.min(),y.max())
plt.hist(y,bins=edges,histtype="step",log=True)
#"""

"""
#i=2番目のスピルについて、チャンネルごとのヒット数
i=2
plt.title("Channel Hits")
plt.xlabel("Channel Number")
plt.ylabel("Entry")
#全部でall_ch
plt.scatter(np.arange(all_ch),hittimes_data_in_spill_i(sig_data,eventnum_data,i))
#ヒットのあったCH一覧
print(np.nonzero(hittimes_data_in_spill_i(sig_data,eventnum_data,i)))
#"""

"""
#全てのスピルについて、チャンネルごとのヒット数の合計
plt.title("Total Channel Hits")
plt.xlabel("Channel Number")
plt.ylabel("Entry")
y = np.zeros(all_ch)
#スピルの数は全部でspillcount_dataの要素数個
for i in range(spillcount_data.size):
    y += hittimes_data_in_spill_i(sig_data,eventnum_data,i)
    
#全部でall_ch
plt.scatter(np.arange(all_ch),y)
#ヒットのあったCH一覧
print(np.nonzero(y))
#"""

"""
#ヒットのあったCH一覧(FMA_HPC, J1用に直したもの)
print((np.where(hitch_data-32>0)[0]-45))
#plt.scatter(np.arange(18), np.roll(hitch_data,32)[:18])
#logで見て出ない、他はないということ
plt.scatter(np.arange(18), np.log(np.roll(hitch_data,32)[:18]))
plt.xticks(np.arange(18))
#"""

"""
#i=2番目のスピル、n=45 CHについて、TDCのクロックの値の間隔
i=2
n=45
plt.title("No.n Channel Hits")
plt.xlabel("Clock Count")
plt.ylabel("Entry")
#plt.hist(tdcdata_in_spill_i_ch_n(tdc_data,sig_data,eventnum_data,i,n))
#間隔なら、
plt.hist(np.diff(tdcdata_in_spill_i_ch_n(tdc_data,sig_data,eventnum_data,i,n),n=1))
#"""

"""
#全てのスピル、n=45 CHについて、TDCのクロックの値の間隔
n=45
plt.title("No.n Channel Hits")
plt.xlabel("Clock Count")
plt.ylabel("Entry")
y = np.empty(0,dtype=np.int32)
#スピルの数は全部でspillcount_dataの要素数個
for i in range(spillcount_data.size):
    y = np.append(y,np.diff(tdcdata_in_spill_i_ch_n(tdc_data,sig_data,eventnum_data,i,n),n=1))
edges = np.arange(y.min(),y.max())
plt.hist(y,bins=edges,histtype="step",log=True)
#"""

"""
#you can use hex binary as str
data_str = file.read(data_unit).hex()
print(type(data_str))
print(data_str)

data_bin = bytes.fromhex(data_str)
print(type(data_bin))
print(data_bin)
"""
