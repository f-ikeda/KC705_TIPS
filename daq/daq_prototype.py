#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 00:23:41 2020

@author: f-ikeda
"""

"""
TODO
Ctrl+C押されたときの、データの保存の処理を要チェック
"""

import time

from sitcpy.rbcp import Rbcp
import socket

import numpy as np

class Client(object):
    
    def __init__(self, ip_address="192.168.10.16", tcp_port=24, data_unit=13, savedata_path=""):
        self._ip_address = ip_address
        self._tcp_port = tcp_port
        #unit in bytes
        self._data_unit = data_unit
        #path to save data
        self._savedata_path = savedata_path
        
        #for Rbcp
        self._rbcp = None
        #for socket
        self._sock = None
        
        #header (10 byte)
        self._header = bytes.fromhex("AB B0 00 12 34 56 70 12 34 56")
        #footer (11 byte)
        self._footer = bytes.fromhex("0F EE 00 AA AA AA AA AA AA AA AA")
        
        #initialization of a header data
        self._spillcount_data = np.empty(0,dtype=np.int8)
        #initialization of a footer data
        self._eventmatching_data = np.empty(0,dtype=np.int8)
        #initialization of sig data
        self._sig_data = np.empty(0,dtype=np.int64)
        #initialization of tdc data
        self._tdc_data = np.empty(0,dtype=np.int32)
        #initialization of event number data
        self._eventnum_data = np.empty(0,dtype=np.int64)
        self._eventnum = 0
        
        #initialization of spill number
        self._spillnum = 0
        #how many spills in a file
        self._spillmax = 300
        
    def connect(self):
        print("Starting connect()..")
        #connect to KC705 via Rbcp 
        self._rbcp = Rbcp(self._ip_address)
        
        #connect to KC705 via socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(2.0)
        self._sock.connect((self._ip_address, self._tcp_port))
        
        print("Finiched connect()!")
        
    def disconnect(self):
        print("Starting disconnect()..")
        #disconnect from KC705
        self._sock.close()
        
        print("Finished disconnect()!")
        
    def pack(self, bytes_flagment):
        print("Starting shaper()..")
        #pack the data into apropriate numpyarray
        
        #上位10 byte分で比較
        if bytes_flagment[:len(self._header)] == self._header:
            #下位2 byte分が欲しい
            self._spillcount_data = np.append(self._spillcount_data,int.from_bytes(bytes_flagment[11:],"big"))
            
            #if header comes, then initialize eventnum
            self._eventnum = 0
            #cout how many spill comes
            self._spillnum += 1
        
        #下位11 byte分で比較
        elif bytes_flagment[-len(self._footer):] == self._footer:
            #上位2 byte分が欲しい
            self._eventmatching_data = np.append(self._eventmatching_data,int.from_bytes(bytes_flagment[:2],"big"))
            
            #if footer comes, then append eventnum into eventnum_data
            self._eventnum_data = np.append(self._eventnum_data,self._eventnum)
            
            if self._spillnum == self._spillmax:
                self.save()
         
        else:
            #上位77 bit分が欲しいから、上位80 bit(10 byte)分をとり、その中の下位3 bit分を捨てるため右シフト
            self._sig_data = np.append(self._sig_data,int.from_bytes(bytes_flagment[:10],"big")>>3)
            #下位27 bit分が欲しいから、下位32 bit(4 byte)分をとり、その中の上位5 bit分を捨てるため、上位5 bitが0で下位27 bitが1の2^27-1と&
            self._tdc_data = np.append(self._tdc_data,int.from_bytes(bytes_flagment[9:],"big")&(pow(2,27)-1))
            self._eventnum += 1
        
        print("Finished save()!")
    
    def save(self):
        print("Starting save()..")
        #保存するファイルの名前の変更はここで行う
        #self._savedata_path += str(hoge)などとして
        #save the all numpy arrays as a npz
        np.savez(self._savedata_path,self._spillcount_data,self._eventmatching_data,self._sig_data,self._tdc_data,self._eventnum_data)
        
        #initialize the all numpy arrays as empty
        self._spillcount_data = np.empty(0,dtype=np.int8)
        self._eventmatching_data = np.empty(0,dtype=np.int8)
        self._sig_data = np.empty(0,dtype=np.int64)
        self._tdc_data = np.empty(0,dtype=np.int32)
        self._eventnum_data = np.empty(0,dtype=np.int64)
        print("Finished save()!")
    
    def reset(self):
        print("Starting reset()..")
        #reset KC705
        self._rbcp.write(0x02,"1")
        time.sleep(0.01)
        self._rbcp.write(0x02,"0")
        time.sleep(0.01)
    
        print("Finished reset()!")
    
    def start(self):
        print("Starting start()..")
        #make KC705 to start transfering data
        self._rbcp.write(0x01,"1")
        
        print("Finished start()!")
        
    def stop(self):
        print("Starting stop()..")
        #make KC705 to stop transfering data
        time.sleep(3)
        self._rbcp.write(0x01,"0")
        
        print("Finished stop()!")
        
    def recieve(self):
        print("Starting recieve()..")
        #start daq & recieve data from SiTCP device
        #基本的に、常にデータを受信すること。ソケットは閉じない
        
        self.start()
        
        #recieved data will be stored in this bytes
        bytes_flagment = bytes()
        
        try:
            while True:
                #recieve and pack&save data in the unit of self._data_unit
                bytes_flagment += bytes(self.sock.recv(self._data_unit))
                self.pack(bytes_flagment)
        
        #recieve data until Ctrl-C comes
        except KeyboardInterrupt:
            self.stop()
            self.save()
            
        print("Finished recieve()!")

if __name__ == '__main__':
    client = Client()
    
    client.connect()
    
    client.recieve()
    
    