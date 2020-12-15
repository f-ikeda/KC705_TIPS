#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 00:23:41 2020

@author: f-ikeda
"""

"""
未使用チャンネルのマスクとDAQスタートのレジスタ書き換えは、別にスクリプトでやれ
#"""

"""
TODO
Ctrl+C押されたときの、データの保存の処理を要チェック
#"""

import time
import datetime

import socket

class Client(object):
    
    def __init__(self, ip_address="192.168.10.16", tcp_port=24, data_unit=13):
        self._ip_address = ip_address
        self._tcp_port = tcp_port
        #unit in bytes
        self._data_unit = data_unit
        #path to save data
        self._savedata_path = None
        #for saving data
        self._fileout = None
        #for saving data
        self._spillchunk = 1000
        #for saving data especially when ctrl+c comes
        self._saved_flag = 0
        
        #for Rbcp
        self._rbcp = None
        #for socket
        self._sock = None
        
        #header (10 byte)
        self._header = bytes.fromhex("AB B0 00 12 34 56 70 12 34 56")
        #footer (11 byte)
        self._footer = bytes.fromhex("0F EE 00 AA AA AA AA AA AA AA AA")
        
        #in order to begin to save the data with header
        self._firstheader_flag = 0
        
        #self._header_flag = 0
        self._footer_flag = 0
        
        self._eventnum = 0
        
    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(3.0)
        self._sock.connect((self._ip_address, self._tcp_port))
        
    def disconnect(self):
        self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()
    
    def save(self, data):
        #print("Starting save()..")
        #保存するファイルの名前の変更はここで行う
        #self._savedata_path += str(time.time())などとして
        
        if self._firstheader_flag:
            self._fileout.write(data)
            self._saved_flag = 1
        
        if self._eventnum % self._spillchunk == 0:
            self._savedata_path = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("No."+str(self._eventnum)+" spill was, saved!")
        
    def teller(self, bytes_flagment):
        
        #上位10 byte分で比較
        if bytes_flagment[:len(self._header)] == self._header:
            self._firstheader_flag = 1
            #self._header_flag = 1
            self._footer_flag = 0
        
        #下位11 byte分で比較
        elif bytes_flagment[-len(self._footer):] == self._footer:
            #self._header_flag = 0
            self._footer_flag = 1
            self._eventnum += 1
        
    def recieve(self):
        
        #recieved data will be stored in this bytes
        bytes_flagment = bytes()
        
        try:
            while not True:              
                #recieve and save data in the unit of self._data_unit
                bytes_flagment = bytes(self._sock.recv(self._data_unit))
                self._saved_flag = 0
                self.teller(bytes_flagment)
                self.save(bytes_flagment)
        
        #recieve data until ctrl+c comes
        except KeyboardInterrupt:
            print("Stopped daq, wow!")
            #not yet saved lateset bytes_flagment
            #つまり、recvしてsaveまだの状態でctrl+cが押されたときのことを懸念している
            if not self._saved_flag:
                self.teller(bytes_flagment)
                self.save(bytes_flagment)
            while not self._footer_flag:             
                bytes_flagment = bytes(self._sock.recv(self._data_unit))
                self._saved_flag = 0
                self.teller(bytes_flagment)
                self.save(bytes_flagment)

if __name__ == '__main__':
    
    client = Client()
    
    client._savedata_path = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")   
    client._fileout = open(client._savedata_path,mode="wb")
    
    client.connect()
    client.recieve()
    client.disconnect()
    
    client._fout.close()
    
    