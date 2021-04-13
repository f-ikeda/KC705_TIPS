'''
USEAGE: $ python3 daqer.py file_single_size(kB, option)
ABOUT: using socket with blocking-mode, just repeat recv and saving data.
       the file name will be YearMonthDay_HourMonthSecondMillisecond.dat at the first time when data saved.
       when the size of one file reaches file_single_size(kB) or 500(MB, default), the next new file is created at daqqed_data/.
       this program does NOT make any decisions about Headers or Footers. it just receives and saves the data as it comes in.
       the program will continue to run until ctrl+c is pressed.


done
・受信した中身が正しい内容であることは、P3(100(Hz)~2.6(kbps))だけを入力して、スピルカウントおよびnc+リダイレクトの結果と照らして確認した
・ファイルを自動で切り替えたとき、ファイル間の区切りでデータの欠けがないことは、P3(100(Hz)~2.6(kbps))だけを入力して、スピルカウントを見て確認した
・self._buffer_sizeに達しない中途半端なサイズのデータも、きちんと受け取れていることを確認した
・ファイルを自動で切り替えるさいに、書き込まれるファイルのサイズが13bytesの倍数になるよう、受信したデータのうち13bytesの定数倍だけsave()して、残りを次回に回すようにした
・現在のファイルに書き込んだデータサイズをカウントするようにした
todo
・P3に100(kHz)(~2.6(Mbps))の信号を入れてヘッダーとフッターだけ見ても、nc+リダイレクトではまともにデータ取得できるのに、daqer.pyではできない！！(4bytesの欠けや、ファイルサイズが指定通りを越してしまう！)
ただし、macbookとLANアダプタ使用
others
・KC705に接続できない場合、あるいは接続が途中で切れた場合の処理について
・ctrl+cが押されたときにも、DATA SAVED: 保存先を適切に表示する
・print()での表示を気の利いたものにする
'''

import sys
import os
import socket
import datetime

DATA_UNIT = 13
# bytes
SAFETY_FACTOR = 400
# the ratio of the 'real' buffer size to the amount of data received at any one time


class Client(object):

    def __init__(self, ip_address='192.168.10.16', tcp_port=24, buffer_data_num=100, file_single_size=500*1000*1000):
        self._ip_address = ip_address
        self._tcp_port = tcp_port

        self._buffer_size = DATA_UNIT * buffer_data_num
        # number of data received at one time, in (bytes)

        self._file_path = None
        self._file = None
        self._file_single_size = file_single_size
        # maximum size of a single file, in (bytes)
        self._file_written_size = 0
        # how many bytes have already been written to current file, in (bytes)

        self._sock = None
        # socket
        self._sock_timeout = None
        # the timeout is deactivated, i.e. completely blocking mode

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self._sock_timeout)
        self._sock.connect((self._ip_address, self._tcp_port))

    def disconnect(self):
        self._sock.close()

    def save(self, data):
        if (self._file_written_size >= self._file_single_size):
            self._file.close()
            print('DATA SAVED: ' + self._file_path)
            self._file_path = 'daqqed_data/' + \
                datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f') + '.dat'
            # year, month, day, _, hour, minute, second, millisecond

            self._file = open(client._file_path, mode='wb')
            # re-open with new name
            self._file_written_size = 0
            # count-clear

        self._file.write(data)
        self._file_written_size += len(data)
        # count-up

    def recieve(self):
        print('DAQ START: ' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f'))

        bytes_locker = bytes()
        # in case ctrl+c is pressed here, not to be undefined
        bytes_handbag = bytes()
        try:
            while True:
                # i.e. always..
                bytes_locker = bytes_handbag
                # recieved data will be stored in this bytes
                bytes_locker = bytes(self._sock.recv(self._buffer_size))
                bytes_suitcase = bytes_locker[:len(
                    bytes_locker)-len(bytes_locker) % DATA_UNIT]
                bytes_handbag = bytes_locker[len(bytes_suitcase):]

                # len(bytes_locker) = len(bytes_suitcase) + len(bytes_handbag)
                #                           13*X(bytes)           <13(bytes)

                self.save(bytes_suitcase)

        except KeyboardInterrupt:
            # when ctrl+c comes
            if (len(bytes_locker) != 0):
                self.save(bytes_locker)
                # because there may be data in bytes_locker that has not yet been saved
            print('\nDAQ STOP: ' +
                  datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f'))


if __name__ == '__main__':

    argument = sys.argv
    if(len(argument) <= 1):
        file_single_size = 500*1000*1000
        # 500(MB)
    else:
        file_single_size = int(argument[1]) * 1000
        # argument[1](kB), note that argument[any] is just str
        print('FILE SINGLE SIZE: ' + str(file_single_size))

    client = Client(file_single_size=file_single_size)

    client.connect()

    print('CURRENT BUFFER SIZE [bytes]: ' + str(
        int(client._sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))))
    print('BUFFER SIZE SET AS [bytes]: ' +
          str(int(client._buffer_size * SAFETY_FACTOR)))
    client._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(
        client._buffer_size * SAFETY_FACTOR))
    # this is the real buffer size (in kernel)...

    client._file_path = 'daqqed_data/' + \
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f') + '.dat'
    # year, month, day, _, hour, minute, second, millisecond
    client._file = open(client._file_path, mode='wb')

    client.recieve()
    client._file.close()

    client.disconnect()
