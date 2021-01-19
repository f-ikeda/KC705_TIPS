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
・ファイルを自動で切り替えるさいに、書き込まれるファイルのサイズが13bytesの倍数になるよう、byte_flagmentのサイズを、self.save()に投げる前にifで判断するようにした
・生成されるファイルサイズがself._file_single_sizeで指定した値と食い違っていたのは、ファイルサイズを確認する時点で実はまだ真に書き込みは行われていなかったためであり、write()のあとすぐさまflush()するようにした
todo
・レート耐性(nc+リダイレクトとほぼ同程度まで耐えらえるか)
・flush()の是非について、というよりも、「データの受信」と「ディスクへの書き込み」を別にすべきなのか？？？
・flush()を、recv()とは別のthreadで定期的に呼び出せば、recv()に戻るまでの時間が短くなるのか？？？
others
・KC705に接続できない場合、あるいは接続が途中で切れた場合の処理について
・ctrl+cが押されたときにも、DATA SAVED: 保存先を適切に表示する
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
        if (os.path.getsize(self._file_path) >= self._file_single_size):
            self._file.close()
            print('DATA SAVED: ' + self._file_path)
            self._file_path = 'daqqed_data/' + \
                datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f') + '.dat'
            # year, month, day, _, hour, minute, second, millisecond

            self._file = open(client._file_path, mode='wb')
            # re-open with new name

        self._file.write(data)
        self._file.flush()
        # os.path.getsize()が評価される時点で、現に受け取った全てを書き込んでいるとは限らない
        # 従って、ここで確実にファイルバッファをフラッシュすることで、確実に書き込み済にする
        # このせいでrecv()に戻るまでが遅くなっては本末転倒なので、要検討

    def recieve(self):
        print('DAQ START: ' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f'))

        try:
            while True:
                # i.e. always..
                bytes_flagment = bytes()
                # recieved data will be stored in this bytes
                bytes_flagment = bytes(self._sock.recv(self._buffer_size))
                if (len(bytes_flagment) % DATA_UNIT == 0):
                    # recv()の挙動をよくわかっていないのだが、self._buffer_sizeまで溜めていっきに読み込むのではなく、実は逐次byte_flagmentに格納しているらしい(appendみたい？)
                    # 必ずしもself._buffer_sizeに達してからrecv()を抜けるわけではないよう
                    # 従って、ここで保存は13(bytes)の倍数ごとと指定しておく
                    self.save(bytes_flagment)

        except KeyboardInterrupt:
            # when ctrl+c comes
            if (len(bytes_flagment) != 0):
                self.save(bytes_flagment)
                # because there may be data in bytes_flagment that has not yet been saved
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
