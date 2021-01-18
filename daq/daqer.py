'''
ABOUT: using socket with blocking-mode、just repeat recv and saving data.
       the file name will be YearMonthDay_HourMonthSecondMillisecond.dat at the first time when data saved.
       when the size of one file reaches 500 [MB], the next new file is created.
       this program does NOT make any decisions about Headers or Footers. it just receives and saves the data as it comes in.
       the program will continue to run until ctrl+c is pressed.

todo
・取りこぼしがないか
その他
・タイムアウトの挙動について(データなしで起動してみる)
'''
import os
import socket
import datetime

DATA_UNIT = 13
# bytes
SAFETY_FACTOR = 400
# the ratio of the 'real' buffer size to the amount of data received at any one time


class Client(object):

    def __init__(self, ip_address='192.168.10.16', tcp_port=24, buffer_data_num=100, file_single_size=500*1024*1024):
        self._ip_address = ip_address
        self._tcp_port = tcp_port

        self._buffer_size = DATA_UNIT * buffer_data_num
        # number of data received at one time

        self._file_path = None
        self._file = None
        self._file_single_size = file_single_size
        # maximum size of a single file, in [bytes]

        self._sock = None
        # socket
        self._sock_timeout = 3.0
        # [s]

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self._sock_timeout)
        self._sock.connect((self._ip_address, self._tcp_port))

    def disconnect(self):
        self._sock.close()

    def save(self, data):
        if (os.path.getsize(self._file_path) > self._file_single_size):
            self._file.close()
            print('DATA SAVED: ' + self._file_path)
            self._file_path = 'daqqed_data/' + \
                datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f') + '.dat'
            # year, month, day, _, hour, second, millisecond

            self._file = open(client._file_path, mode='wb')
            # re-open with new name

        self._file.write(data)

    def recieve(self):
        print('DAQ START: ' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f'))

        try:
            while True:
                # i.e. always..
                bytes_flagment = bytes()
                # recieved data will be stored in this bytes
                bytes_flagment = bytes(self._sock.recv(self._buffer_size))
                self.save(bytes_flagment)

        except KeyboardInterrupt:
            # when ctrl+c comes
            if (len(bytes_flagment) != 0):
                self.save(bytes_flagment)
                # because there may be data in bytes_flagment that has not yet been saved
            print('DAQ STOP: ' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f'))


if __name__ == '__main__':

    client = Client()

    print('CURRENT BUFFER SIZE [bytes]: ' + str(
        int(client._sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))))
    print('BUFFER SIZE SET AS [bytes]: ' +
          str(int(client._buffer_size * SAFETY_FACTOR)))
    client._sock.setsockopt(
        socket.SOL_SOCKET, socket.SO_RCVBUF, int(client._buffer_size * SAFETY_FACTOR))
    # this is the real buffer size (in kernel)...

    client._file_path = 'daqqed_data/' + \
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f') + '.dat'
    # year, month, day, _, hour, minute, second, millisecond
    client._file = open(client._file_path, mode='wb')

    client.connect()
    client.recieve()
    client.disconnect()

    client._file.close()
