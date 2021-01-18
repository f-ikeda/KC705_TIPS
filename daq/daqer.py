'''
todo
・そもそも動作するのか
・動作するなら、取りこぼしがないか
・動作するなら、データの保存先を定期的に変える
'''

import socket

import datetime

DATA_UNIT = 13
# bytes


class Client(object):

    def __init__(self, ip_address="192.168.10.16", tcp_port=24, buffer_data_num=1000):
        self._ip_address = ip_address
        self._tcp_port = tcp_port

        self._buffer_size = DATA_UNIT * buffer_data_num

        self._file_path = None
        self._file = None

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
        '''
        if (self._firstheader_flag != 0) and (self._eventnum % self._spillchunk == 0):
            self._file_path = "test_data/" + \
                datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            client._file = open(client._file_path, mode="wb")
            client._file.close()
        '''

        self._file.write(data)

    def recieve(self):

        bytes_flagment = bytes()
        # recieved data will be stored in this bytes

        try:
            while True:
                # i.e. always..
                bytes_flagment = bytes()
                # recieved data will be stored in this bytes
                bytes_flagment = bytes(self._sock.recv(self._buffer_size))
                self.save(bytes_flagment)

        except KeyboardInterrupt:
            # when ctrl+c comes
            print("Stopped daq!")


if __name__ == '__main__':

    client = Client()

    client.connect()
    print('buffer size default' +
          str(client._sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)))
    client._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1040000)

    client._file_path = "daqqed_data/" + \
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    client._file = open(client._file_path, mode="wb")

    client.recieve()
    client.disconnect()

    client._file.close()
