# purpose: to read (KC705-MCS's) raw binary file with some options

import os
import sys

# 辞書の表示を綺麗に
import pprint

# some caliculations
import numpy as np

# handle command line options
from argparse import ArgumentParser

# ofr graphic
import numpy as np
import matplotlib.pyplot as plt


def get_option():
    # define command line options
    argparser = ArgumentParser()
    argparser.add_argument(
        '-f', '--file', required=False, type=str, default=None, help='path to file')
    argparser.add_argument(
        '-s', '--spillinfo', action='store_true', help='print information of each spill')
    argparser.add_argument(
        '-n', '--nonzero', action='store_true', help='print relative time when there are hits of nonzero')
    argparser.add_argument('-gs', '--graphspill', type=int,
                           help='draw graph of certain spillcount')
    argparser.add_argument('-ch', '--channel', type=int, default=99,
                           help='draw graph of certain channel, use with -gs')

    return argparser.parse_args()


class bit:
    # 1 word, bytes
    SIZE_HEADER = 8
    SIZE_FOOTER = 8
    SIZE_DATA = 148
    DATA_TYPE = np.dtype(np.uint16)

    # header, bits
    SIZE_HEADER_MAGICWORD = 32
    HEADER_MAGICWORD = 0xAAAAAAAA
    SIZE_HEADER_SPACE = 12
    HEADER_SPACE = 0x000
    SIZE_HEADER_BUFFERLABEL = 4
    SIZE_HEADER_SPILLCOUNT = 16

    # footer
    SIZE_FOOTER_MAGICWORD = 16
    FOOTER_MAGICWORD = 0xFFFF
    SIZE_FOOTER_EVENTMATCH = 16
    SIZE_FOOTER_RECORDEDMRSYNC = 32


def plot_spill(data_with_a_spill):
    fig = plt.figure(figsize=(8, 6))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    ax1 = fig.add_subplot(2, 1, 1)

    # 転置して時間を横軸に
    ax1.imshow(data_with_a_spill.sum(axis=0).T,
               cmap='viridis', aspect='auto', origin='lower')
    ax1.set_title('spillcount:' + str(args.graphspill))
    ax1.set_xlabel('CLK')
    ax1.set_ylabel('bit-fields')

    if (args.channel != 99):
        ax2 = fig.add_subplot(2, 1, 2)
        x_range = data_with_a_spill.sum(axis=0).T.shape[1]
        y = data_with_a_spill.sum(
            axis=0).T[args.channel]
        entries = y.sum()
        ax2.step(np.arange(x_range) * 5, y, where='post',
                 label='Entries:' + str(entries))
        ax2.set_yscale('log')
        ax2.set_ylim(0.1, y.max() * 10)

        ax2.grid(axis="y")
        ax2.minorticks_on()
        ax2.grid(which="both", axis="x")

        ax2.legend()
        ax2.set_title('ch:' + str(args.channel))
        ax2.set_xlabel('ns')
        ax2.set_ylabel('Entries')

    plt.show()
    sys.exit()


def main(path_to_file):

    # アウトプットごとの情報
    data_with_a_output = np.zeros((0, 74), dtype=np.uint16)
    # スピルごとの情報
    data_with_a_spill = np.empty((0, 1088, 74), dtype=np.uint16)
    spill_count = -1
    spill_count_old = -1
    bufferlabel_list = []
    recordedmrsync_list = []
    # key: spillcount : [output_count, total_hit_in_a_spill]
    spill_info = {}
    with open(path_to_file, 'rb') as f:
        while f.tell() != os.path.getsize(path_to_file):

            bytearray_1word = f.read(bit.SIZE_HEADER)
            int_1word = int.from_bytes(bytearray_1word, 'big')

            if ((int_1word >> (4 * 8)) == bit.HEADER_MAGICWORD):
                # ヘッダーが来たら

                if (spill_count_old != -1):  # 最初の一回は飛ばすため
                    # アウトプットごとの(以前のアウトプットの, 以前のヘッダーに属する)のデータを詰める
                    data_with_a_spill = np.append(
                        data_with_a_spill, np.array([data_with_a_output]), axis=0)
                    # 初期化
                    data_with_a_output = np.empty((0, 74), dtype=np.uint16)

                bufferlabel_list.append((int_1word >> 16) & 0b1111)
                spill_count = (int_1word & 0xFF)
                if (spill_count != spill_count_old) and (spill_count_old != -1):

                    # 新しいスピルになったら
                    # スピルごと(以前のスピルの, 最初の一回目の偽切り替わりは無視するため!=-1)のデータを詰める
                    output_count = data_with_a_spill.shape[0]
                    total_hits_in_a_spill = data_with_a_spill.sum()
                    spill_info[spill_count_old] = np.array([
                        output_count, bufferlabel_list[0:-1], recordedmrsync_list, total_hits_in_a_spill], dtype=object)

                    if (args.graphspill == spill_count_old):
                        plot_spill(data_with_a_spill)

                    # &初期化
                    data_with_a_spill = np.empty(
                        (0, 1088, 74), dtype=np.uint16)
                    bufferlabel_list = [bufferlabel_list[-1]]
                    recordedmrsync_list = []

                spill_count_old = spill_count

            elif ((int_1word >> (6 * 8)) == bit.FOOTER_MAGICWORD):
                #  フッターが来たら
                recordedmrsync_list.append(int_1word & 0xFFFFFFFF)

                if (f.tell == os.path.getsize(path_to_file)):
                    # 最後の最後に限りここでアウトプットごとのデータを詰める(次のヘッダーがないから)
                    data_with_a_spill = np.append(
                        data_with_a_spill, np.array([data_with_a_output]), axis=0)
                    # 最後の最後に限りここでスピルごとのデータを詰める(次のヘッダーがないから)
                    output_count = data_with_a_spill.shape[0]
                    total_hits_in_a_spill = data_with_a_spill.sum()
                    spill_info[spill_count_old] = np.array([
                        output_count, bufferlabel_list, recordedmrsync_list, total_hits_in_a_spill], dtype=object)

            else:
                # データの場合
                f.seek(-1 * bit.SIZE_HEADER, 1)
                # relative timeごとのデータを詰める
                bytearray_data = f.read(bit.SIZE_DATA)
                data = np.frombuffer(bytearray_data, bit.DATA_TYPE)
                data_with_a_output = np.vstack((data_with_a_output, data))

                if args.nonzero and (np.count_nonzero(data != 0)):
                    # ヒットがあった場合のみアウトプットごとに情報表示モード
                    print('spillcount:', 'output no.:', 'bufferlabel', 'relative time(ns):', 'total hits:',
                          spill_count,  data_with_a_spill.shape[0] + 1, bufferlabel_list[-1],  data_with_a_output.shape[0] * 5, data.sum())

    if args.spillinfo:
        # 全部読み切ったら
        print('\nlist of spillcount:', spill_info.keys())
        # 辞書をキーごとに表示
        print(
            '{spillcount: [outputcount, [bufferlabel list], [recordedmrsync list], total hits in this spill]}')
        pprint.pprint(spill_info)


if __name__ == '__main__':
    args = get_option()

    path_to_file = args.file
    print('path_to_file:', path_to_file)
    main(path_to_file)
