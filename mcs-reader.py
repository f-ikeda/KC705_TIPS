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
# log scale colorbar with imshow
import matplotlib.colors as mcolors

# オプションによっては、最後のスピルを表示できないかも(-gsと同様に対処)


class pycolor:
    # use as print(pycolor.RED, 'foo', pycolor.END)
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RETURN = '\033[07m'
    ACCENT = '\033[01m'
    FLASH = '\033[05m'
    RED_FLASH = '\033[05;41m'
    END = '\033[0m'


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
    argparser.add_argument(
        '-gt', '--graphtotalhit', action='store_true', help='draw graph of total hits for each spill')
    argparser.add_argument(
        '-ot', '--output', type=int, help='draw graph of recorded mrsyncs and total hits for each output of certain spillcount')
    argparser.add_argument(
        '-df', '--outputdiff', action='store_true', help='with -ot, draw hist of diff recorded mrsyncs and total hits for each output of certain spillcount')
    argparser.add_argument(
        '-if', '--skip', action='store_true', help='draw graph with skip initial and final spills with -gt')
    argparser.add_argument(
        '-mt', '--mtplot', action='store_true', help='draw mt. plot (1 file/spillが前提、要注意)')
    argparser.add_argument(
        '-t', '--time', action='store_true', help='skip 8bytes timestamp after header')

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


def prlot_mt(data_with_a_spill):

    fig = plt.figure(figsize=(8, 6))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    ax1 = fig.add_subplot(1, 1, 1)

    output_times = data_with_a_spill.shape[0]  # axis=0は出力の回数でラベルされている
    bin_size = data_with_a_spill.shape[1]  # binの数、1088になるはず
    mt_array = data_with_a_output = np.zeros(
        (output_times, bin_size), dtype=np.uint16)
    for i in range(output_times):
        mt_array[i] = data_with_a_spill[i].sum(axis=2)

    # make log scale colorbar
    norm_mtplot = mcolors.SymLogNorm(
        linthresh=1, vmin=0.8, vmax=mt_array.max()*10)
    # エントリなければ色塗らない
    cmap_mtplot = plt.cm.viridis
    cmap_mtplot.set_under('white')

    # entiries = data_with_a_spill.sum(axis=0).T.sum()
    extries = 0

    # 転置して時間を横軸に
    img_imshow = ax1.imshow(mt_array,
                            cmap=cmap_mtplot, aspect='auto', origin='lower', norm=norm_mtplot, interpolation='none')
    fig.colorbar(img_imshow, label='Entiries:' +
                 str(entries), orientation='vertical')
    ax1.set_title('mt plot')
    ax1.set_xlabel('CLK')
    ax1.set_ylabel('output times')

    plt.show()
    sys.exit()


def plot_spill(data_with_a_spill):

    fig = plt.figure(figsize=(8, 6))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    ax1 = fig.add_subplot(2, 1, 1)

    # make log scale colorbar
    norm_hitmap = mcolors.SymLogNorm(
        linthresh=1, vmin=0.8, vmax=data_with_a_spill.sum(axis=0).T.max()*10)
    # エントリなければ色塗らない
    cmap_hitmap = plt.cm.viridis
    cmap_hitmap.set_under('white')

    entiries = data_with_a_spill.sum(axis=0).T.sum()

    # 転置して時間を横軸に
    img_imshow = ax1.imshow(data_with_a_spill.sum(axis=0).T,
                            cmap=cmap_hitmap, aspect='auto', origin='lower', norm=norm_hitmap, interpolation='none')
    fig.colorbar(img_imshow, label='Entiries:' +
                 str(entiries), orientation='vertical')
    ax1.set_title('spillcount:' + str(args.graphspill))
    ax1.set_xlabel('CLK')
    ax1.set_ylabel('bit-fields')

    if (args.channel != 99):
        ax2 = fig.add_subplot(2, 1, 2)
        x_range = data_with_a_spill.sum(axis=0).T.shape[1]
        y = data_with_a_spill.sum(
            axis=0).T[args.channel]
        entries_ch = y.sum()
        ax2.step(np.arange(x_range) * 5, y, where='post',
                 label='Entries:' + str(entries_ch))
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


def plot_totalhit(spill_info):

    if args.skip:
        first_key = next(iter(spill_info), None)
        del spill_info[first_key]
        last_key = next(reversed(spill_info), None)
        del spill_info[last_key]

    fig = plt.figure(figsize=(8, 6))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    ax1 = fig.add_subplot(1, 1, 1)
    x = []
    for key in spill_info.keys():
        x.append(key)
    y = np.array([info[-3] for info in list(spill_info.values())])
    entries = y.sum()
    ax1.step(x, y, where='post', label='Entries:' + str(entries))

    # print entires for each spill
    for i in range(len(x)):
        ax1.text(x[i], y.max() * 2, str(y[i]), size='small')

    ax1.set_yscale('log')
    ax1.set_ylim(0.1, y.max() * 10)

    ax1.grid(axis="y")
    ax1.minorticks_on()
    ax1.grid(which="both", axis="x")

    ax1.legend()
    ax1.set_title('spillcount vs. total hits')
    ax1.set_xlabel('spillcount')
    ax1.set_ylabel('Entries')

    plt.show()
    sys.exit()


def plot_mrsyncs(spill_info, spillcount):

    fig = plt.figure(figsize=(8, 6))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    ax1 = fig.add_subplot(2, 1, 1)
    outputcount = spill_info[0]
    x = range(1, outputcount+1)
    # recorded mrsyncs lists(/output)
    y1 = np.array(spill_info[-5])
    entries_y1 = y1[-1]
    ax1.step(x, y1, 'C0', where='post', label='#RecordedMRSync' +
             '\nTotal/spill:' + str(entries_y1), alpha=0.6)

    # recorded totalhits lists(/output)
    ax2 = ax1.twinx()
    y2 = np.array(spill_info[-2])
    entries_y2 = y2.sum()
    ax2.step(x, y2, 'C1', where='post', label='hits' + '\nTotal/spill:' +
             str(entries_y2) + '\nTotal outputs times:' + str(outputcount), alpha=0.6)

    ax1.set_ylim(0, y1.max() * 1.2)
    ax2.set_ylim(0, y2.max() * 1.2)

    ax1.grid(axis="y")
    ax1.minorticks_on()
    ax1.grid(which="both", axis="x")

    # legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='lower right')

    ax1.set_title('spillcount:' + str(spillcount) +
                  '\noutputcounts vs. #RecordedMRSync, hits/output')
    ax1.set_xlabel('output times')
    ax1.set_ylabel('Entiries (#RecordedMRSync)')
    ax2.set_ylabel('Entiries (hits)')

    if args.outputdiff:
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        y1_diff = np.diff(np.array(y1))
        ax3.hist(y1_diff, bins=50, log=True)
        ax3.set_title('diff of #RecordedMRSync between outputs')
        ax3.set_ylabel('Entiries')

        ax3.grid(axis="y")
        ax3.minorticks_on()
        ax3.grid(which="both", axis="x")

        y2_diff = np.diff(np.array(y2, dtype='i8'))
        ax4.hist(y2_diff, bins=50, log=True)
        ax4.set_title('diff of hits/output between outputs')
        ax4.set_ylabel('Entiries')

        ax4.grid(axis="y")
        ax4.minorticks_on()
        ax4.grid(which="both", axis="x")

    plt.show()
    sys.exit()


def main(path_to_file):
    # bytes
    readed_size = 0

    # macで必要
    bit.DATA_TYPE = bit.DATA_TYPE.newbyteorder('>')

    # アウトプットごとの情報
    data_with_a_output = np.zeros((0, 74), dtype=np.uint16)
    hit_ch_list = np.empty(0, dtype=np.uint16)
    # スピルごとの情報
    data_with_a_spill = np.empty((0, 1088, 74), dtype=np.uint16)
    spill_count = -1
    spill_count_old = -1
    bufferlabel_list = []
    recordedmrsync_list = []
    em_list = []
    total_hit_in_outputs_list = []
    hit_ch_unique = np.empty(0, dtype=np.uint16)
    # key: spillcount : [output_count, total_hit_in_a_spill]
    spill_info = {}
    with open(path_to_file, 'rb') as f:
        while f.tell() != os.path.getsize(path_to_file):

            bytearray_1word = f.read(bit.SIZE_HEADER)
            readed_size += bit.SIZE_HEADER
            int_1word = int.from_bytes(bytearray_1word, 'big')

            if ((int_1word >> (4 * 8)) == bit.HEADER_MAGICWORD):
                # ヘッダーが来たら
                if args.time:
                    timestamp_8bytes = f.read(8)
                    readed_size += bit.SIZE_HEADER

                if (spill_count_old != -1):  # 最初の一回は飛ばすため
                    # アウトプットごとの(以前のアウトプットの, 以前のヘッダーに属する)のデータを詰める
                    total_hit_in_outputs_list.append(data_with_a_output.sum())
                    hit_ch_unique = np.insert(
                        hit_ch_unique, hit_ch_unique.size, hit_ch_list)
                    data_with_a_spill = np.append(
                        data_with_a_spill, np.array([data_with_a_output]), axis=0)
                    # 初期化
                    data_with_a_output = np.empty((0, 74), dtype=np.uint16)
                    hit_ch_list = np.empty(0, dtype=np.uint16)

                bufferlabel_list.append((int_1word >> 16) & 0b1111)
                spill_count = (int_1word & 0xFFFF)
                if (spill_count != spill_count_old) and (spill_count_old != -1):

                    # 新しいスピルになったら
                    # スピルごと(以前のスピルの, 最初の一回目の偽切り替わりは無視するため!=-1)のデータを詰める
                    output_count = data_with_a_spill.shape[0]
                    total_hits_in_a_spill = data_with_a_spill.sum()
                    spill_info[spill_count_old] = np.array([
                        output_count, bufferlabel_list[0:-1], recordedmrsync_list, em_list, total_hits_in_a_spill, total_hit_in_outputs_list, list(set(hit_ch_unique))], dtype=object)

                    if (args.graphspill == spill_count_old):
                        plot_spill(data_with_a_spill)
                    if (args.output == spill_count_old):
                        plot_mrsyncs(
                            spill_info[spill_count_old], spill_count_old)

                    # &初期化
                    data_with_a_spill = np.empty(
                        (0, 1088, 74), dtype=np.uint16)
                    bufferlabel_list = [bufferlabel_list[-1]]
                    recordedmrsync_list = []
                    em_list = []
                    total_hit_in_outputs_list = []
                    hit_ch_unique = np.empty(0, dtype=np.uint16)

                spill_count_old = spill_count

            elif ((int_1word >> (6 * 8)) == bit.FOOTER_MAGICWORD):
                #  フッターが来たら
                em_list.append((int_1word >> 32) & 0xFFFF)
                recordedmrsync_list.append(int_1word & 0xFFFFFFFF)

                # f.tell() == os.path.getsize(path_to_file) method doesnt work
                if (readed_size == os.path.getsize(path_to_file)):
                    # 最後の最後に限りここでアウトプットごとのデータを詰める(次のヘッダーがないから)
                    data_with_a_spill = np.append(
                        data_with_a_spill, np.array([data_with_a_output]), axis=0)
                    total_hit_in_outputs_list.append(data_with_a_output.sum())
                    hit_ch_unique = np.insert(
                        hit_ch_unique, hit_ch_unique.size, hit_ch_list)
                    # 最後の最後に限りここでスピルごとのデータを詰める(次のヘッダーがないから)
                    output_count = data_with_a_spill.shape[0]
                    total_hits_in_a_spill = data_with_a_spill.sum()
                    spill_info[spill_count_old] = np.array([
                        output_count, bufferlabel_list, recordedmrsync_list, em_list, total_hits_in_a_spill, total_hit_in_outputs_list, list(set(hit_ch_unique))],  dtype=object)

                    if (args.graphspill == spill_count_old):
                        plot_spill(data_with_a_spill)

                    if args.mtplot:
                        prlot_mt(data_with_a_spill)
            else:
                # データの場合
                f.seek(-1 * bit.SIZE_HEADER, 1)
                readed_size -= bit.SIZE_HEADER
                # relative timeごとのデータを詰める
                bytearray_data = f.read(bit.SIZE_DATA)
                readed_size += bit.SIZE_DATA
                data = np.frombuffer(bytearray_data, bit.DATA_TYPE)
                data_with_a_output = np.vstack((data_with_a_output, data))
                hit_ch_list = np.insert(
                    hit_ch_list, hit_ch_list.size, np.nonzero(data)[0])

                if args.nonzero and (np.count_nonzero(data != 0)):
                    # ヒットがあった場合のみアウトプットごとに情報表示モード
                    print('spillcount:', 'output no.:', 'bufferlabel', 'relative time(ns):', 'total hits:',
                          spill_count,  data_with_a_spill.shape[0] + 1, bufferlabel_list[-1],  data_with_a_output.shape[0] * 5, data.sum())

    '''
    if args.spillinfo:
        # 全部読み切ったら
        print('\nlist of spillcount:', spill_info.keys())
        # 辞書をキーごとに表示
        print(
            '{spillcount: [outputcount, [bufferlabel list], [recordedmrsync list], [e.m. list], total hits in this spill, [total hits in output list], [hit chs]]}')
        pprint.pprint(spill_info)
    '''

    if args.spillinfo:
        print('\nlist of spillcount:', spill_info.keys())
        for spill_i in spill_info.keys():
            # 辞書をキーごとに表示
            print(pycolor.RED, '-------- -------- soillcount:',
                  spill_i, pycolor.END)
            print(pycolor.GREEN, 'hits/spill:',
                  spill_info[spill_i][4], pycolor.END)
            print(pycolor.BLUE, 'hits chs list:',
                  spill_info[spill_i][6], pycolor.END)
            print('outputcount:', spill_info[spill_i][0])
            print('recordedmrsync:', spill_info[spill_i][2])
            print('hits/output:', spill_info[spill_i][5])
            print('bufferlabel:', spill_info[spill_i][1])
            print('e.m.:', spill_info[spill_i][3])

    if args.graphtotalhit:
        plot_totalhit(spill_info)


if __name__ == '__main__':
    args = get_option()

    path_to_file = args.file
    print(pycolor.CYAN, 'path_to_file:', path_to_file, pycolor.END)
    main(path_to_file)
