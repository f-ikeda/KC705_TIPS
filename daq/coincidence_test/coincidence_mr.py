# 仮定: BはAより遅れている
#      AとMRは同時

import numpy as np


def Case1():
    print('Case1()')

    coincidenced = []
    mrsync_tmp = []

    index_A = 0
    index_B = 0
    while index_A < len(sig) and index_B < len(sig):
        if(sig[index_A] & A != 0):
            tdc_A_latest = tdc[index_A]

            if(sig[index_B] & B != 0):
                tdc_B_latest = tdc[index_B]

                diff_B_A = tdc_B_latest - tdc_A_latest
                if(diff_B_A < DELAY):
                    index_B += 1
                    continue
                elif(diff_B_A > DELAY):
                    index_A += 1
                    continue
                else:
                    # つまりコインシデンスが取れたとき
                    coincidenced.append(tdc[index_A] - mrsync[index_A])
                    mrsync_tmp.append(mrsync[index_A])
                    index_A += 1
                    index_B += 1
                    continue
                    # index_A, index_B同士でコインシデンスが取れた時点で、index_A, index_Bのいずれかを+1したもの同士ではコインシデンスが取れないことは明白なため(幅1CLKの場合)
            else:
                index_B += 1
                continue
                # Bへのヒットを見つける

        else:
            index_A += 1
            continue
            # Aへのヒットを見つける

    print('len(sig) :' + str(len(sig)))
    print('len(tdc) :' + str(len(tdc)))
    print('len(mrsync) :' + str(len(mrsync)))
    print('coincidenced: ' + str(coincidenced))
    print('mrsync_tmp: ' + str(mrsync_tmp))


def coincidence(tuple_tdc_A_and_mrsync_A, tuple_tdc_B_and_mrsync_B):
    # Case1を下敷きに、再起処理のできるよう関数化
    # 重要なのは、インデックスの対応はあくまでAとBとで独立に満たせば十分ということ
    # 従って、入力: (tdc(A),mrsync(A)),(tdc(B),mrsync(B))
    # とし、出力: (tdc(coincidenced),mrsync(coincidenced))
    # と、2入力1出力のすべての形式を揃えることで、coincidence(C,coincidene(A,B))のように書くことができる
    # ここで、tdc(A)をあらわに書くと、np.extract((sig & A != 0), tdc)、mrsyncについても同様

    print('coincidence()')

    tdc_coincidenced = []
    mrsync_coincidenced = []

    index_A = 0
    index_B = 0
    while index_A < len(tuple_tdc_A_and_mrsync_A[0]) or index_B < len(tuple_tdc_B_and_mrsync_B[0]):
        tdc_A_latest = tuple_tdc_A_and_mrsync_A[0][index_A]
        tdc_B_latest = tuple_tdc_B_and_mrsync_B[0][index_B]

        diff_B_A = tdc_B_latest - tdc_A_latest
        if(diff_B_A < DELAY):
            index_B += 1
            continue
        elif(diff_B_A > DELAY):
            index_A += 1
            continue
        else:
            # つまりコインシデンスが取れたとき
            tdc_coincidenced.append(
                tuple_tdc_A_and_mrsync_A[0][index_A] - tuple_tdc_A_and_mrsync_A[1][index_A])
            mrsync_coincidenced.append(tuple_tdc_A_and_mrsync_A[1][index_A])
            index_A += 1
            index_B += 1
            continue
            # index_A, index_B同士でコインシデンスが取れた時点で、index_A, index_Bのいずれかを+1したもの同士ではコインシデンスが取れないことは明白なため(幅1CLKの場合)

    print('tdc_coincidenced: ' + str(tdc_coincidenced))
    print('mrsync_coincidenced: ' + str(mrsync_coincidenced))

    return tdc_coincidenced, mrsync_coincidenced


# detectors
A = 0b001
B = 0b010
# MR Sync
MR = 0b100

# テストデータ
sig = [MR+A, B+A, A, B, B+A, B, B, MR, B+A, A, A, MR+B, B, B, A]
tdc = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
mrsync = [0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 11, 11, 11, 11]
# 期待される結果 = [1, 2, 4, 2, 3]
#  mrsync_tmp   = [0, 0, 0, 7, 7]
# '''

'''
# テストデータ
sig = [MR, A, B, B+A, MR+B+A, A, B, MR, MR, MR, B+A, B+A, A, A, B]
tdc = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
mrsync = [0, 0, 0, 0, 4, 4, 4, 7, 8, 9, 9, 9, 9, 9, 9]
# 期待される結果 = [1, 0, 3]
#  mrsync_tmp   = [0, 4, 9]
# '''

DELAY = 2

Case1()

sig = np.array(sig)
tdc = np.array(tdc)
mrsync = np.array(mrsync)

condition_A = (sig & A != 0)
condition_B = (sig & B != 0)
tdc_A = np.extract(condition_A, tdc).tolist()
tdc_B = np.extract(condition_B, tdc).tolist()
mrsync_A = np.extract(condition_A, mrsync).tolist()
mrsync_B = np.extract(condition_B, mrsync).tolist()
coincidence((tdc_A, mrsync_A), (tdc_B, mrsync_B))
# 引数の渡し方に問題あるっぽい
