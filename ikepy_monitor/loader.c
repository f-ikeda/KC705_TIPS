// before using, pls do "$ gcc -cpp -fPIC -shared loader.c -lm -o loader.so -O2"

#include <stdio.h>
// bool型
#include <stdbool.h>

// $ gcc -cpp -fPIC -shared loader.c -lm -o loader.so -O3

// bytes
// DIN    <= {HEADER[31:0],SPLCOUNT[15:0],4'd0,BOARD_ID[3:0],48'h0123_4567_89AB}; // HEADER =  REG_HEADER[31:0]
//            x08_Reg[7:0]    <= 8'h01;   // Header
//            x09_Reg[7:0]    <= 8'h23;   // Header
//            x0A_Reg[7:0]    <= 8'h45;   // Header
//            x0B_Reg[7:0]    <= 8'h67;   // Header
// DIN    <= {FOOTER[31:0],SPLCOUNT[15:0],EMCOUNT[15:0],wrCnt[31:0],8'hAB}; // FOOTER = REG_FOOTER[31:0]
//            x0C_Reg[7:0]    <= 8'hAA;   // Footer
//            x0D_Reg[7:0]    <= 8'hAA;   // Footer
//            x0E_Reg[7:0]    <= 8'hAA;   // Footer
//            x0F_Reg[7:0]    <= 8'hAA;   // Footer
// DIN    <= {SIG[76:0],COUNTER[26:0]}; // 104-bits
// {MainHodo[63:0],PMR[11:0],MR_Sync,COUNTER[26:0]}

// Header(上位32bit)の判定
bool header_or_not(const unsigned char *array_13bytes)
{
    if (((array_13bytes[0] << (8 * 3)) | (array_13bytes[1] << (8 * 2)) | (array_13bytes[2] << (8 * 1)) | array_13bytes[3]) == 0x01234567)
    {
        return true;
    }
    else
    {
        return false;
    }
}

// Footer(上位32bit)の判定
bool footer_or_not(const unsigned char *array_13bytes)
{
    if ((((unsigned int)array_13bytes[0] << (8 * 3)) + ((unsigned int)array_13bytes[1] << (8 * 2)) + ((unsigned int)array_13bytes[2] << (8 * 1)) + (unsigned int)array_13bytes[3]) == 0xAAAAAAAAu)
    {
        return true;
    }
    else
    {
        return false;
    }
}

// 13bytesについての操作
void packer(unsigned char *array_13bytes, unsigned long long *tmp_sig_mppc, unsigned short *tmp_sig_pmt, unsigned char *tmp_sig_mrsync, signed long long *tmp_tdc)
{
    /*
    // debug print
    for (int i = 0; i < 13; i++)
    {
        printf("%02X ", array_13bytes[i]);
    }
    printf("\n");
    */

    *tmp_sig_mppc = 0;
    for (int j = 0; j < 8; j++)
    {
        *tmp_sig_mppc += ((unsigned long long)array_13bytes[j] << (8 * (8 - j - 1)));
    }
    *tmp_sig_pmt = (((unsigned short)array_13bytes[8] << 4) | (((unsigned short)array_13bytes[9] & 0xF0) >> 4));
    *tmp_sig_mrsync = ((unsigned char)array_13bytes[9] & 0x08) >> 3;

    *tmp_tdc = (((signed long long)array_13bytes[9] & 0x07) << (8 * 3)) | ((signed long long)array_13bytes[10] << (8 * 2)) | ((signed long long)array_13bytes[11] << (8 * 1)) | (signed long long)array_13bytes[12];

    /*
    // debug print
    printf("tmp_sig_mppc: %016llX\n", *tmp_sig_mppc);
    printf("tmp_sig_pmt: %04hX\n", *tmp_sig_pmt);
    printf("tmp_sig_mrsync: %02hhX\n", *tmp_sig_mrsync);
    printf("tmp_tdc: %08llX\n", *tmp_tdc);
    */
}

// 指定したオフセット(offset_to_read、ヘッダーの位置)から、1spill分だけ読む
// 配列のサイズはpythonに任せる(mallocは余裕があったら)
void a_spill_loader(unsigned long long *sig_mppc, signed long long *tdc_mppc, signed long long *mrsync_mppc,
                    unsigned short *sig_pmt, signed long long *tdc_pmt, signed long long *mrsync_pmt,
                    unsigned char *sig_mrsync, signed long long *mrsync, unsigned int *indexes,
                    char *file_path, long *offset_to_read, long *data_num, long *skip_time)
// (sig_mppc, tdc_mppc, mrsync_mppc, sig_pmt, tdc_pmt, mrsync_pmt,  sig_mrsync, mrsync, indexes, file_path, offset_to_read, data_num)
{
    /*
    printf("   sig_mppc    %llu\n",    sig_mppc[0]);
    printf("   tdc_mppc    %lld\n",    tdc_mppc[0]);
    printf("mrsync_mppc    %lld\n", mrsync_mppc[0]);
    printf("   sig_pmt     %u\n"  ,    sig_pmt[0]);
    printf("   tdc_pmt     %lld\n",    tdc_pmt[0]);
    printf("mrsync_pmt     %lld\n", mrsync_pmt[0]);
    printf("   sig_mrsync  %u\n"  , (unsigned int)sig_mrsync[0]);
    printf("       mrsync  %lld\n",     mrsync[0]);
    printf("indexes        %u %u %u\n", indexes[0], indexes[1], indexes[2]);
    printf("filename       %s\n", file_path);
    printf("offset_to_read %ld\n", offset_to_read[0]);
    */

    FILE *pointer_file;
    pointer_file = fopen(file_path, "rb");
    // ファイルの読み込みチェック
    if (pointer_file == NULL)
    {
        printf("pointer_file: NULL");
        return;
    }
    // ヘッダーの位置までオフセットで飛ばす
    int seekret = fseek(pointer_file, offset_to_read[0] * 13, SEEK_SET);
    if (seekret)
    {
        printf("fseek error: (ret = %d)\n", seekret);
        return;
    }
    // タイムスタンプを飛ばす
    if (skip_time[0] == 999)
    {
        int seekret = fseek(pointer_file, 8, SEEK_CUR);
        if (seekret)
        {
            printf("fseek error: (ret = %d)\n", seekret);
            return;
        }
    }

    // bytes
    int data_unit = 13;
    unsigned char array_13bytes[13];

    // ある13bytesのデータ一個について
    unsigned long long tmp_sig_mppc;
    unsigned short tmp_sig_pmt;
    unsigned char tmp_sig_mrsync;
    signed long long tmp_tdc;
    signed long long tmp_mrsync = -1;

    // 配列にどこまで書き込んだか
    unsigned int tmp_index_mppc = 0;
    unsigned int tmp_index_pmt = 0;
    unsigned int tmp_index_mrsync = 0;

    // オーバーフローを考慮するため
    signed long long tmp_tdc_old = 0;
    unsigned int tmp_overflow_count = 0;

    // header_flagはいらない？
    bool header_flag = false;
    bool footer_flag = false;
    // 既にファイルポインタはヘッダーの位置まで来てるので、次にフッターが来るまでループ
    // while (!footer_flag)
    for (long count_loop = 0; count_loop < data_num[0]; count_loop++)
    {
        // 13bytesだけ読み込んで
        int readret = fread(array_13bytes, sizeof(char), data_unit, pointer_file);

        /*
        // int readret = fread(array_13bytes, data_unit, 1, pointer_file);
        if (readret != data_unit)
        {
            printf("fread error? (ret %d)\n", readret);
            return;
	    }
        */

        // ヘッダーの場合
        if (header_or_not(array_13bytes))
        {
            header_flag = true;
            // タイムスタンプを飛ばす
            if (skip_time[0] == 999)
            {
                int seekret = fseek(pointer_file, 8, SEEK_CUR);
                if (seekret)
                {
                    printf("fseek error: (ret = %d)\n", seekret);
                    return;
                }
            }
        }
        // フッターの場合
        else if (footer_or_not(array_13bytes))
        {
            footer_flag = true;
        }
        // TDCの場合
        else
        {
            // sig_とtdcを読み込んで
            packer(array_13bytes, &tmp_sig_mppc, &tmp_sig_pmt, &tmp_sig_mrsync, &tmp_tdc);

            //オーバーフローへの対処
            if (tmp_tdc - tmp_tdc_old < 0)
            {
                tmp_overflow_count += 1;
            }
            tmp_tdc_old = tmp_tdc;

            if (tmp_overflow_count > 0)
            {
                tmp_tdc += tmp_overflow_count * (2 << (27 - 1));
            }

            // mrsyncにヒットがあった場合
            if (tmp_sig_mrsync != 0)
            {
                tmp_mrsync = tmp_tdc;

                sig_mrsync[tmp_index_mrsync] = tmp_sig_mrsync;
                mrsync[tmp_index_mrsync] = tmp_mrsync;
                tmp_index_mrsync += 1;
            }

            // mppcにヒットがあった場合
            if (tmp_sig_mppc != 0)
            {
                sig_mppc[tmp_index_mppc] = tmp_sig_mppc;
                tdc_mppc[tmp_index_mppc] = tmp_tdc;
                // tmp_mrsyncは先に更新しておくこと
                mrsync_mppc[tmp_index_mppc] = tmp_mrsync;
                tmp_index_mppc += 1;
            }

            // pmtにヒットがあった場合
            if (tmp_sig_pmt != 0)
            {
                sig_pmt[tmp_index_pmt] = tmp_sig_pmt;
                tdc_pmt[tmp_index_pmt] = tmp_tdc;
                // tmp_mrsyncは先に更新しておくこと
                mrsync_pmt[tmp_index_pmt] = tmp_mrsync;
                tmp_index_pmt += 1;
            }
        }
    }

    indexes[0] = tmp_index_mppc;
    indexes[1] = tmp_index_pmt;
    indexes[2] = tmp_index_mrsync;

    // 最後にファイルを閉じて終了
    fclose(pointer_file);
}

// 13bytesについての操作
void get_spillcount_header(unsigned char *array_13bytes, unsigned long long *tmp_spillcount_header)
{
    *tmp_spillcount_header = 0;
    *tmp_spillcount_header += (unsigned long long)array_13bytes[4] << 8;
    *tmp_spillcount_header += (unsigned long long)array_13bytes[5];
    // printf("(unsigned long long)array_13bytes[5]: %llu\n", (unsigned long long)array_13bytes[5]);
}

void find_spills(unsigned long long *spillcount, unsigned long long *offset, unsigned long long *tdcnum, unsigned int *indexes,
                 char *file_path, long *data_num, long *skip_time)
{
    FILE *pointer_file;
    pointer_file = fopen(file_path, "rb");
    // ファイルの読み込みチェック
    if (pointer_file == NULL)
    {
        printf("pointer_file: NULL");
        return;
    }

    // bytes
    int data_unit = 13;
    unsigned char array_13bytes[13];

    unsigned long long tmp_offset = 0;
    unsigned long long readed_count = 0;
    unsigned long long spillcount_header = 0;
    unsigned long long readed_spills = 0;

    bool header_flag = false;
    for (long count_loop = 0; count_loop < data_num[0]; count_loop++)
    {
        // 13bytesだけ読み込んで
        int readret = fread(array_13bytes, sizeof(char), data_unit, pointer_file);
        // ヘッダーの場合
        if (header_or_not(array_13bytes))
        {
            // タイムスタンプを飛ばす
            if (skip_time[0] == 999)
            {
                int seekret = fseek(pointer_file, 8, SEEK_CUR);
                if (seekret)
                {
                    printf("fseek error: (ret = %d)\n", seekret);
                    return;
                }
            }

            tmp_offset = readed_count;
            get_spillcount_header(array_13bytes, &spillcount_header);
            // printf("spillcount_header: %llu\n", spillcount_header);
            // printf("data_num[0]: %lu\n", data_num[0]);

            header_flag = true;
        }
        // フッターの場合
        else if (footer_or_not(array_13bytes))
        {
            // ヘッダーを読んでいれば
            if (header_flag)
            {
                spillcount[readed_spills] = spillcount_header;
                offset[readed_spills] = tmp_offset;
                tdcnum[readed_spills] = readed_count - tmp_offset;
                readed_spills += 1;
                header_flag = false;
            }
        }
        // TDCの場合
        // else
        // {
        // }

        readed_count += 1;
        // printf("readed_count: %llu\n", readed_count);
    }

    indexes[0] = readed_spills - 1;
}