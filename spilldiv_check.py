# purpose: to check the behaviour of spill division function in MCS-mode

# look .v source @ https://github.com/yfujii02/TDCKC705/blob/97f929a42d6964ba140df2789492e7eb19a70705/extinction.srcs/sources_1/new/top_mcs.v#L172
# x1A_value = SPILLDIV
################
#        if(RESET) begin
#            sw_mem <= 1'b0;
#        end else if (ENABLE) begin
#            if ((|NMRSYNC[10:0])==1'b0) begin
#                case (SPILLDIV)
#                    4'h0 : sw_mem <= (NMRSYNC[11:11]==1'h1)? 1'b1 : 1'b0;
#                    4'h1 : sw_mem <= (NMRSYNC[12:11]==2'h2)? 1'b1 : 1'b0;
#                    4'h2 : sw_mem <= (NMRSYNC[13:11]==3'h4)? 1'b1 : 1'b0;
#                    4'h3 : sw_mem <= (NMRSYNC[14:11]==4'h8)? 1'b1 : 1'b0;
#                    4'h4 : sw_mem <= (NMRSYNC[15:11]==5'h10)? 1'b1 : 1'b0;
#                    4'h5 : sw_mem <= (NMRSYNC[16:11]==6'h20)? 1'b1 : 1'b0;
#                    4'h6 : sw_mem <= (NMRSYNC[17:11]==7'h40)? 1'b1 : 1'b0;
#                    4'h7 : sw_mem <= (NMRSYNC[18:11]==8'h80)? 1'b1 : 1'b0;
#                    4'h8 : sw_mem <= (NMRSYNC[19:11]==9'h100)? 1'b1 : 1'b0;
#                    4'h9 : sw_mem <= (NMRSYNC[20:11]==10'h200)? 1'b1 : 1'b0;
#                    default : sw_mem <= 1'b0;
#                endcase
#            end else begin
#                sw_mem <= 1'b0;
#            end
#        end else begin
#            sw_mem <= 1'b0;
#        end
################

import sys

args = sys.argv
x1A_value = int(args[1])
print('(2 ** (x1A_value + 1) - 1):', bin((2 ** (x1A_value + 1) - 1)))

count_old = 0
for i in range(500000):
    mrsync = i
    if (((mrsync & 0b11111111111) == 0b00000000000)
            & (((mrsync >> 11) & (2 ** (x1A_value + 1) - 1)) == (2 ** x1A_value))):
        print('mrsync:', mrsync,
              'diff:', (mrsync - count_old))
        count_old = mrsync
