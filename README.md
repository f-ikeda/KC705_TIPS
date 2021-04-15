# KC705_TIPS
for using KC705 with SiTCP

- mcs-reader.py
    - purpose: for debugging of kc705's mcs-mode
    - how to use  
plz look at  
`$ python3 mcs-reader.py -h`
    - first, after daq,  
`$ python3 mcs-reader.py -f foo.dat -s`  
this -s option tells some info of each spill
    - second, if u want to look at some hits, then  
`$ python3 mcs-reader.py -f foo.dat -n`
    - third, if u find spills with hits using above -s or -n options, then  
`$ python3 mcs-reader.py -f foo.dat -gs [spill count]`  
this can plot the accumrated mcs spectra of a spill
    - fourth, if u want to look at certain channel,  
`$ python3 mcs-reader.py -f foo.dat -gs [spill count] -ch [channel no.]`  
this can plot the accumrated mcs spectra of a channel of a spill
