WAV_FILE=./audio_seven.wav

GLOG_minloglevel=2 python scmddet_run.py --input_wav $WAV_FILE --output prog.txt

cat filter_w.txt | xxd -r -p > filter_w.temp.bin
xxd -p -e -g 2 filter_w.temp.bin > filter_w.temp.txt
xxd -r filter_w.temp.txt > filter_w.bin
rm filter_w.temp.bin filter_w.temp.txt
