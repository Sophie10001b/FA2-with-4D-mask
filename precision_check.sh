device="cuda:0"
dtype="bfloat16"
B=1
Tq=8191
Tk=8191
Hq=32
Hk=8
D=128

python ./benchmark/precision_check.py \
    --device $device \
    --dtype $dtype \
    --B $B \
    --Tq $Tq \
    --Tk $Tk \
    --Hq $Hq \
    --Hk $Hk \
    --D $D \
    --cases no_causal causal random_mask \
    --check-backward
