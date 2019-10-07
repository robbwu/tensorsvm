# tensorsvm

Sample cmdline: 

```
./tensorsvm-train-mixed -k 400 -c 32 -g 0.0002 -t 2  /home/pwu6/LIBSVM_DATA/covtype.libsvm.binary.1 && /home/pwu6/thundersvm/build/bin/thundersvm-predict /home/pwu6/LIBSVM_DATA/covtype.libsvm.binary.2  /home/pwu6/LIBSVM_DATA/covtype.libsvm.binary.1.model output
```

-analysis: (non-essential) computes and prints out the low rank approximation error
