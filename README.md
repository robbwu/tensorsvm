# TensorSVM
TensorSVM is a high performance GPU based SVM tool.

## How to train

Sample cmdline:

### Disabling TensorCore
```
./tensorsvm-train -k 32 -c 32 -g 0.01 -t 2 SUSY.training SUSY.model
```

### Enabling TensorCore
```
./tensorsvm-train -tensorcore -k 32 -c 32 -g 0.01 -t 2 SUSY.training SUSY.model
```

### Parameters
Rank: -k
C: -c
gamma: -g
type: -t 0 Linear kernel, -t 2 RBF kernel
Enabling TensorCore: -tensorcore
'SUSY.training' is training file path
'SUSY.model' is output model

## How to predict
We are using ThunderSVM https://github.com/Xtra-Computing/thundersvm to predict.
```
./thundersvm-predict SUSY.testing SUSY.model output
```

Sample cmdline: 

```
./tensorsvm-train-mixed -k 400 -c 32 -g 0.0002 -t 2  /home/pwu6/LIBSVM_DATA/covtype.libsvm.binary.1 && /home/pwu6/thundersvm/build/bin/thundersvm-predict /home/pwu6/LIBSVM_DATA/covtype.libsvm.binary.2  /home/pwu6/LIBSVM_DATA/covtype.libsvm.binary.1.model output
```
