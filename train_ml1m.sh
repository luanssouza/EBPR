python train_EBPR.py --model BPR --use_cuda false --dataset ml-1m > train_bpr_ml1m.log 
python train_EBPR.py --model UBPR --use_cuda false --dataset ml-1m > train_ubpr_ml1m.log 
python train_EBPR.py --model EBPR --use_cuda false --dataset ml-1m > train_ebpr_ml1m.log 
python train_EBPR.py --model UEBPR --use_cuda false --dataset ml-1m > train_uebpr_ml1m.log