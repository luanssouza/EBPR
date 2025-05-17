python train_EBPR.py --model BPR --use_cuda false > train_bpr_ml100k.log 
python train_EBPR.py --model UBPR --use_cuda false > train_ubpr_ml100k.log 
python train_EBPR.py --model EBPR --use_cuda false > train_ebpr_ml100k.log 
python train_EBPR.py --model UEBPR --use_cuda false > train_uebpr_ml100k.log