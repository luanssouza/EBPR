mkdir Output
mkdir Output/checkpoints 
mkdir Output/results

python train_EBPR.py --model BPR --use_cuda true
python train_EBPR.py --model UBPR --use_cuda true 
python train_EBPR.py --model EBPR --use_cuda true
python train_EBPR.py --model UEBPR --use_cuda true

python train_EBPR.py --model BPR --use_cuda true --dataset ml-1m
python train_EBPR.py --model UBPR --use_cuda true --dataset ml-1m
python train_EBPR.py --model EBPR --use_cuda true --dataset ml-1m
python train_EBPR.py --model UEBPR --use_cuda true --dataset ml-1m

python train_EBPR.py --model BPR --use_cuda true --dataset coat
python train_EBPR.py --model UBPR --use_cuda true --dataset coat
python train_EBPR.py --model EBPR --use_cuda true --dataset coat
python train_EBPR.py --model UEBPR --use_cuda true --dataset coat

python train_EBPR.py --model BPR --use_cuda true --dataset eletronics
python train_EBPR.py --model UBPR --use_cuda true --dataset eletronics
python train_EBPR.py --model EBPR --use_cuda true --dataset eletronics
python train_EBPR.py --model UEBPR --use_cuda true --dataset eletronics

python train_EBPR.py --model BPR --use_cuda true --dataset amazon
python train_EBPR.py --model UBPR --use_cuda true --dataset amazon
python train_EBPR.py --model EBPR --use_cuda true --dataset amazon
python train_EBPR.py --model UEBPR --use_cuda true --dataset amazon

python train_EBPR.py --model BPR --use_cuda true --dataset rentrunway
python train_EBPR.py --model UBPR --use_cuda true --dataset rentrunway
python train_EBPR.py --model EBPR --use_cuda true --dataset rentrunway
python train_EBPR.py --model UEBPR --use_cuda true --dataset rentrunway