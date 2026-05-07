python train.py --config config/Qwen3VL_2B_fp4.yaml
python add_perturbation.py --config config/Qwen3VL_2B_fp4.yaml

python train.py --config config/Qwen3VL_2B_fp4_nf4.yaml
python add_perturbation.py --config config/Qwen3VL_2B_fp4_nf4.yaml

python train.py --config config/Qwen3VL_2B_fp4_nf4_int8.yaml
python add_perturbation.py --config config/Qwen3VL_2B_fp4_nf4_int8.yaml