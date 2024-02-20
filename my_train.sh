CUDA_VISIBLE_DEVICES=1
python ./XLTU/main.py --data_dir_sl=./data/sl_EN_2_UG/ --data_dir_bc=./data/bc_EN_2_UG/ --output_dir=./my_model/my_model_CNN_CRF28_EN_2_UG_results/ --num_train_epochs=40 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=XLMR --model_size=base
python ./XLTU/map_results.py --data_path ./my_model/my_model_CNN_CRF28_EN_2_UG_results/

