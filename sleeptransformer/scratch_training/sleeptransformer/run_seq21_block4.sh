CUDA_VISIBLE_DEVICES="11,-1" python3 train_sleeptransformer.py --eeg_train_data "../../file_list/scratch_training/eeg/train_list.txt" --eeg_eval_data "../../file_list/scratch_training/eeg/eval_list.txt" --eeg_test_data "../../file_list/scratch_training/eeg/test_list.txt" --eog_train_data "" --eog_eval_data "" --eog_test_data "" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir './scratch_training_1chan/n1/' --seq_len 21 --early_stopping True --num_blocks 4
CUDA_VISIBLE_DEVICES="11,-1" python3 train_sleeptransformer.py --eeg_train_data "../../file_list_shhs2/scratch_training/eeg/train_list.txt" --eeg_eval_data "../../file_list_shhs2/scratch_training/eeg/eval_list.txt" --eeg_test_data "../../file_list_shhs2/scratch_training/eeg/test_list.txt" --eog_train_data "" --eog_eval_data "" --eog_test_data "" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir './scratch_training_1chan_shhs2/n1/' --seq_len 21 --early_stopping True --num_blocks 4
CUDA_VISIBLE_DEVICES="11,-1" python3 train_sleeptransformer.py --eeg_train_data "../../file_list_shhs1/scratch_training/eeg/train_list.txt" --eeg_eval_data "../../file_list_shhs1/scratch_training/eeg/eval_list.txt" --eeg_test_data "../../file_list_shhs1/scratch_training/eeg/test_list.txt" --eog_train_data "" --eog_eval_data "" --eog_test_data "" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir './scratch_training_1chan_shhs1/n1/' --seq_len 21 --early_stopping True --num_blocks 4
CUDA_VISIBLE_DEVICES="11,-1" python3 train_sleeptransformer.py --eeg_train_data "../../file_list/scratch_training/eeg/train_list.txt" --eeg_eval_data "../../file_list/scratch_training/eeg/eval_list.txt" --eeg_test_data "../../file_list/scratch_training/eeg/test_list.txt" --eog_train_data "../../file_list/scratch_training/eog/train_list.txt" --eog_eval_data "../../file_list/scratch_training/eog/eval_list.txt" --eog_test_data "../../file_list/scratch_training/eog/test_list.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir './scratch_training_2chan/n1/' --seq_len 21 --early_stopping True --num_blocks 4
CUDA_VISIBLE_DEVICES="11,-1" python3 train_sleeptransformer.py --eeg_train_data "../../file_list/scratch_training/eeg/train_list.txt" --eeg_eval_data "../../file_list/scratch_training/eeg/eval_list.txt" --eeg_test_data "../../file_list/scratch_training/eeg/test_list.txt" --eog_train_data "../../file_list/scratch_training/eog/train_list.txt" --eog_eval_data "../../file_list/scratch_training/eog/eval_list.txt" --eog_test_data "../../file_list/scratch_training/eog/test_list.txt" --emg_train_data "../../file_list/scratch_training/emg/train_list.txt" --emg_eval_data "../../file_list/scratch_training/emg/eval_list.txt" --emg_test_data "../../file_list/scratch_training/emg/test_list.txt" --out_dir './scratch_training_3chan/n1/' --seq_len 21 --early_stopping True --num_blocks 4

python3 train_sleeptransformer.py --eeg_train_data "../../file_list_snuh_100/scratch_training/eeg/train_list.txt" --eeg_eval_data "../../file_list_snuh_100/scratch_training/eeg/eval_list.txt" --eeg_test_data "../../file_list_snuh_100/scratch_training/eeg/test_list.txt" --eog_train_data "" --eog_eval_data "" --eog_test_data "" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir './scratch_training_1chan_snuh_100/n1/' --seq_len 21 --early_stopping True --num_blocks 4
python3 train_sleeptransformer.py --eeg_train_data "../../file_list_snuh_100_prep/scratch_training/eeg/train_list.txt" --eeg_eval_data "../../file_list_snuh_100_prep/scratch_training/eeg/eval_list.txt" --eeg_test_data "../../file_list_snuh_100_prep/scratch_training/eeg/test_list.txt" --eog_train_data "" --eog_eval_data "" --eog_test_data "" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir './scratch_training_1chan_snuh_100_prep/n1/' --seq_len 21 --early_stopping True --num_blocks 4

python3 test_sleeptransformer.py --eeg_train_data "../../file_list/scratch_training/eeg/train_list.txt" --eeg_eval_data "../../file_list/scratch_training/eeg/eval_list.txt" --eeg_test_data "../../file_list/scratch_training/eeg/test_list.txt" --eog_train_data "" --eog_eval_data "" --eog_test_data "" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir './scratch_training_1chan/n1/' --seq_len 21 --num_blocks 4
python3 test_sleeptransformer.py --eeg_train_data "../../file_list_snuh_100/scratch_training/eeg/train_list.txt" --eeg_eval_data "../../file_list_snuh_100/scratch_training/eeg/eval_list.txt" --eeg_test_data "../../file_list_snuh_100/scratch_training/eeg/test_list.txt" --eog_train_data "" --eog_eval_data "" --eog_test_data "" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir './scratch_training_1chan_snuh_100/n1/' --seq_len 21 --num_blocks 4
python3 test_sleeptransformer.py --eeg_train_data "../../file_list_shhs2/scratch_training/eeg/train_list.txt" --eeg_eval_data "../../file_list_shhs2/scratch_training/eeg/eval_list.txt" --eeg_test_data "../../file_list_shhs2/scratch_training/eeg/test_list.txt" --eog_train_data "" --eog_eval_data "" --eog_test_data "" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir './scratch_training_1chan_shhs2/n1/' --seq_len 21 --num_blocks 4
python3 test_sleeptransformer.py --eeg_train_data "../../file_list_shhs1/scratch_training/eeg/train_list.txt" --eeg_eval_data "../../file_list_shhs1/scratch_training/eeg/eval_list.txt" --eeg_test_data "../../file_list_shhs1/scratch_training/eeg/test_list.txt" --eog_train_data "" --eog_eval_data "" --eog_test_data "" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir './scratch_training_1chan_shhs1/n1/' --seq_len 21 --num_blocks 4