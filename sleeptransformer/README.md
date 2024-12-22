# SleepTransformer

It's the repository for SleepTransformer
- https://ieeexplore.ieee.org/abstract/document/9697331

## How to use

You need to fix the folder paths in all the files.

1. preprocessing

- `shhs_data.m` for SHHS

```
matlab -batch "shhs_data"
```

- `snuh_data.ipynb` for SNUH


2. convert

- `convert.m` for SNUH

```
matlab -batch "convert"
```

- `reindex.py` for SHHS

```
python reindex.py
```

3. data split

- `data_split_eval.m` for SNUH, SHHS

```
matlab -batch "data_split_eval"
```

4. genlist

- `genlist_scratch_training.m` for SNUH, SHHS

```
matlab -batch "genlist_scratch_training"
```

5. train

```
python3 train_sleeptransformer.py --eeg_train_data "../../file_list_snuh_100/scratch_training/eeg/train_list.txt" --eeg_eval_data "../../file_list_snuh_100/scratch_training/eeg/eval_list.txt" --eeg_test_data "../../file_list_snuh_100/scratch_training/eeg/test_list.txt" --eog_train_data "" --eog_eval_data "" --eog_test_data "" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir './scratch_training_1chan_snuh_100/n1/' --seq_len 21 --early_stopping True --num_blocks 4
```

6. test

```
python3 test_sleeptransformer.py --eeg_train_data "../../file_list_snuh_100/scratch_training/eeg/train_list.txt" --eeg_eval_data "../../file_list_snuh_100/scratch_training/eeg/eval_list.txt" --eeg_test_data "../../file_list_snuh_100/scratch_training/eeg/test_list.txt" --eog_train_data "" --eog_eval_data "" --eog_test_data "" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir './scratch_training_1chan_snuh_100/n1/' --seq_len 21 --num_blocks 4
```

7. eval

- `aggregate_sleeptransformer.m` for SNUH, SHHS

```
matlab -batch "aggregate_sleeptransformer"
```