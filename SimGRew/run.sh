# python -u main.py --dataset 'Cora' --train_lr 0.01 --val_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.10 --train_w_decay 0.0005 --val_w_decay 0.0005 --th None --device 'cuda:0' | tee Cora.txt

# python -u main.py --dataset 'Citeseer' --train_lr 0.01 --val_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.10 --train_w_decay 0.0005 --val_w_decay 0.0005 --th 1.0 --device 'cuda:0' | tee Citeseer.txt

# python -u main.py --dataset 'Pubmed' --train_lr 0.01 --val_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.10 --train_w_decay 0.0005 --val_w_decay 0.0005 --th None --device 'cuda:0' | tee Pubmed.txt

# python -u main.py --dataset 'Chameleon' --train_lr 0.01 --val_lr 0.0 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --val_w_decay 0.0 --th 0.0 --device 'cuda:0' | tee Chameleon2.txt


# python -u main.py --dataset 'Film' --train_lr 0.01 --val_lr 0.0 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --val_w_decay 0.0 --th None --device 'cuda:0' | tee Film.txt


# python -u main.py --dataset 'Squirrel' --train_lr 0.01 --val_lr 0.0 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --val_w_decay 0.0 --th 0.0 --device 'cuda:0' | tee Squirrel2.txt


# python -u main.py --dataset 'Cornell' --train_lr 0.01 --val_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --val_w_decay 0.0005 --th None --device 'cuda:0' | tee Cornell.txt


# python -u main.py --dataset 'Texas' --train_lr 0.01 --val_lr 0.0 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --val_w_decay 0.05 --th 0.0  --device 'cuda:0' | tee Texas.txt 


# python -u main.py --dataset 'Wisconsin' --train_lr 0.01 --val_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 128 --train_iter 1000 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --val_w_decay 0.0005 --th None --device 'cuda:0' | tee Wisconsin.txt


# --------------------------------------------------------------------------------------------------------------------------

# python -u large_scale_graphs_main.py --dataset 'arxiv-year' --train_lr 0.001 --val_lr 0.0 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 30 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --val_w_decay 0.0 --th None --num_parts 100 --num_splits 5 --device 'cuda:0' | tee arxiv_year.txt


# python -u large_scale_graphs_main.py --dataset 'snap-patents' --train_lr 0.01 --val_lr 0.0 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --val_w_decay 0.0 --th None --num_parts 1000 --num_splits 5 --device 'cuda:0' | tee snap_patents.txt

# python -u large_scale_graphs_main.py --dataset 'Penn94' --train_lr 0.01 --val_lr 0.0 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --val_w_decay 0.0 --th None --num_parts 10 --num_splits 5 --device 'cuda:0' | tee Penn94.txt


# python -u large_scale_graphs_main.py --dataset 'pokec' --train_lr 0.01 --val_lr 0.0 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --val_w_decay 0.0 --th None --num_parts 500 --num_splits 5 --device 'cuda:0' | tee pokec.txt

# python -u large_scale_graphs_main.py --dataset 'twitch-gamers' --train_lr 0.001 --val_lr 0.0 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 100 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --val_w_decay 0.0 --th None --num_parts 100 --num_splits 5 --device 'cuda:0' | tee twitch_gamers.txt


python -u large_scale_graphs_main.py --dataset 'genius' --train_lr 0.001 --val_lr 0.0 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --val_w_decay 0.0 --th 0.0 --num_parts 100 --num_splits 5 --device 'cuda:0' | tee output/genius.txt


#------------------------------------------------------------------------------------------------------------------------------------------------------------------


# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --val_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --val_w_decay 0.0005 --th None --num_parts 50 --num_splits 10 --device 'cuda:0'


# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --val_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 2 --hidden_dim 64 --train_iter 20 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --val_w_decay 0.0005 --th None --num_parts 50 --num_splits 10 --device 'cuda:0'


# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --val_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --val_w_decay 0.0005 --th None --num_parts 50 --num_splits 10 --device 'cuda:0'


# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.001 --val_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.10 --train_w_decay 0.0005 --val_w_decay 0.0005 --th 1.0 --num_parts 10 --num_splits 1 --device 'cuda:0'


# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --val_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --de_lambda 0.0 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --val_w_decay 0.0005 --th None --num_parts 50 --num_splits 10 --device 'cuda:0'



# ----------------------------------------------------

