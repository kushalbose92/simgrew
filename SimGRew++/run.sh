# python -u main.py --dataset 'Cora' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --alpha 5.0 --device 'cuda:0'

# python -u main.py --dataset 'Citeseer' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.10 --train_w_decay 0.0005 --th 1.0 --alpha 1.0 --device 'cuda:0'

# python -u main.py --dataset 'Pubmed' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.10 --train_w_decay 0.0005 --th None --alpha 1.0 --device 'cuda:0'

# python -u main.py --dataset 'Chameleon' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --alpha 5.0 --device 'cuda:0' | tee Chameleon.txt



# python -u main.py --dataset 'Film' --train_lr 0.1 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --alpha 5.0 --device 'cuda:0' | tee Film.txt


# python -u main.py --dataset 'Squirrel' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --alpha 5.0 --device 'cuda:0' | tee Squirrel.txt

# python -u main.py --dataset 'Cornell' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.0005 --th None --alpha 0.0 --device 'cuda:0' | tee Cornell.txt


# python -u main.py --dataset 'Texas' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.0005 --th None  --alpha 1.0 --device 'cuda:0' | tee Texas.txt


# python -u main.py --dataset 'Wisconsin' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --alpha 0.0 --device 'cuda:0' | tee Wisconsin.txt


# --------------------------------------------------------------------------------------------------------------------------

# python -u large_scale_graphs_main.py --dataset 'arxiv-year' --train_lr 0.01  --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th 0.0 --num_parts 100 --num_splits 5 --alpha 0.0 --device 'cuda:0' | tee arxiv_year.txt


# python -u large_scale_graphs_main.py --dataset 'snap-patents' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 30 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_parts 1000 --num_splits 1 --alpha 0.0 --device 'cuda:0' | tee snap_patents.txt


# python -u large_scale_graphs_main.py --dataset 'Penn94' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_parts 100 --num_splits 5 --alpha 1.0 --device 'cuda:0' | tee Penn94.txt


# python -u large_scale_graphs_main.py --dataset 'pokec' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_parts 2000 --num_splits 5 --alpha 1.0 --device 'cuda:0' | tee pokec.txt

# python -u large_scale_graphs_main.py --dataset 'twitch-gamers' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_parts 100 --num_splits 5 --alpha 5.0 --device 'cuda:0' | tee twitch_gamers.txt


# python -u large_scale_graphs_main.py --dataset 'genius' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_parts 100 --num_splits 5 --alpha 0.0 --device 'cuda:0' | tee genius.txt

# ---------------------------------------------------------

# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0005 --th None --num_splits 10 --rewiring True --model gat-sep --alpha 5.0 --device 'cuda:0'


# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 25 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0005 --th None --num_splits 10 --rewiring False --model gat --alpha 20.0 --device 'cuda:0'


# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0005 --th None --num_splits 1 --rewiring False --model gcn --alpha 5.0 --device 'cuda:0'


# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0005 --th None --num_splits 1 --rewiring False model gcn --alpha 5.0 --device 'cuda:0'


# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 3 --mlp_layers 1 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0005 --th 0.0 --num_splits 1 --rewiring False --model gcn  --alpha 5.0 --device 'cuda:0'


# -----------------------------------------------------------------

# TUDatasets --- Molecular graphs

python -u tu_datasets_main.py --dataset 'ENZYMES' --train_lr 0.001 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_splits 1 --batch_size 8 --rewiring simgrew --model gcn --alpha 0.0 --device 'cuda:0' | tee output/enzymes.txt

# python -u tu_datasets_main.py --dataset 'MUTAG' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_splits 25 --batch_size 8 --rewiring simgrew --model gcn --alpha 2.0 --device 'cuda:0' | tee output/mutag.txt

# python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th 0.0 --num_splits 25 --batch_size 8 --rewiring simgrew --model gcn --alpha 2.0 --device 'cuda:0' | tee output/proteins.txt

# python -u tu_datasets_main.py --dataset 'COLLAB' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_splits 25 --batch_size 8 --rewiring simgrew --model gcn --alpha 1.0 --device 'cuda:0' | tee output/collab.txt


# python -u tu_datasets_main.py --dataset 'REDDIT-BINARY' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th 0.0 --num_splits 25 --batch_size 1 --rewiring simgrew --model gcn --alpha 2.0 --device 'cuda:0' | tee output/reddit-binary.txt


# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th 0.0 --num_splits 25 --batch_size 16 --rewiring simgrew --model gcn --alpha 2.0 --device 'cuda:0' | tee output/imdb-binary.txt

# --------------------------

#  For GIN

# python -u tu_datasets_main.py --dataset 'ENZYMES' --train_lr 0.001 --seed 0 --num_layers 3 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_splits 25 --batch_size 8 --rewiring gtr --model gin --alpha 0.0 --device 'cuda:0' | tee output/enzymes.txt

# python -u tu_datasets_main.py --dataset 'MUTAG' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th 0.0 --num_splits 25 --batch_size 8 --rewiring gtr --model gin --alpha 0.0 --device 'cuda:0' | tee output/mutag.txt

# python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th 0.0 --num_splits 25 --batch_size 8 --rewiring gtr --model gin --alpha 0.0 --device 'cuda:0' | tee output/proteins.txt

# python -u tu_datasets_main.py --dataset 'COLLAB' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_splits 25 --batch_size 8 --rewiring gtr --model gin --alpha 0.0 --device 'cuda:0' | tee output/collab.txt

# python -u tu_datasets_main.py --dataset 'REDDIT-BINARY' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_splits 25 --batch_size 1 --rewiring gtr --model gin --alpha 0.0 --device 'cuda:0' | tee output/reddit-binary.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_splits 25 --batch_size 16 --rewiring simgrew --model gin --alpha 0.0 --device 'cuda:0' | tee output/imdb-binary.txt



# -----------------------------------------------------------

# LRGB

# python -u lrgb_main.py --dataset 'PASCALVOC' --train_lr 0.0005 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.1 --th None --batch_size 32 --rewiring None --model gcn --alpha 0.0 --device 'cuda:0' | tee pascal-voc-sp.txt

# python -u lrgb_main.py --dataset 'COCO' --train_lr 0.0005 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.1 --th None --batch_size 32 --rewiring None --model gcn --alpha 0.0 --device 'cuda:0' | tee coco-sp.txt

