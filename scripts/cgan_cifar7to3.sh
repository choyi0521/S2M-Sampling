python src/main.py --config_path ./configs/biggan_cifar7to3_pd.yaml --checkpoint_dir ./checkpoints/cGAN_cifar7to3 --session_name cgan_cifar7to3_0-1-2 --intersection_indices 0 --difference_indices 1 2 --temperature_logits 1.2 --gammas 0.5 1.0 1.0

python src/main.py --config_path ./configs/biggan_cifar7to3_pd.yaml --checkpoint_dir ./checkpoints/cGAN_cifar7to3 --session_name cgan_cifar7to3_0+1-2 --intersection_indices 0 1 --difference_indices 2 --temperature_logits 1.0 --gammas 0.5 0.5 1.0

python src/main.py --config_path ./configs/biggan_cifar7to3_pd.yaml --checkpoint_dir ./checkpoints/cGAN_cifar7to3 --session_name cgan_cifar7to3_1-0-2 --intersection_indices 1 --difference_indices 0 2 --temperature_logits 1.2 --gammas 1.0 0.5 1.0

python src/main.py --config_path ./configs/biggan_cifar7to3_pd.yaml --checkpoint_dir ./checkpoints/cGAN_cifar7to3 --session_name cgan_cifar7to3_1+2-0 --intersection_indices 1 2 --difference_indices 0 --temperature_logits 1.0 --gammas 1.0 0.5 0.5

python src/main.py --config_path ./configs/biggan_cifar7to3_pd.yaml --checkpoint_dir ./checkpoints/cGAN_cifar7to3 --session_name cgan_cifar7to3_2-0-1 --intersection_indices 2 --difference_indices 0 1 --temperature_logits 1.2 --gammas 1.0 1.0 0.5

python src/main.py --config_path ./configs/biggan_cifar7to3_pd.yaml --checkpoint_dir ./checkpoints/cGAN_cifar7to3 --session_name cgan_cifar7to3_0+2-1 --intersection_indices 0 2 --difference_indices 1 --temperature_logits 1.0 --gammas 0.5 1.0 0.5

python src/main.py --config_path ./configs/biggan_cifar7to3_pd.yaml --checkpoint_dir ./checkpoints/cGAN_cifar7to3 --session_name cgan_cifar7to3_0+1+2 --intersection_indices 0 1 2 --temperature_logits 0.2 --gammas 0.5 0.5 0.5
