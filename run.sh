python cache_activations.py --cache_n_examples 5000

python sae_reconstruction.py --num_samples 5000 --batch_size 10 --token_types llm_text vlm_img vlm_imgtext

python sae_descriptions_eval.py --n_features_per_layer 3 --num_samples 1000 --num_freq_features 5000 --frequency_threshold 0.05 --batch_size 10 --token_types vlm_img llm_text