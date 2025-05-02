# ./train_gpt2cu \-o "question_1_experiment_2" \-1d 1 \-1c 64 \-1h 1 \-l 0.001 \-u 70 \-x 1000 -b 128
# python upload_to_s3.py "10605nchermakhw5" "question_1_experiment_2" "question_1_experiments"

# sudo shutdown -h now

#!/bin/bash

# Function to generate a log-uniform random number between 1e-5 and 1e-2
sample_alpha() {
python3 -c "import numpy as np; print(np.exp(np.random.uniform(np.log(1e-4), np.log(1e-1))))"
}

# tiniest
for i in {1..10}
do
  alpha=$(sample_alpha)
  alpha_tag=$(printf "%.2e" "$alpha")

  exp_name="question_2_tiniest_${i}_lr${alpha_tag}"

  echo "Starting tiniest training run $i with learning rate $alpha"

  ./train_gpt2cu \-o "$exp_name" \-1d 1 \-1c 64 \-1h 1 \-l "$alpha" \-u 70 \-x 1000 \-b 128

  python upload_to_s3.py "10605nchermakhw5" "$exp_name" "question_2_strat3"
done

# tinier
for i in {1..10}
do
  alpha=$(sample_alpha)
  alpha_tag=$(printf "%.2e" "$alpha")

  exp_name="question_2_tinier_${i}_lr${alpha_tag}"

  echo "Starting tinier training run $i with learning rate $alpha"

  ./train_gpt2cu \-o "$exp_name" \-1d 2 \-1c 128 \-1h 2 \-l "$alpha" \-u 70 \-x 1000 \-b 128

  python upload_to_s3.py "10605nchermakhw5" "$exp_name" "question_2_strat3"
done

# tiny
for i in {1..10}
do
  alpha=$(sample_alpha)
  alpha_tag=$(printf "%.2e" "$alpha")

  exp_name="question_2_tiny_${i}_lr${alpha_tag}"

  echo "Starting tiny training run $i with learning rate $alpha"

  ./train_gpt2cu \-o "$exp_name" \-1d 3 \-1c 192 \-1h 3 \-l "$alpha" \-u 70 \-x 1000 \-b 128

  python upload_to_s3.py "10605nchermakhw5" "$exp_name" "question_2_strat3"
done

# Shutdown machine after all runs complete
sudo shutdown -h now
