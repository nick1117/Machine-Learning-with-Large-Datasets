#!/bin/bash

sample_alpha() {
  python3 -c "import numpy as np; print(np.exp(np.random.uniform(np.log(1e-4), np.log(1e-1))))"
}

# TINIEST
# Trial 1
alpha=$(sample_alpha)
./train_gpt2cu \-o "question_2_tiniest_1" \-1d 1 \-1c 64 \-1h 1 \-l "$alpha" \-u 70 \-x 1000 \-b 128
python upload_to_s3.py "10605nchermakhw5" "question_2_tiniest_1" "question_2_strat3"

# Trial 2
alpha=$(sample_alpha)
./train_gpt2cu \-o "question_2_tiniest_2" \-1d 1 \-1c 64 \-1h 1 \-l "$alpha" \-u 70 \-x 1000 \-b 128
python upload_to_s3.py "10605nchermakhw5" "question_2_tiniest_2" "question_2_strat3"

# Trial 3
alpha=$(sample_alpha)
./train_gpt2cu \-o "question_2_tiniest_3" \-1d 1 \-1c 64 \-1h 1 \-l "$alpha" \-u 70 \-x 1000 \-b 128
python upload_to_s3.py "10605nchermakhw5" "question_2_tiniest_3" "question_2_strat3"

# Trial 4
alpha=$(sample_alpha)
./train_gpt2cu \-o "question_2_tiniest_4" \-1d 1 \-1c 64 \-1h 1 \-l "$alpha" \-u 70 \-x 1000 \-b 128
python upload_to_s3.py "10605nchermakhw5" "question_2_tiniest_4" "question_2_strat3"

# Trial 5
alpha=$(sample_alpha)
./train_gpt2cu \-o "question_2_tiniest_5" \-1d 1 \-1c 64 \-1h 1 \-l "$alpha" \-u 70 \-x 1000 \-b 128
python upload_to_s3.py "10605nchermakhw5" "question_2_tiniest_5" "question_2_strat3"

# Trial 6
alpha=$(sample_alpha)
./train_gpt2cu \-o "question_2_tiniest_6" \-1d 1 \-1c 64 \-1h 1 \-l "$alpha" \-u 70 \-x 1000 \-b 128
python upload_to_s3.py "10605nchermakhw5" "question_2_tiniest_6" "question_2_strat3"

# Trial 7
alpha=$(sample_alpha)
./train_gpt2cu \-o "question_2_tiniest_7" \-1d 1 \-1c 64 \-1h 1 \-l "$alpha" \-u 70 \-x 1000 \-b 128
python upload_to_s3.py "10605nchermakhw5" "question_2_tiniest_7" "question_2_strat3"

# Trial 8
alpha=$(sample_alpha)
./train_gpt2cu \-o "question_2_tiniest_8" \-1d 1 \-1c 64 \-1h 1 \-l "$alpha" \-u 70 \-x 1000 \-b 128
python upload_to_s3.py "10605nchermakhw5" "question_2_tiniest_8" "question_2_strat3"

# Trial 9
alpha=$(sample_alpha)
./train_gpt2cu \-o "question_2_tiniest_9" \-1d 1 \-1c 64 \-1h 1 \-l "$alpha" \-u 70 \-x 1000 \-b 128
python upload_to_s3.py "10605nchermakhw5" "question_2_tiniest_9" "question_2_strat3"

# Trial 10
alpha=$(sample_alpha)
./train_gpt2cu \-o "question_2_tiniest_10" \-1d 1 \-1c 64 \-1h 1 \-l "$alpha" \-u 70 \-x 1000 \-b 128
python upload_to_s3.py "10605nchermakhw5" "question_2_tiniest_10" "question_2_strat3"
