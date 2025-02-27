# Vector Sum
The task is you have two 2d tensor of shape (size, size) which is float16 datatype.
You have to do matrix addition using cuda kernel and send the result back. This
is based on gpu mode discord server leaderboard. 

## Kernel Details
It is simple one and self evident.

## Code
There are two files:
- `submission.py`: The file used to submit for the leaderboard. You can submit using:
```bash
/leaderboard submit benchmark {submit the submission.y in ui} leaderboard_name:vectoradd gpu:T4
```
- `vector_add.ipynb`: You can open this file with T4 gpu runtime in colab and run it.
## Results on Leader board
### T4
```
seed: 31232; size: 1024
 ⏱ 64.5 ± 0.79 µs
 ⚡ 62.2 µs 🐌 141 µs

seed: 4052; size: 2048
 ⏱ 192 ± 1.7 µs
 ⚡ 189 µs 🐌 204 µs

seed: 2146; size: 4096
 ⏱ 683 ± 4.7 µs
 ⚡ 678 µs 🐌 692 µs

seed: 3129; size: 8192
 ⏱ 2.66 ± 0.006 ms
 ⚡ 2.65 ms 🐌 2.67 ms

seed: 54352; size: 16384
 ⏱ 10.5 ± 0.02 ms
 ⚡ 10.5 ms 🐌 10.6 ms

Leaderboard vectoradd:
Gokul's submission.py on T4 ran for 0.009162481 seconds!
```
### A100
```
seed: 31232; size: 1024
 ⏱ 43.2 ± 1.13 µs
 ⚡ 39.4 µs 🐌 150 µs

seed: 4052; size: 2048
 ⏱ 56.1 ± 0.56 µs
 ⚡ 53.9 µs 🐌 76.7 µs

seed: 2146; size: 4096
 ⏱ 159 ± 1.5 µs
 ⚡ 156 µs 🐌 169 µs

seed: 3129; size: 8192
 ⏱ 540 ± 4.3 µs
 ⚡ 535 µs 🐌 548 µs

seed: 54352; size: 16384
 ⏱ 2.13 ± 0.005 ms
 ⚡ 2.13 ms 🐌 2.14 ms

Leaderboard vectoradd:
Gokul's submission.py on A100 ran for 0.002118579 seconds!
```

