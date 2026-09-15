# world-model-cache

## The problem
World models predict the next frame of an environment given past frames and actions. Transformer world models only see the last K steps. Walk away from a room for longer than K, come back: the model invents a room instead of remembering it (see "dementia Minecraft" / Oasis).

## The idea
Give the model an external memory: a small bank of snapshots (where I was, what I saw). Before each prediction, fetch the snapshots taken nearest to the current position and let the model look at them. The bank has a fixed size C, so every new snapshot forces us to DELETE an old one. Which one to delete is the whole project. That is a cache eviction problem, and cache theory has an answer: Belady's rule (evict what is needed furthest in the future) is optimal if you know the future. On recorded trajectories, we DO know the future. So we can compute the optimal policy, compare simple rules against it, and train a small network to imitate it.

## The system (3 parts)
1. Tokenizer (VQ-VAE): compresses each 64x64 frame into 16 discrete tokens. A snapshot = position + 16 tokens (~18 bytes).
2. World model (small GPT-style transformer): reads [retrieved snapshots + last K steps of actions and frame tokens], predicts the next frame's 16 tokens one by one.
3. The cache (our contribution): the bank, the retrieval rule, and 5 eviction policies: FIFO, LRU, keep-spatial-coverage, learned (trained on Belady labels), and the Belady oracle itself as upper bound.

Data: Memory Maze offline dataset (30k recorded maze walks with images, actions, and true positions). Nothing to simulate; positions make retrieval and the oracle computable.
Paper: "Evaluating Long-Term Memory in 3D Mazes" (Pasukonis, Lillicrap, Hafner 2022), https://arxiv.org/abs/2210.13383. Code + datasets: https://github.com/jurgisp/memory-maze

## The result (2 artifacts)
1. One plot: prediction correctness at revisited places (y) vs memory size C (x), five curves: no memory, FIFO, LRU, learned, oracle. Claim: the learned policy closes most of the gap between LRU and optimal.
2. One 30s clip: same walk, same memory size, toggle LRU vs learned: the revisited room hallucinates vs survives.

## The plan
1. Data: load dataset, find all revisit events (same place, N steps apart).
2. Tokenizer: train VQ-VAE until rooms are recognizable in reconstructions.
3. World model: train transformer; show the failure: accuracy at revisits collapses when N > K.
4. Cache: add bank + retrieval; fine-tune so the model uses it (verify: mask the bank, accuracy must crash); implement Belady on recordings; implement the 4 policies; make the plot.
5. Compare + demo + short paper.

## Why it matters
Long-term consistency is the named open problem of interactive world models. Everyone adds memory; nobody has measured their heuristics against the optimum or learned the deletion policy
