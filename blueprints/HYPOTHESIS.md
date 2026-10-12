# does a world model know what it remembers?
Metamemory is knowing what your own memory holds.
We trained a world model to use its memory, never to judge it: nothing told it when a place is known and when it is new.

So we look inside. When the model enters a place it has in memory and a place it does not, do its activations differ?
If they do, the information is there, without ever being taught.

why it matters: on places it does not have, our model predicts worse with memory than without (evaluate_cache.ipynb section 3).

## three kinds of place

- known: the bank holds this view (within 0.5, facing within ~45 degrees)
- unknown: it does not
- fake known: the bank holds an entry at this pose, but with the frame of another place

## hypotheses

H1: known and unknown look different inside. a linear probe on the activations separates them.
- pass: AUROC 0.8
- expected to be easy: the model sees the offset of every entry, so it can read "known" off its input. a predictor that only sees pose will do as well. H1 is the sanity check, not the result.

H2: known and fake known look different inside. same pose, wrong content.
- pass: AUROC 0.7
- a predictor that only sees pose is at 0.5 here by construction
- yes: the model judges what its memory holds, not only where it is. this is the result.
- no: the model only checks the address. also a result.

## signal

taken during the read, at the tokens of the frame being predicted:
- read stats per block: highest attention on one entry, entropy over entries, norm of what the read adds (4 blocks x 3 = 12 numbers)
- the residual stream after each block (4 x 256)

## plan

1. get the signal out, in `src/models/world_model.py`:
   - `forward(..., return_stats=False)`: when on, also return the read stats and the residual per block
   - compute the attention by hand when on (the torch function does not return weights)
   - default off, so training and the old notebooks do not change

2. build the three sets, new `notebooks/evaluate_metamemory.ipynb`:
   - known / unknown: revisit events from the eval episodes, split by the covered rule
   - fake known: take a known event, keep the pose of the entry that covers it, swap its 16 tokens with those of an entry far away in the same bank
   - same episodes, same steps for known and fake known, only that one entry changes

3. probes, same notebook:
   - logistic regression, fit on half of the eval episodes, test on the other half
   - H1: known vs unknown, for read stats, residual, pose only
   - H2: known vs fake known, for read stats, residual, pose only (should be 0.5)
   - AUROC per block, 3 seeds, bootstrap error bars

4. one figure: AUROC per block for H1 and H2, pose only as the dashed line. then write.

no training needed.

optional after: does the signal say when the read hurts (loss with cache vs without), and does skipping the read then help.

## not claiming

- a gate, or that memory hurts in new places (arXiv 2604.18215 has both)
- that the model uses the signal, only that it is there
- anything about awareness

## open

- 0.8 and 0.7 are my guesses, fix them before running
- how would the model notice wrong content: the frames just before the revisit show nearly the same view, so the window and the entry disagree. check that the events have this overlap.
- prior work only read through summaries, check the pdfs
