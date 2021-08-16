# Bolzmann
 
This is my own implementation to the network described in https://github.com/EverettYou/EFL 
and the corresponding article (https://arxiv.org/abs/1709.01223). The calculation parts of the Code are taken from the original repository
and simply 'translated' to `pytorch`.

Instead of `tensorflow`, which is used in the original implementation, in this one `pytorch` is used.

## Limitations
I have not implemented the full network functionality, but the parts I needed and found the most interesting.
Therefore, this implementation misses:
- The regularization term (It worked without as well for me).
- All sample methods besides `weighted` since the article states 'We use
the interval-weighted scheme to sample the entanglement
regions and prepare the training data for this study.' (See 'B. Choosing the Central Charge' at 
  https://arxiv.org/abs/1709.01223)
