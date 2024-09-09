# Harrys Chess Bot
The idea behind this bot is what if we could train a neural network to accurately predict a top level engine's evaluation given a chess position.
From there we can build a bot by giving a collection of all possible moves to the model and playing the one with the best evaluation.
This is rather simple as we can use tools like python-chess to grab all positions and then convert each position into a feature vector for the neural network.

It goes without saying that a neural network created in this fashion does not understand chess. If anything, it understands top chess engines like Stockfish - or can at least approximate their evaluation for a given position.

Future plans for this project are to make an api that can take in a fen position and use the neural network to output the best move. This can then be expanded into tracking the game as its being played so you could actually play against it.

I think a pretty achievable goal is just to have it be able to outplay me - someone who peaked in the 1300s on chesscom. Perhaps 'pretty achievable' is an understatement...

## Current Training Progress
```
$ python train.py 
Input features shape: (7842, 64)
Target values shape: (7842, 1)
Best parameters found:  {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': (128, 128, 128, 128, 128), 'learning_rate': 'constant', 'max_iter': 200, 'solver': 'adam'}
Best score found:  0.000778258466979856
```

## Project Information
python version 3.12

The dataset for the training can be found at https://database.lichess.org/lichess_db_eval.jsonl.zst