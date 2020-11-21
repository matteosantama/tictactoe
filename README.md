# TicTacToe
Custom built framework for training an RL-enabled Tic Tac Toe player. There are
three types of players currently implemented: `RandomPlayer` that chooses a random
move, `UserPlayer` that makes a user-provided move, and an `RLPlayer` that requires
pre-training before it is able to make intelligent moves.

Currently only Tic Tac Toe is supported, but the `Game` class was built generic enough to 
allow implementation of additional games. Similarly, player types can be added by 
simply sub-classing `Player` and implementing `move()`. 