import abc
from collections import defaultdict
from dataclasses import dataclass, field
import random
from typing import (
    Tuple, Dict, List, DefaultDict, Set, 
    Optional, Iterable
)

import numpy as np


@dataclass
class Player(abc.ABC):
    
    # To differentiate players with the same base strategy
    name: str
    token: Optional[int] = field(default=None, hash=False)

    @abc.abstractmethod
    def move(self, board: np.array) -> Tuple[int, int]:
        raise NotImplementedError
        
    def learn(self, won: bool) -> None:
        """
        Hook to do learning if the player is intelligent.
        For most players, we do nothing and just pass.
        """
        pass
        
    def __hash__(self) -> int:
        """Hash based on class and name attribute."""
        return hash(str(self.__class__) + self.name)


class RandomPlayer(Player):
    
    def move(self, board: np.array) -> Tuple[int, int]:
        """Randomly select an open position on the board."""
        open_spots = np.argwhere(board == 0)
        choice = np.random.choice(len(open_spots))
        return tuple(open_spots[choice])
    
    
class UserPlayer(Player):
    
    def move(self, board: np.array) -> Tuple[int, int]:
        """User-defined move."""
        print('Current Board:')
        print(board)
        while True:
            try:
                move = input('Enter \'row,col\' (0-based):')
                a, b = move.split(',')
                move = (int(a), int(b))
                if board[move] != 0:
                    raise ValueError
                return move
            except ValueError:
                print('Uh oh, invalid move!\n')
                pass

    
State = Tuple[int, ...]

@dataclass(unsafe_hash=True)
class RLPlayer(Player):
    """
    Losses and ties are equally dis-incentivized in the agent.
    """
    
    training: bool = field(hash=False, default=True)
    epsilon: float = 0.01
    gamma: float = 0.2
    
    values: DefaultDict[State, float] = field(
        default_factory=lambda: defaultdict(float), init=False, 
        repr=False, hash=False)
        
    history: List[State] = field(
        default_factory=list, init=False, repr=False, hash=False)
        
    def reachable_states(self, board: np.array) -> Iterable[State]:
        flattened = board.flatten()
        for i in range(len(flattened)):
            if flattened[i] == 0:
                state = flattened.copy()
                state[i] = self.token
                yield tuple(state)
    
    def move(self, board: np.array) -> Tuple[int, int]:
        reachables = {}
        for state in self.reachable_states(board):
            reachables[state] = self.values[state]
        
        if self.training and np.random.rand() < self.epsilon:
            choices = list(reachables)
        else:
            greedy = max(reachables.values())
            choices = [x for x in reachables if reachables[x] == greedy]
            
        next_state = random.choice(choices)
        np_next_state = np.array(next_state).reshape(board.shape)
        _move = tuple(np.argwhere(np_next_state != board).flatten())
                
        self.history.append(next_state)
        return _move
    
    def learn(self, won: bool) -> None:
        rewards = {True: 1., False: -1., None: 0.}
        for i, state in enumerate(reversed(self.history)):
            new_reward = rewards[won] * np.power(self.gamma, i)
            self.values[state] = 0.1 * new_reward + 0.9 * self.values[state]
        self.history = []