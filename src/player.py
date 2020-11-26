import abc
from dataclasses import dataclass, field
import random
from typing import (
    Tuple, Dict, List, Set, 
    Optional, Iterable
)

import numpy as np


@dataclass
class Player(abc.ABC):
    
    # To differentiate players with the same base strategy
    name: str
    token: Optional[int] = field(default=None, hash=False, repr=False)

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
    Losses and ties are equally disincentivized in the agent.
    """
    
    epsilon: float = 0.1
    alpha: float = 0.9
    
    values: Dict[State, float] = field(
        default_factory=dict, init=False, repr=False, hash=False)
        
    history: List[State] = field(
        default_factory=list, init=False, repr=False, hash=False)

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = epsilon
        
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
            if state not in self.values:
                self.values[state] = 0.5
            reachables[state] = self.values[state]
        
        if np.random.rand() < self.epsilon:
            choices = list(reachables)
        else:
            greedy = max(reachables.values())
            choices = [x for x in reachables if reachables[x] == greedy]
            
        next_state = random.choice(choices)
        np_next_state = np.array(next_state).reshape(board.shape)
        move_indices = tuple(np.argwhere(np_next_state != board).flatten())
                
        self.history.append(next_state)
        return move_indices
    
    def learn(self, won: bool) -> None:
        """A 'won' value of 'None' indicates the game was a tie."""
        reward = 1. if won is not False else 0.
        self.values[self.history[-1]] = reward
        for i, state in enumerate(reversed(self.history)):
            discount = np.power(self.alpha, i)
            self.values[state] += discount * (reward - self.values[state])
        self.history = []