import abc
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

from .player import Player
from .utils import learnable


@dataclass
class Game(abc.ABC):

    board:          np.array
    max_players:    int
    players:        List[Player]       = field(default_factory=list)
    tokens:         Dict[Player, int]  = field(default_factory=dict)
    up:             int                = 0
    verbose:        bool               = False

    def register(self, player: Player) -> None:
        """Register another player for the game"""
        if player in self.tokens:
            raise RuntimeError(f'{player} already in game')
        if len(self.players) + 1 > self.max_players:
            raise RuntimeError(f'Too many players trying to enter')
        self.players.append(player)
        self.tokens[player] = len(self.players)
        player.token = self.tokens[player]

    @abc.abstractmethod
    def was_winning_move(self, player: Player) -> bool:
        raise NotImplemented
        
    def ended_in_draw(self) -> bool:
        return np.all(self.board != 0)
    
    def reset(self) -> None:
        if self.verbose:
            print('Resetting board...')
        self.board = np.zeros_like(self.board)
        self.up = 0

    @learnable
    def play(self) -> Player:
        """
        The players alternate making moves until either someone wins,
        or a tie is determined.
        """
        if np.count_nonzero(self.board) != 0:
            raise RuntimeError('Game was not reset') 
        done = False
        while not done:
            player_to_go = self.players[self.up]
            move = player_to_go.move(self.board)
            self.board[move] = self.tokens[player_to_go]
            if self.verbose:
                print(f'{player_to_go} moved:\n{self.board}')
            if self.was_winning_move(player_to_go):
                if self.verbose:
                    print(f'{player_to_go} wins!\n')
                return player_to_go
            done = self.ended_in_draw()
            self.up = (self.up + 1) % len(self.players)
        if self.verbose:
            print('Game ended in a tie!\n')
        return None

    
class TicTacToe(Game):

    def __init__(self, verbose=False) -> None:
        board = np.zeros(shape=(3,3))
        super().__init__(board, 2, verbose=verbose)

    def was_winning_move(self, player: Player) -> bool:
        token = self.tokens[player]
        rows = np.all(self.board == token, axis=1)
        cols = np.all(self.board == token, axis=0)
        ldiag = np.all(self.board.diagonal() == token)
        rdiag = np.all(np.fliplr(self.board).diagonal() == token)
        return rows.any() or cols.any() or ldiag or rdiag
