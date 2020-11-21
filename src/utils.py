import functools


def learnable(play):
    """
    Decorator function that is specifically designed to wrap
    the game.play() method. This allows us to do any player
    training before we fully exit the function.
    """
    @functools.wraps(play)
    def decorated(_self):
        winner = play(_self)
        for p in _self.players:
            p.learn(winner == p if winner else None)
        return winner
    return decorated
