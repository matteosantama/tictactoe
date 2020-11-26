from src import TicTacToe, RLPlayer, UserPlayer


if __name__ == '__main__':
    rl = RLPlayer('rl')
    trainer = RLPlayer('trainer', epsilon=0.4)
    user = UserPlayer('user')
    
    tictactoe = TicTacToe()
    tictactoe.register(trainer)
    tictactoe.register(rl)
    
    print('Welcome to TicTacToe!')
    print('Please hold... training your opponent')
    n = 60000
    for i in range(n):
        if (i + 1) % 5000 == 0:
            print(f'{(i + 1) / n * 100:3.1f}% trained')
        _ = tictactoe.play()
        tictactoe.reset()
        
    rl.set_epsilon(0.01)
    print('Ready to play!')
    tictactoe = TicTacToe()
    tictactoe.register(user)
    tictactoe.register(rl)

    while True:
        winner = tictactoe.play()
        if winner == user:
            print('Congrats! You won!\n')
        elif winner is None:
            print('Tie.\n')
        else:
            print('Better luck next time...\n')
        tictactoe.reset()