from src import TicTacToe, RLPlayer, RandomPlayer, UserPlayer

if __name__ == '__main__':
    rl = RLPlayer('rl')
    random = RandomPlayer('random')
    user = UserPlayer('user')
    
    tictactoe = TicTacToe()
    tictactoe.register(random)
    tictactoe.register(rl)
    
    n = input(f'Define number of training iterations: ')
    for i in range(int(n)):
        if (i + 1) % 500 == 0:
            print(f'{i + 1}/{n} iterations completed')
        _ = tictactoe.play()
        tictactoe.reset()
        
        
    print('Ready to play!')
    tictactoe = TicTacToe()
    tictactoe.register(user)
    tictactoe.register(rl)
    rl.training = False

    while True:
        winner = tictactoe.play()
        if winner == user:
            print('Congrats! You won!\n')
        elif winner is None:
            print('Tie.\n')
        else:
            print('Better luck next time...\n')
        tictactoe.reset()