
import math
import random

# Function to print the board
def print_board(board):
    for row in board:
        print(" | ".join(row))
    print("\n")

# Check for available moves
def is_moves_left(board):
    for row in board:
        if " " in row:
            return True
    return False

# Evaluate the board to check for a winner
def evaluate(board):
    # Check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != ' ':
            return 10 if board[i][0] == 'X' else -10
        if board[0][i] == board[1][i] == board[2][i] != ' ':
            return 10 if board[0][i] == 'X' else -10
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != ' ':
        return 10 if board[0][0] == 'X' else -10
    if board[0][2] == board[1][1] == board[2][0] != ' ':
        return 10 if board[0][2] == 'X' else -10
    return 0

# Minimax function to determine the best move
def minimax(board, depth, is_max):
    score = evaluate(board)
    if score == 10:
        return score - depth  # Favor faster wins
    if score == -10:
        return score + depth  # Favor slower losses
    if not is_moves_left(board):
        return 0

    if is_max:
        best = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    best = max(best, minimax(board, depth + 1, False))
                    board[i][j] = ' '
        return best
    else:
        best = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    best = min(best, minimax(board, depth + 1, True))
                    board[i][j] = ' '
        return best

# Find the best move for AI
def find_best_move(board):
    best_val = -math.inf
    best_move = (-1, -1)
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'X'
                move_val = minimax(board, 0, False)
                board[i][j] = ' '
                if move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val
    return best_move

# Main function to play the game
def play_game():
    print("Select Mode:")
    print("1. Play with AI")
    print("2. Play with a Friend")
    mode = input("Enter your choice (1 or 2): ")
    if mode == '1':
        player1 = input("Enter your name: ")
        player2 = "AI"
        players = [(player1, 'O'), (player2, 'X')]
    elif mode == '2':
        player1 = input("Enter Player 1 name: ")
        player2 = input("Enter Player 2 name: ")
        players = [(player1, 'X'), (player2, 'O')]
        random.shuffle(players)
    else:
        print("Invalid choice! Exiting...")
        return

    print(f"{players[0][0]} is assigned '{players[0][1]}' and {players[1][0]} is assigned '{players[1][1]}'")
    board = [[' ' for _ in range(3)] for _ in range(3)]
    current_player = 0

    while True:
        print_board(board)
        if not is_moves_left(board) or evaluate(board) != 0:
            break
        if players[current_player][0] == "AI":
            ai_move = find_best_move(board)
            board[ai_move[0]][ai_move[1]] = 'X'
        else:
            try:
                row, col = map(int, input(f"{players[current_player][0]}, enter row and column (0-2) separated by space: ").split())
                if row not in range(3) or col not in range(3):
                    print("Invalid input! Please enter values between 0 and 2.")
                    continue
                if board[row][col] == ' ':
                    board[row][col] = players[current_player][1]
                else:
                    print("Invalid move! Try again.")
                    continue
            except ValueError:
                print("Invalid input! Please enter numeric values between 0 and 2.")
                continue

        if not is_moves_left(board) or evaluate(board) != 0:
            break
        current_player = 1 - current_player  # Switch turns

    print_board(board)
    result = evaluate(board)
    if result == 10:
        print(f"{player2 if player2 == 'AI' else player1} wins!")
    elif result == -10:
        print(f"{player1 if player1 != 'AI' else player2} wins!")
    else:
        print("It's a draw!")

# Run the game
play_game()
