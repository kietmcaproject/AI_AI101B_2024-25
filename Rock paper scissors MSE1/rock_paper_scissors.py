import random

def get_computer_choice():
    """Generates a random choice for the computer."""
    choices = ["rock", "paper", "scissors"]
    return random.choice(choices)

def get_player_choice():
    """Gets the player's choice."""
    while True:
        player_choice = input("Enter your choice (rock, paper, or scissors): ").lower()
        if player_choice in ["rock", "paper", "scissors"]:
            return player_choice
        else:
            print("Invalid choice. Please try again.")

def determine_winner(player_choice, computer_choice):
    """Determines the winner of the game."""
    print(f"You chose {player_choice}, computer chose {computer_choice}.")

    if player_choice == computer_choice:
        return "It's a tie!"
    elif (player_choice == "rock" and computer_choice == "scissors") or \
         (player_choice == "paper" and computer_choice == "rock") or \
         (player_choice == "scissors" and computer_choice == "paper"):
        return "You win!"
    else:
        return "Computer wins!"

def play_game():
    """Plays a single round of Rock-Paper-Scissors."""
    player_choice = get_player_choice()
    computer_choice = get_computer_choice()
    result = determine_winner(player_choice, computer_choice)
    print(result)

# Start the game
play_game()
