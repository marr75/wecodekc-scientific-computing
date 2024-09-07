from typing_extensions import Literal, Annotated, Optional

import utils

BoardType = list[list[str]]
Sides = Literal["white", "black"]
Pieces = Literal["king", "queen", "rook", "bishop", "knight", "pawn"]
pieces: dict[Sides, dict[Pieces, str]] = {
    "black": {
        "king": "♔",
        "queen": "♕",
        "rook": "♖",
        "bishop": "♗",
        "knight": "♘",
        "pawn": "♙",
    },
    "white": {
        "king": "♚",
        "queen": "♛",
        "rook": "♜",
        "bishop": "♝",
        "knight": "♞",
        "pawn": "♟",
    },
}


def print_board(board):
    # Print the column letters
    print("  a b c d e f g h")
    for i in range(8):
        # Print the row numbers
        print(str(8 - i), end=" ")
        for j in range(8):
            # Print the piece or an empty square
            piece = board[i][j]
            print(piece if piece else ".", end=" ")
        print(8 - i)
    print("  a b c d e f g h")


# Example of a starting chess board
board: BoardType = [
    ["♖", "♘", "♗", "♕", "♔", "♗", "♘", "♖"],
    ["♙", "♙", "♙", "♙", "♙", "♙", "♙", "♙"],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    ["♟", "♟", "♟", "♟", "♟", "♟", "♟", "♟"],
    ["♜", "♞", "♝", "♛", "♚", "♝", "♞", "♜"],
]
print_board(board)


def peek_board():
    """
    Display the current state of the board.
    """
    print_board(board)
    formatted_board = "\n".join(" ".join(row) for row in board)
    return formatted_board


def find_piece(side: Sides, piece: Pieces):
    """
    Find the locations of a piece on the board.
    """
    print("find_piece")
    locations = []
    for i in range(8):
        for j in range(8):
            if board[i][j] == pieces[side][piece]:
                locations.append((i, j))
    return locations


def is_valid_move(piece: Pieces, start_i: int, start_j: int, end_i: int, end_j: int):
    """
    Check if a move is valid.
    """
    if piece == "king":
        return abs(start_i - end_i) <= 1 and abs(start_j - end_j) <= 1
    if piece == "rook":
        is_vertical_move = start_i == end_i and start_j != end_j
        is_horizontal_move = start_i != end_i and start_j == end_j
        return is_vertical_move or is_horizontal_move
    if piece == "bishop":
        # diagonal moves
        return abs(start_i - end_i) == abs(start_j - end_j)
    elif piece == "queen":
        # Queen can move like a rook or a bishop
        is_vertical_move = start_i == end_i and start_j != end_j
        is_horizontal_move = start_i != end_i and start_j == end_j
        is_diagonal_move = abs(start_i - end_i) == abs(start_j - end_j)
        return is_vertical_move or is_horizontal_move or is_diagonal_move
    elif piece == "knight":
        # L-shaped moves
        return (abs(start_i - end_i) == 2 and abs(start_j - end_j) == 1) or (
            abs(start_i - end_i) == 1 and abs(start_j - end_j) == 2
        )
    elif piece == "pawn":
        # Pawns can only move forward one square, except on their first move where they can move two squares
        if start_j != end_j:
            return False
        if start_i == 1 or start_i == 6:
            return abs(start_i - end_i) <= 2
        return abs(start_i - end_i) == 1
    return False


def move_piece(side: Sides, piece: Pieces, end: str, start: Optional[str]):
    """
    Move a piece on the board. Returns the updated board.
    You need to supply a side and a piece and an end location.
    If there is only one piece of that type, you can move it without specifying the start location.
    If there are multiple pieces of that type, you need to specify the start location.
    """
    print("move_piece")
    # Find the piece
    locations = find_piece(side, piece)
    if not locations:
        return "Piece not found."
    if len(locations) > 1 and not start:
        return "Multiple pieces found. Please specify the start location."
    if start:
        start_i = 8 - int(start[1])
        start_j = ord(start[0]) - ord("a")
        if (start_i, start_j) not in locations:
            return "Piece not found at start location."
    else:
        start_i, start_j = locations[0]
    # Find the end location
    end_i = 8 - int(end[1])
    end_j = ord(end[0]) - ord("a")
    if not is_valid_move(piece, start_i, start_j, end_i, end_j):
        return "Invalid move."
    # Move the piece
    board[end_i][end_j] = board[start_i][start_j]
    board[start_i][start_j] = "."
    winner = assess_victory()
    if winner != "No winner yet.":
        return winner
    return peek_board()


def assess_victory():
    """
    Assess the victory conditions of the game.
    """
    print("assess_victory")
    white_king = find_piece("white", "king")
    black_king = find_piece("black", "king")
    if not white_king:
        return "Black wins!"
    elif not black_king:
        return "White wins!"
    else:
        return "No winner yet."


system_message = (
    "You are playing chess. Use the tools to move pieces on the board and assess the victory conditions. "
    "The board is displayed after each move so move_piece can be your primary interaction. "
    "Let the player start and then make your move each round. "
    "You can stop when the game is over. "
)

if __name__ == "__main__":
    conversation = utils.Conversation(
        system_message=system_message, tools=[peek_board, is_valid_move, find_piece, move_piece]
    )
    conversation.main_loop()
