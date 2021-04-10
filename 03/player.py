class MyPlayer:
    """Player which is based on moves value computation."""
 
    def __init__(self, my_color, opponent_color, board_size=8):
        self.name = 'forstluk'  # student username
        self.my_color = my_color  # my color -> 1 or 0
        self.opponent_color = opponent_color  # opponent color -> 1 or 0
        self.board = []  # received board from system
        self.my_return_combination = ()  # my final return
        self.possible_moves = []  # combination of possible moves
        self.moves_value = []  # based on distance and number of opponent's cells between my cell and target cell
        self.rows = board_size
        self.columns = board_size
        self.position_is_empty = -1
 
        self.great_moves = [(0, 0), (0, 7), (7, 0), (7, 7)]                               # table of good or bad moves
        self.really_bad_moves = [(1, 1), (1, 6), (6, 1), (6, 6)]                          # comes from Wikipedia
        self.bad_moves = [(1, 0), (6, 0), (0, 1), (7, 1), (0, 6), (7, 6), (1, 7), (6, 7)]
 
    def move(self, board):
        self.possible_moves = []
        self.moves_value = []
        self.board = board
 
        self.find_possible_moves()  # it will find every possible move and provide me values of every move
 
        if(len(self.possible_moves) == 0):
            print(self.possible_moves)
            return None
        else:
            self.find_winning_moves()  # compute values of every move and find the best one
 
        return self.my_return_combination
 
    def find_possible_moves(self):
        """
        it will find every possible move on the board
        algorithm will find every opponent's cell which is next to the empty cell
        afterwards it will send opponent's cell position to the next function which will
        find out if it's possible to move on the empty position
        """
        for row in range(0, self.rows):
            for column in range(0, self.columns):
                if(self.board[row][column] == self.opponent_color):
                    if (row + 1 < self.rows and self.board[row + 1][column] == self.position_is_empty):
                        self.bottom_side(row, column)
                    if (row + 1 < self.rows and column + 1< self.columns and self.board[row + 1][column + 1] == self.position_is_empty):
                        self.diagonal_right_bottom(row, column)
                    if (row + 1 < self.rows and column != 0 and self.board[row + 1][column - 1] == self.position_is_empty):
                        self.diagonal_left_bottom(row, column)
                    if (row != 0 and self.board[row - 1][column] == self.position_is_empty):
                        self.upper_side(row, column)
                    if (row != 0 and column + 1< self.columns and self.board[row - 1][column + 1] == self.position_is_empty):
                        self.diagonal_right_upper(row, column)
                    if (row != 0 and column != 0 and self.board[row - 1][column - 1] == self.position_is_empty):
                        self.diagonal_left_upper(row, column)
                    if (column + 1 < self.columns and  self.board[row][column + 1] == self.position_is_empty):
                        self.right_side(row,column)
                    if (column != 0 and self.board[row][column - 1] == self.position_is_empty):
                        self.left_side(row, column)
 
    def find_winning_moves(self):
        """
        computing of importance of single moves based on how many opponent's cells are between my cell and target cell
        :return: self.return_combination is determined there
        """
        for move in self.possible_moves:
            for great in self.great_moves:
                if move == great:  # if it's possible to play one of the best moves available, play it
                    self.my_return_combination = move
                    return 0
 
        for i in range(len(self.possible_moves)):
            for j in range(len(self.really_bad_moves)):
                if(self.possible_moves[i] == self.really_bad_moves[j]):  # array of really bad moves, where can
                    self.moves_value[i] = -5                              # opponent take control over whole board
            for z in range(len(self.bad_moves)):    # not as bad moves, but still not as good as others
                if (self.possible_moves[i] == self.bad_moves[z]):
                    self.moves_value[i] -= 5    # edit of importance of move
 
        maximal_value = -6
        final_index = 0
        for i in range(len(self.moves_value)):  # calculating the best move with biggest reward
            if self.moves_value[i] > maximal_value:
                maximal_value = self.moves_value[i]
                final_index = i
 
        self.my_return_combination = self.possible_moves[final_index]
 
    def diagonal_left_upper(self, row, column):
        """
        checks upper left diagonal position
        :param row: opponents position row
        :param column: opponents position column
        :return: True or False -> found move or not
        """
        diagonal_mover = 1
        if(row-1 < 0 or column-1 < 0):
            return False
        elif(self.board[row-diagonal_mover][column-diagonal_mover] == self.position_is_empty):
            while(row+diagonal_mover < self.rows and column+diagonal_mover < self.columns):
                if(self.board[row+diagonal_mover][column+diagonal_mover] == self.opponent_color):
                    diagonal_mover += 1
                    continue
                elif(self.board[row+diagonal_mover][column+diagonal_mover] == self.position_is_empty):
                    return False
                elif(self.board[row+diagonal_mover][column+diagonal_mover] == self.my_color):
                    self.moves_value.append(diagonal_mover)
                    self.possible_moves.append((row-1,column-1))
                    return True
                return False
        else:
            return False
    def diagonal_left_bottom(self, row, column):
        """
        checks bottom left diagonal position
        :param row: opponents position row
        :param column: opponents position column
        :return: True or False -> found move or not
        """
        diagonal_mover = 1
        if (row+1 >= self.rows or column - 1 < 0):
            return False
        elif (self.board[row+diagonal_mover][column - diagonal_mover] != self.my_color):
            while (row-diagonal_mover != -1 and column + diagonal_mover < self.columns):
                if (self.board[row - diagonal_mover][column + diagonal_mover] == self.opponent_color):
                    diagonal_mover += 1
                    continue
                elif (self.board[row - diagonal_mover][column + diagonal_mover] == self.position_is_empty):
                    return False
                elif (self.board[row - diagonal_mover][column + diagonal_mover] == self.my_color):
                    self.moves_value.append(diagonal_mover)
                    self.possible_moves.append((row + 1, column - 1))
                    return True
                return False
        else:
            return False
    def diagonal_right_upper(self, row, column):
        """
        checks upper right diagonal position
        :param row: opponents position row
        :param column: opponents position column
        :return: True or False -> found move or not
        """
        diagonal_mover = 1
        if (row - 1 < 0 or column+1 >= self.columns):
            return False
        elif (self.board[row - diagonal_mover][column + diagonal_mover] != self.my_color):
            while (row + diagonal_mover < self.rows and column - diagonal_mover != -1):
                if (self.board[row + diagonal_mover][column - diagonal_mover] == self.opponent_color):
                    diagonal_mover += 1
                    continue
                elif (self.board[row + diagonal_mover][column - diagonal_mover] == self.position_is_empty):
                    return False
                elif (self.board[row + diagonal_mover][column - diagonal_mover] == self.my_color):
                    self.moves_value.append(diagonal_mover)
                    self.possible_moves.append((row - 1, column + 1))
                    return True
                return False
        else:
            return False
    def diagonal_right_bottom(self, row,column):
        """
        checks bottom right diagonal position
        :param row: opponents position row
        :param column: opponents position column
        :return: True or False -> found move or not
        """
        diagonal_mover = 1
        if(row+1 >= self.rows or column+1 >= self.columns):
            return False
        elif(self.board[row+diagonal_mover][column+diagonal_mover] == self.position_is_empty):
            while(row-diagonal_mover != -1 and column-diagonal_mover != -1):
                if(self.board[row-diagonal_mover][column-diagonal_mover] == self.opponent_color):
                    diagonal_mover += 1
                    continue
                elif(self.board[row-diagonal_mover][column-diagonal_mover] == self.position_is_empty):
                    return False
                elif(self.board[row-diagonal_mover][column-diagonal_mover] == self.my_color):
                    self.moves_value.append(diagonal_mover)
                    self.possible_moves.append((row+1,column+1))
                    return True
                return False
        else:
            return False
 
    def left_side(self, row, column):
        side_mover = 1
        if(column - side_mover == -1):
            return False
        elif(self.board[row][column-side_mover] == self.position_is_empty):
            while (column + side_mover < self.columns):
                if(self.board[row][column+side_mover] == self.opponent_color):
                    side_mover += 1
                    continue
                elif(self.board[row][column+side_mover] == self.position_is_empty):
                    return False
                elif (self.board[row][column+side_mover] == self.my_color):
                    self.moves_value.append(side_mover)
                    self.possible_moves.append((row, column - 1))
                    return True
                return False
        else:
            return False
    def right_side(self, row, column):
        side_mover = 1
        if (column + side_mover >= self.columns):
            return False
        elif (self.board[row][column + side_mover] == self.position_is_empty):
            while (column - side_mover != -1):
                if (self.board[row][column - side_mover] == self.opponent_color):
                    side_mover += 1
                    continue
                elif (self.board[row][column - side_mover] == self.position_is_empty):
                    return False
                elif (self.board[row][column - side_mover] == self.my_color):
                    self.moves_value.append(side_mover)
                    self.possible_moves.append((row, column + 1))
                    return True
                return False
        else:
            return False
    def upper_side(self, row, column):
        side_mover = 1
        if (row - side_mover == -1):
            return False
        elif (self.board[row - side_mover][column] == self.position_is_empty):
            while (row + side_mover < self.rows):
                if (self.board[row+side_mover][column] == self.opponent_color):
                    side_mover += 1
                    continue
                elif (self.board[row+side_mover][column] == self.position_is_empty):
                    return False
                elif (self.board[row+side_mover][column] == self.my_color):
                    self.moves_value.append(side_mover)
                    self.possible_moves.append((row-1, column))
                    return True
                return False
        else:
            return False
    def bottom_side(self, row, column):
        side_mover = 1
        if (row + side_mover >= self.rows):
            return False
        elif (self.board[row + side_mover][column] == self.position_is_empty):
            while (row - side_mover != -1):
                if (self.board[row - side_mover][column] == self.opponent_color):
                    side_mover += 1
                    continue
                elif (self.board[row - side_mover][column] == self.position_is_empty):
                    return False
                elif (self.board[row - side_mover][column] == self.my_color):
                    self.moves_value.append(side_mover)
                    self.possible_moves.append((row + 1, column))
                    return True
        else:
            return False
