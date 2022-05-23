import chess
import gym
import numpy as np

from chessrl.model import ActorCritic


class ChessEnv(gym.Env):
    def __init__(self, opponent: ActorCritic):
        self._ready = False
        self.opponent = opponent
        self.board = None

        # from_space -> to_space
        self.action_space = gym.spaces.Discrete(64 * 64)

    def reset(self):
        self.board = chess.Board()
        self._ready = True

        return self.make_obs()

    def make_obs(self, mirror=False):
        """
        :return: numpy array of the board
        """
        board = self.board.mirror() if mirror else self.board

        # https://github.com/arjangroen/RLC/blob/master/RLC/real_chess/environment.py#L37
        # first 5 channels are for each piece
        # each position on the board is +1 if there is a white piece and -1 if there is a black piece
        obs = np.zeros((6, 8, 8))
        # 6th channel stores the # of moves made so far, since game can only go to 75 moves
        obs[5] = 1. / board.fullmove_number

        piece_symbol_to_channel = dict(p=0, r=1, n=2, b=3, q=4, k=5, P=0, R=1, N=2, B=3, Q=4, K=5)
        for pos, piece in board.piece_map().items():
            row, col = divmod(pos, 8)
            channel = piece_symbol_to_channel[piece.symbol()]
            obs[channel, row, col] = 1 if piece.color else -1

        return obs.astype(np.float32)

    def get_legal_moves_mask(self):
        """
        :return: 64,64 mask where it is True if from_square -> to_square is legal
        """
        mask = np.zeros(shape=(64, 64)).astype(bool)
        for move in self.board.generate_legal_moves():
            mask[move.from_square, move.to_square] = True
        return mask

    def _check_done(self):
        # Check if game has terminated
        outcome = self.board.outcome()
        if outcome is None:
            reward = 0.0
            done = False
        else:
            done = True
            if outcome.winner is None:
                reward = 0.0
            else:
                reward = 1.0 if outcome.winner else -1.0

        return done, reward

    def pi_to_chess_move(self, pi: np.ndarray):
        # action is a 64x64 array of from->to moves
        pi = pi.copy()
        # mask illegal moves
        pi[~self.get_legal_moves_mask()] = 0.0
        # from_square_to_square is a tuple of (from_square, to_square)
        from_square_to_square = np.unravel_index(np.argmax(pi, axis=None), pi.shape)
        move = chess.Move(*from_square_to_square)
        return move

    def mask_illegal_moves_from_pi_in_place(self, pi):
        pi[~self.get_legal_moves_mask().flatten()] = 0.0
        pi /= pi.sum()
        return pi

    @staticmethod
    def int_to_chess_move(i):
        from_square, to_square = np.unravel_index(i, (64, 64))
        return chess.Move(from_square, to_square)

    def step(self, action: chess.Move):
        """
        """
        if not self._ready:
            raise AssertionError(f'must call reset() first')
        if not self.opponent:
            raise AssertionError(f'must init with an opponent to call step()')

        if isinstance(action, int):
            action = self.int_to_chess_move(action)

        # make the move
        self.board.push(action)

        obs = self.make_obs()

        done, reward = self._check_done()

        return obs, reward, done, dict()
