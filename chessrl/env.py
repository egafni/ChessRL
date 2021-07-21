import chess
import gym
import numpy as np

import typing
if typing.TYPE_CHECKING:
    from chessrl.agent import ChessAgent


class ChessEnv(gym.Env):
    def __init__(self, opponent: ChessAgent):
        self._ready = False
        self.opponent = opponent
        self.board = None

    def reset(self):
        self.board = chess.Board()
        self._ready = True

        return self.make_obs()

    def make_obs(self):
        """
        :return: numpy array of the board
        """
        # https://github.com/arjangroen/RLC/blob/master/RLC/real_chess/environment.py#L37
        # first 5 channels are for each piece
        # each position on the board is +1 if there is a white piece and -1 if there is a black piece
        obs = np.zeros((6, 8, 8))
        # 6th channel stores the # of moves made so far, since game can only go to 75 moves
        obs[5] = 1. / self.board.fullmove_number

        piece_symbol_to_channel = dict(p=0, r=1, n=2, b=3, q=4, k=5, P=0, R=1, N=2, B=3, Q=4, K=5)
        for pos, piece in self.board.piece_map().items():
            row, col = divmod(pos, 8)
            channel = piece_symbol_to_channel[piece.symbol()]
            obs[channel, row, col] = 1 if piece.color else -1

        return obs

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

    def step(self, action):
        """
        """
        if not self._ready:
            raise AssertionError(f'must call reset() first')

        if isinstance(action, np.ndarray):
            # action is a 64x64 array of from->to moves
            # mask illegal moves
            action = action[~self.get_legal_moves_mask()] = 0.0
            # from_to is a tuple of (from_square, to_square)
            from_to = np.unravel_index(np.argmax(action, axis=None), action.shape)
            move = chess.Move(*from_to)
        elif isinstance(action, chess.Move):
            move = action
        else:
            raise TypeError(f'{action} is invalid')

        # make the move
        self.board.push(move)

        obs = self.make_obs()

        done, reward = self._check_done()

        return obs, reward, done, dict()
