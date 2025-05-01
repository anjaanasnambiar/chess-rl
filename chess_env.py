import chess
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ChessEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board = chess.Board()

        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.uint8)
        self.action_space = spaces.Discrete(4672)

        self.move_lookup = [move.uci() for move in self.board.legal_moves]
        while len(self.move_lookup) < 4672:
            self.move_lookup.append("0000")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.board.reset()
        self._update_move_lookup()
        return self._get_obs(), {}

    def step(self, action):
        move_str = self.move_lookup[action]
        move = chess.Move.from_uci(move_str)
        reward = 0
        terminated = False
        truncated = False

        if move in self.board.legal_moves:
            self.board.push(move)
            self._update_move_lookup()
        else:
            reward = -1
            terminated = True
            return self._get_obs(), reward, terminated, truncated, {}

        if self.board.is_checkmate():
            reward = 1
            terminated = True
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            reward = 0.5
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _update_move_lookup(self):
        self.move_lookup = [m.uci() for m in self.board.legal_moves]
        while len(self.move_lookup) < 4672:
            self.move_lookup.append("0000")

    def _get_obs(self):
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        obs = np.zeros((8, 8, 12), dtype=np.uint8)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                idx = piece_map[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
                obs[row, col, idx] = 1
        return obs

    def render(self):
        print(self.board)
