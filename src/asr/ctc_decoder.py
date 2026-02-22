"""
OR-Symphony: CTC Decoder

Decodes CTC logits into text using greedy search or beam search.
Handles SentencePiece detokenization (▁ → space boundaries).

MedASR specifics:
  - Token 0 = <blk> (CTC blank)
  - Token 1 = <s> (BOS)
  - Token 2 = </s> (EOS)
  - Token 3 = <unk>
  - Token 4 = ▁ (word boundary / space)
  - Subsampling factor: 4

Usage:
    decoder = CTCDecoder("onnx_models/medasr/tokens.txt")
    text, confidence = decoder.decode(logits, lengths)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Special token IDs
BLANK_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SPACE_ID = 4  # ▁ token


class CTCDecoder:
    """
    CTC greedy decoder with SentencePiece detokenization.

    Decodes CTC logits by:
    1. Taking argmax at each timestep (greedy)
    2. Removing blank tokens (id=0)
    3. Collapsing consecutive duplicate tokens
    4. Joining SentencePiece tokens (▁ → word boundary)
    5. Computing confidence from softmax probabilities
    """

    def __init__(
        self,
        tokens_path: Union[str, Path],
        subsampling_factor: int = 4,
    ) -> None:
        """
        Initialize the CTC decoder.

        Args:
            tokens_path: Path to tokens.txt vocabulary file.
            subsampling_factor: Model's temporal subsampling factor.
        """
        self.tokens_path = Path(tokens_path)
        self.subsampling_factor = subsampling_factor
        self._id_to_token: Dict[int, str] = {}
        self._token_to_id: Dict[str, int] = {}
        self._load_vocab()

    def _load_vocab(self) -> None:
        """Load token vocabulary from file."""
        if not self.tokens_path.exists():
            raise FileNotFoundError(f"Tokens file not found: {self.tokens_path}")

        with open(self.tokens_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    token_str = parts[0]
                    token_id = int(parts[1])
                    self._id_to_token[token_id] = token_str
                    self._token_to_id[token_str] = token_id

        logger.info("Loaded %d tokens from %s", len(self._id_to_token), self.tokens_path)

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    def decode(
        self,
        logits: np.ndarray,
        lengths: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, float, List[int]]]:
        """
        Decode a batch of CTC logits.

        Args:
            logits: (N, T, V) float32 array of CTC logits.
            lengths: (N,) int64 array of valid output lengths.
                     If None, uses full T for all samples.

        Returns:
            List of (text, confidence, token_ids) tuples, one per batch item.
        """
        if logits.ndim == 2:
            logits = logits[np.newaxis, :, :]  # add batch dim

        batch_size = logits.shape[0]
        if lengths is None:
            lengths = np.full(batch_size, logits.shape[1], dtype=np.int64)

        results = []
        for i in range(batch_size):
            valid_logits = logits[i, : lengths[i], :]
            text, confidence, token_ids = self._decode_single(valid_logits)
            results.append((text, confidence, token_ids))

        return results

    def decode_single(
        self,
        logits: np.ndarray,
        length: Optional[int] = None,
    ) -> Tuple[str, float, List[int]]:
        """
        Decode a single utterance's CTC logits.

        Args:
            logits: (T, V) float32 logits.
            length: Valid output length. If None, uses full T.

        Returns:
            Tuple of (text, confidence, token_ids).
        """
        if length is not None:
            logits = logits[:length, :]
        return self._decode_single(logits)

    def _decode_single(self, logits: np.ndarray) -> Tuple[str, float, List[int]]:
        """
        Greedy CTC decode for a single utterance.

        Args:
            logits: (T, V) float32 logits.

        Returns:
            (decoded_text, confidence, raw_token_ids)
        """
        # Compute softmax probabilities for confidence
        probs = self._softmax(logits)

        # Greedy: argmax at each timestep
        best_ids = np.argmax(logits, axis=-1)  # (T,)
        best_probs = np.max(probs, axis=-1)  # (T,)

        # CTC collapse: remove blanks, then deduplicate consecutive tokens
        collapsed_ids: List[int] = []
        collapsed_probs: List[float] = []
        prev_id = -1

        for t_idx in range(len(best_ids)):
            token_id = int(best_ids[t_idx])
            prob = float(best_probs[t_idx])

            # Skip blank tokens
            if token_id == BLANK_ID:
                prev_id = token_id
                continue

            # Skip consecutive duplicates
            if token_id == prev_id:
                continue

            # Skip special tokens (BOS, EOS, UNK)
            if token_id in (BOS_ID, EOS_ID, UNK_ID):
                prev_id = token_id
                continue

            collapsed_ids.append(token_id)
            collapsed_probs.append(prob)
            prev_id = token_id

        # Convert token IDs to text
        text = self._tokens_to_text(collapsed_ids)

        # Confidence: geometric mean of non-blank token probabilities
        if collapsed_probs:
            log_probs = np.log(np.clip(collapsed_probs, 1e-10, 1.0))
            confidence = float(np.exp(np.mean(log_probs)))
        else:
            confidence = 0.0

        return text, confidence, collapsed_ids

    def _tokens_to_text(self, token_ids: List[int]) -> str:
        """
        Convert token IDs to text using SentencePiece convention.

        SentencePiece uses ▁ (U+2581) to mark word boundaries:
          - "▁the" → " the"
          - "s" → "s" (continuation of previous word)
          - "▁" alone → " " (standalone space)
        """
        if not token_ids:
            return ""

        pieces = [self._id_to_token.get(tid, "") for tid in token_ids]
        text = "".join(pieces)

        # Replace SentencePiece ▁ with spaces
        text = text.replace("\u2581", " ")

        # Clean up
        text = text.strip()
        # Collapse multiple spaces
        while "  " in text:
            text = text.replace("  ", " ")

        return text

    def compute_timestamps(
        self,
        logits: np.ndarray,
        length: Optional[int] = None,
        frame_shift_ms: float = 10.0,
    ) -> List[Tuple[str, float, float]]:
        """
        Compute per-token timestamps based on frame positions.

        Args:
            logits: (T, V) logits.
            length: Valid length.
            frame_shift_ms: Feature extraction hop size.

        Returns:
            List of (token_text, start_time_s, end_time_s) tuples.
        """
        if length is not None:
            logits = logits[:length, :]

        best_ids = np.argmax(logits, axis=-1)
        frame_duration_s = (frame_shift_ms * self.subsampling_factor) / 1000.0

        timestamps = []
        prev_id = -1

        for t_idx in range(len(best_ids)):
            token_id = int(best_ids[t_idx])
            if token_id == BLANK_ID or token_id == prev_id or token_id in (BOS_ID, EOS_ID, UNK_ID):
                prev_id = token_id
                continue

            token_text = self._id_to_token.get(token_id, "")
            start_s = t_idx * frame_duration_s
            end_s = (t_idx + 1) * frame_duration_s

            timestamps.append((token_text, start_s, end_s))
            prev_id = token_id

        return timestamps

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Compute softmax along last axis."""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
