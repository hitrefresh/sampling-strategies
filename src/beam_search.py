# Implement beam search, good sentence decoding strategy for NLP tasks.
# High probability product of the sentence

# Use log prob to prevent underflow
# Min-heap instead of sorting the sequences
# If vocabular V is small, we can do exact Viterbi decoding with time complexity O(L * V^2)
# where L is the max length of the sequence

from dataclasses import dataclass
from typing import List, Tuple
import math
import heapq


@dataclass
class Node:
    # Current word, Specials words: <sos>, <eos>
    # for start and end of sentence
    word: str

    # Same as vocabulary size
    children: List["Node"]

    # Parent for current node, None for root
    parent: "Node"

    # Probability of the word given the parent sequence
    prob: float


class Sequence:
    # Log p robability of the sequence
    logprob: float

    # last node of the sequence
    last_node: Node

# Vocabulary size = V, Beam width = B, Max length = L
# Time complexity: O(L * (B * V + B * log(B*V))) = O(L * B * V)
# Space complexity: O(B * V)
def beam_search(root: Node, beam_width: int) -> Tuple[List[str], float]:
    """
    Beam search algorithm to find the best approx sequence of words
    Args:
        root: root node of the search tree
        beam_width: width of the beam
    Returns:
        Tuple of list of words in the best sequence and log probability of the sequence
    """
    curr_node = root

    final_sequences: List[Sequence] = []
    curr_sequences: List[Sequence] = [
        Sequence(
            last_node=curr_node,
            prob=math.log(max(curr_node.prob, 1e-12)),  # to prevent underflow
        )
    ]

    while len(curr_sequences) > 0:
        next_sequences: List[Sequence] = []
        for seq in curr_sequences:
            if seq.last_node.word == "<eos>":
                # It was in top-K at current level
                # and sequence prob will get smaller
                final_sequences.append(seq)
                beam_width -= 1
            else:
                for node in seq.last_node.children:
                    next_sequences.append(
                        Sequence(
                            logprob=math.log(max(node.prob, 1e-12)) + seq.logprob,
                            last_node=node,
                        )
                    )
        # Select top-K elements from the seq
        curr_sequences = heapq.nlargest(
            beam_width, next_sequences, key=lambda seq: seq.logprob
        )

    max_seq = max(final_sequences, key=lambda seq: seq.logprob)

    # Travese the max prob sequence to get the words
    max_seq_words = []
    node = max_seq.last_node
    while node is not None:
        max_seq_words.append(node.word)
        node = node.parent
    max_seq_words = reversed(max_seq_words)
    return max_seq_words, max_seq.prob
