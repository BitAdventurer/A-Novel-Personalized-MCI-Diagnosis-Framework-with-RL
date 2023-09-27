import pyximport
pyximport.install()

import random
import torch
import config
import util
import Action_lap

# Fetching arguments and selection version from configuration and Action_lap module
args = config.parser.parse_args()

class State:
    """Class to represent a state in the environment."""
    
    def __init__(self, piece):
        """
        Initializes a new state.
        :param piece: The piece representing the current state
        """
        self.piece = piece  # Storing the piece representing the current state
        xx = torch.rand(116, 116)  # Creating a random tensor of shape (116, 116)
        self.data = xx.new_ones(116, 116, dtype=float)  # Creating a ones tensor of shape (116, 116)
    
    def next(self, action):
        """
        Transitions to the next state based on the given action.
        :param action: The action to perform
        :return: The next state after performing the action
        """
        piece = Action_lap.Action(action, self.piece, self.data)  # Getting the piece for the next state
        return State(piece)  # Returning the next state, recursive
    
    def legal_actions(self):
        """
        Gets all legal actions possible from the current state.
        :return: A list of legal actions
        """
        actions = [i for i in range(args.action)]  # Listing all possible actions
        return actions  # Returning the list of legal actions
