import pyximport; pyximport.install()

import random
import torch
import config
import util
import Action

# Fetching arguments and selection version from configuration and Action module
args = config.parser.parse_args()

class State:
    """Class to represent a State in the environment"""
    
    def __init__(self, piece, subj, test):
        """
        Initializes a new state.
        :param piece: The piece representing the current state
        :param subj: Subject for the state
        :param test: A boolean indicating whether this is a test state
        """
        self.piece = piece  # Storing the piece representing the current state
        self.subj = subj  # Storing the subject for the state
        
        # Assigning the data for the state based on whether it's a test or not
        if test:
            self.data = util.test_data()[subj]
        else:
            self.data = util.validation_data(args.split)[subj]

    def next(self, action, test):
        """
        Transitions to the next state based on the given action.
        :param action: The action to perform
        :param test: A boolean indicating whether the next state is a test state
        :return: The next state after performing the action
        """
        piece = Action.Action(action, self.piece, self.data)  # Getting the piece for the next state
        return State(piece, self.subj, test)  # Returning the next state

    def legal_actions(self):
        """
        Gets all legal actions possible from the current state.
        :return: A list of legal actions
        """
        actions = [i for i in range(args.action)]  # Listing all possible actions
        return actions  # Returning the list of legal actions
