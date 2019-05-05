import numpy as np


class UCTAgent:
    def __init__(self, transition_function, ava_action_function,
                 is_terminal_function, reward_function,
                 computational_budget: int, c: float=np.sqrt(2)):
        """Domain independent 2 player zero-sum UCT agent

        Args:
            transition_function (t: S x A -> S)         : Function from state action to new state
            ava_action_function (a: S -> A)             : Function from states to available actions
            is_terminal_function (d: S -> {True, False}): Fucntion from states to boolean values
            reward_function (r: S -> Z)                 : Fuction from states to rewards
            computational_budget (int)                  : Number of searches for MCTS
            c (float)                                   : Exploitation constant for UCT
        """
        self.comp_budget = computational_budget
        self.c = c
        self.transition_function = transition_function
        self.ava_action_function = ava_action_function
        self.is_terminal_function = is_terminal_function
        self.reward_function = reward_function

    def act(self, state, ava_actions):
        return self.UCTSearch(state)

    def UCTSearch(self, state_0):
        # create root node  with state state_0
        self.tree = MCTree(state_0)
        root_node = self.tree.get_head()
        root_node.ava_actions = self.ava_action_function(state_0)
        for _ in range(self.comp_budget):
            node = self.TreePolicy(root_node)
            reward = self.DefaultPolicy(node.get_state())
            self.BackupNegaMax(node, reward)
        return self.BestChild(self.tree.get_head()).action

    def TreePolicy(self, node):
        while not self.is_terminal_function(node.state):
            if not node.is_fully_expanded():
                return self.Expand(node)
            else:
                node = self.BestChild(node)
        return node

    def Expand(self, node):
        # choose a ∈ untried actions from A(s(v))
        ava_actions = self.ava_action_function(node.get_state())
        action = node.get_untried_action(ava_actions)
        new_state = self.transition_function(node.state, action)
        new_ava_actions = self.ava_action_function(new_state)
        return self.tree.expand(node, new_state, action, new_ava_actions)

    def __calculate_bound(self, node) -> float:
        if node.s_i != 0:
            return -node.w_i/node.s_i + \
                   self.c*np.sqrt(np.log(node.parent.s_i)/node.s_i)
        return np.inf

    def BestChild(self, node):
        arg_id = np.argmax(list(map(self.__calculate_bound, node.children)))
        return node.children[arg_id]

    def DefaultPolicy(self, state):
        while not self.is_terminal_function(state):
            ava_actions = self.ava_action_function(state)
            action = np.random.choice(ava_actions)
            state = self.transition_function(state, action)
        return self.reward_function(state)

    def BackupNegaMax(self, node, reward):
        while node is not None:
            node.s_i += 1
            node.w_i += reward[node.player_id]
            node = node.parent


class Node:
    def __init__(self):
        self.parent = None
        self.children = []
        # this node’s number of simulations that resulted in a win
        self.w_i = 0
        # this node’s total number of simulations
        self.s_i = 0

        self.terminal = False
        self.state = None
        self.player_id = None

        self.ava_actions = []

    def get_state(self):
        return self.state

    def is_fully_expanded(self):
        return len(self.ava_actions) == len(self.children)

    def get_untried_action(self, ava_actions):
        for a in ava_actions:
            tried = False
            for child in self.children:
                if child.action == a:
                    tried = True
                    break
            if tried is False:
                return a
        return None

    def __repr__(self):
        return '({}/{}); id: {}'.format(self.w_i, self.s_i, self.player_id)

    def __str__(self, level=0):
        ret = "----"*level+repr(self)+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret


class MCTree:
    def __init__(self, state):
        self.head = Node()
        self.head.player_id = state[1]
        self.head.state = state

    def get_head(self):
        return self.head

    def expand(self, node, state, action, ava_actions):
        new_node = Node()
        new_node.parent = node
        new_node.action = action
        new_node.player_id = state[1]
        node.children.append(new_node)
        new_node.state = state
        new_node.ava_actions = ava_actions
        return new_node

    def __repr__(self):
        return repr(self.head)
