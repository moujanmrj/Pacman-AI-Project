# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            temp = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    temp[state] = 0
                else:
                    maxVal = -100000
                    actions = self.mdp.getPossibleActions(state)
                    for action in actions:
                        tmp = self.mdp.getTransitionStatesAndProbs(state, action)
                        value = 0
                        for prob_state in tmp:
                            value += prob_state[1] * (self.mdp.getReward(state, action, prob_state[1]) + self.discount * self.values[prob_state[0]])
                        maxVal = max(value, maxVal)
                    if maxVal != -100000:
                        temp[state] = maxVal
            self.values = temp


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        QValue = 0
        for prob_state in self.mdp.getTransitionStatesAndProbs(state, action):
            QValue += prob_state[1] * (self.mdp.getReward(state, action, prob_state[1]) + self.discount * self.values[prob_state[0]])
        return QValue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        actions = self.mdp.getPossibleActions(state)
        actionsList = {}
        for action in actions:
            actionsList[action] = self.computeQValueFromValues(state, action)

        return max(actionsList, key=actionsList.get)
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
            if self.mdp.isTerminal(states[i % len(states)]):
                continue
            maxVal = -100000
            for action in self.mdp.getPossibleActions(states[i % len(states)]):
                actionValue = 0
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(states[i % len(states)], action):
                    actionValue += prob * (self.mdp.getReward(states[i % len(states)], action, nextState) + self.discount * self.values[nextState])
                maxVal = max(maxVal, actionValue)
            self.values[states[i % len(states)]] = maxVal

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        plist = {}
        for state in [theState for theState in states if not self.mdp.isTerminal(theState)]:
            for action in self.mdp.getPossibleActions(state):
                for ns in [pair[0] for pair in self.mdp.getTransitionStatesAndProbs(state, action)]:
                    if ns not in plist:
                        plist[ns] = set()
                    plist[ns].add(state)

        pQueue = util.PriorityQueue()
        for state in states:
            if not self.mdp.isTerminal(state):
                maximum = max([self.computeQValueFromValues(state, theAction) for theAction in self.mdp.getPossibleActions(state)])
                pQueue.update(state, -(abs(maximum - self.getValue(state))))

        for i in range(self.iterations):
            if pQueue.isEmpty():
                break
            state = pQueue.pop()
            if not self.mdp.isTerminal(state):
                maximum = max([self.computeQValueFromValues(state, theAction) for theAction in self.mdp.getPossibleActions(state)])
                self.values[state] = maximum
                for p in plist[state]:
                    maximum = max([self.computeQValueFromValues(p, theAction) for theAction in self.mdp.getPossibleActions(p)])
                    if (abs(maximum - self.getValue(p))) > self.theta:
                        pQueue.update(p, -(abs(maximum - self.getValue(p))))

