# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

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
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # value of each state; a Counter is a dict with default 0
    self.Q = util.Counter()
     
    for i in xrange(self.iterations):
      states = self.mdp.getStates()
      for s in states:
        if not self.mdp.isTerminal(s):
          actions = self.mdp.getPossibleActions(s)
          for action in actions: 
            transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(s,action)
            tSum = 0
            for (nextState, T) in transitionStatesAndProbs:
              tSum += T*self.values[(i,s)]
            self.Q[(s,action)] = self.mdp.getReward(s, action, (0,0)) + tSum
          optPolicy = self.getPolicy(s)
          print 'opt = ',optPolicy
          self.values[(i,s)] = self.getQValue(s, optPolicy)
        else:
          continue 

        
             
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]

  def getQValueWithLayer(self, state, action, layer):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    print 'action1 = ',action
    transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state,action)
    tSum = 0
    for (nextState, T) in transitionStatesAndProbs:
      tSum += T*self.values[(layer, state)]
    return self.mdp.getReward(state, action, (0,0)) + tSum


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    print 'action = ',action
    transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state,action)
    tSum = 0
    for (nextState, T) in transitionStatesAndProbs:
      tSum += T*self.values[(self.iterations,state)]
    return self.mdp.getReward(state, action, (0,0)) + tSum

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    print state
    if self.mdp.isTerminal(state):
      return None
    actions = self.mdp.getPossibleActions(state)
    maxQ = -1000
    optA = None
    for a in actions:
      if self.getQValue(state, a) > maxQ:
        maxQ = self.getQValue(state, a)
        optA = a
    return optA
    

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
