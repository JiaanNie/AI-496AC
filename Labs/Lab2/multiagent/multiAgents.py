# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
from game import Actions
from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        #print("#######################################################################\n\n")
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        #print "Pacman current position is:", currentGameState.getPacmanPosition()
        #print "current game score is: ", currentGameState.getScore()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        #print "successor pacman position", newPos
        foods_list = newFood.asList()
        pacman_x, pacman_y = newPos
        nearest_food_distance  = 999999
        nearest_ghost_distance  = 999999
        food_factor = len(foods_list)
        for food in foods_list:
            food_x, food_y  = food
            diff_x =  food_x - pacman_x
            diff_y = food_y - pacman_y
            distance = abs(diff_x) + abs(diff_y)
            if nearest_food_distance > distance:
                nearest_food_distance = distance


        for ghost in newGhostStates:
            ghost_x, ghost_y  = ghost.getPosition()
            diff_x = ghost_x - pacman_x
            diff_y = ghost_y - pacman_y
            distance = abs(diff_x) + abs(diff_y)
            if nearest_ghost_distance > distance:
                nearest_ghost_distance = distance
        factor = 2
        if nearest_ghost_distance <= 2:
            factor  = 500

        return successorGameState.getScore() - 1  + 10* (1/float(nearest_food_distance)) - factor * 1/(nearest_ghost_distance +1)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        pacman_agent = 0
        initial_depth = 0
        bestScore = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(pacman_agent):
            successor_state = gameState.generateSuccessor(pacman_agent, action)
            score = self.minMaxFunctionDispatcher(successor_state, initial_depth, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

    def minMaxFunctionDispatcher(self, current_state, current_depth, agent):
        if current_depth == self.depth or current_state.isWin() or current_state.isLose():
            return self.evaluationFunction(current_state)
        elif agent == 0: # 0 is pacman
            return self.maximizer(current_state, current_depth, agent)
        else: # otherwise is ghost
            return self.minimizer(current_state, current_depth, agent)


    def maximizer(self, current_state, current_depth, agent):
        return_value  = float("-inf")
        for action in current_state.getLegalActions(agent):
            return_value  = max(return_value, self.minMaxFunctionDispatcher(current_state.generateSuccessor(agent, action), current_depth, 1))
        return return_value

    def minimizer(self, current_state, current_depth, agent):
        return_value = float("inf")
        for action in current_state.getLegalActions(agent):
            if agent == current_state.getNumAgents() - 1:  #if agent is the last agent in the game that mean we done for all min agent within the current_depth so we can move on to the next depth set the agent to the pacman again
                return_value  = min(return_value, self.minMaxFunctionDispatcher(current_state.generateSuccessor(agent, action), current_depth + 1, 0))
            else:
                return_value  = min(return_value, self.minMaxFunctionDispatcher(current_state.generateSuccessor(agent, action), current_depth, agent + 1))
        return return_value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        pacman_agent = 0
        initial_depth = 0
        beta = float("inf")
        alpha = float("-inf")
        bestScore = float("-inf")
        bestAction = None

        for action in gameState.getLegalActions(pacman_agent):
            successor_state = gameState.generateSuccessor(pacman_agent, action)
            score = self.minMaxFunctionDispatcher(successor_state, initial_depth, 1, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(bestScore, alpha)
        return bestAction

    def minMaxFunctionDispatcher(self, current_state, current_depth, agent, alpha, beta):
        if current_depth == self.depth or current_state.isWin() or current_state.isLose():
            return self.evaluationFunction(current_state)
        elif agent == 0: # 0 is pacman
            return self.maximizer(current_state, current_depth, agent, alpha, beta)
        else: # otherwise is ghost
            return self.minimizer(current_state, current_depth, agent, alpha, beta)


    def maximizer(self, current_state, current_depth, agent, alpha, beta):
        return_value  = float("-inf")
        for action in current_state.getLegalActions(agent):
            return_value  = max(return_value, self.minMaxFunctionDispatcher(current_state.generateSuccessor(agent, action), current_depth, 1, alpha, beta))
            if return_value >beta:
                return return_value
            alpha =  max(alpha, return_value)
        return return_value

    def minimizer(self, current_state, current_depth, agent, alpha, beta):
        return_value = float("inf")
        for action in current_state.getLegalActions(agent):
            if agent == current_state.getNumAgents() - 1:  #if agent is the last agent in the game that mean we done for all min agent within the current_depth so we can move on to the next depth set the agent to the pacman again
                return_value  = min(return_value, self.minMaxFunctionDispatcher(current_state.generateSuccessor(agent, action), current_depth + 1, 0, alpha, beta))
            else:
                return_value  = min(return_value, self.minMaxFunctionDispatcher(current_state.generateSuccessor(agent, action), current_depth, agent + 1, alpha, beta))
            if return_value < alpha:
                return return_value
            beta = min(beta, return_value)
        return return_value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        pacman_agent = 0
        initial_depth = 0
        bestScore = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(pacman_agent):
            successor_state = gameState.generateSuccessor(pacman_agent, action)
            score = self.expectedMaxFunctionDispatcher(successor_state, initial_depth, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

    def expectedMaxFunctionDispatcher(self, current_state, current_depth, agent):
        if current_depth == self.depth or current_state.isWin() or current_state.isLose():
            return self.evaluationFunction(current_state)
        elif agent == 0: # 0 is pacman
            return self.maximizer(current_state, current_depth, agent)
        else: # otherwise is ghost
            return self.expected_value(current_state, current_depth, agent)


    def maximizer(self, current_state, current_depth, agent):
        return_value  = float("-inf")
        for action in current_state.getLegalActions(agent):
            return_value  = max(return_value, self.expectedMaxFunctionDispatcher(current_state.generateSuccessor(agent, action), current_depth, 1))
        return return_value

    def expected_value(self, current_state, current_depth, agent):
        return_value = 0
        for action in current_state.getLegalActions(agent):
            probability =  1.0/float(len(current_state.getLegalActions(agent)))
            if agent == current_state.getNumAgents() - 1:
                return_value = return_value + probability * self.expectedMaxFunctionDispatcher(current_state.generateSuccessor(agent, action), current_depth + 1, 0 )
            else:
                return_value = return_value + probability * self.expectedMaxFunctionDispatcher(current_state.generateSuccessor(agent, action), current_depth, agent +1 )
        return return_value


        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    pacman_current_x, pacman_current_y= currentGameState.getPacmanPosition()
    current_ghosts_state = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in current_ghosts_state]
    foods_list = currentGameState.getFood().asList()
    closest_ghost = -1
    ghost_factor  = 1
    food_factor = 1
    for ghost in current_ghosts_state:
        ghost_x, ghost_y = ghost.getPosition()
        diff_x = ghost_x - pacman_current_x
        diff_y = ghost_y - pacman_current_y
        ghost_distance = abs(diff_x) + abs(diff_y)
        if ghost_distance > closest_ghost:
            closest_ghost = ghost_distance

    if closest_ghost <=2:
        ghost_factor = 20
    else:
        food_factor =10
        ghost_factor =1
    nearest_food_distance = float("inf")
    for food in foods_list:
        food_x, food_y  = food
        diff_x =  food_x - pacman_current_x
        diff_y = food_y - pacman_current_y
        distance = abs(diff_x) + abs(diff_y)
        if nearest_food_distance > distance:
            nearest_food_distance = distance
    if newScaredTimes[0] > 0:
        food_factor = 100
        ghost_factor = 0

    return currentGameState.getScore()  + food_factor * (1.0/(nearest_food_distance +1)) - (ghost_factor* 1.0/(closest_ghost+1))
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
