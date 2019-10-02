# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# def generalSearch(problem, )
# def GeneralSearch(problem, )

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    # result = depthFirstSearchHelper(problem, problem.getStartState())
    # return result[1]
    ## the two line above is using recursive calling
    stack = util.Stack()
    path = []
    direction = []
    visited_states = []
    stack.push((problem.getStartState(),direction))
    ## pushing inital state

    while stack.isEmpty() == False:
        (current_state, direction) = stack.pop()
        if problem.isGoalState(current_state):
            return direction
        if current_state not in visited_states:
            visited_states.append(current_state)
            for successor in problem.getSuccessors(current_state):
                newpath =  direction + [successor[1]]
                stack.push((successor[0], newpath))
    return path

#the helper function will required a global visited_states list
def depthFirstSearchHelper(problem, current_state):
    result = (False, [])
    if problem.isGoalState(current_state) == True:
        return (True, [])
    else:
        if current_state not in visited_states:
            visited_states.append(current_state)
            for successor in problem.getSuccessors(current_state):
                #successor = ((x,y),"Direction", weight)
                if successor[0] not in visited_states and successor[0]:
                    result = depthFirstSearchHelper(problem, successor[0])
                    if result[0] == True:
                        result = (True , [successor[1]] + result[1])
                        #print(result[1])
                        return result
                close_set.append(successor[0])
        return result
###########################
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    stack = util.Queue()
    path = []
    direction = []
    visited_states = []
    stack.push((problem.getStartState(),direction))
    ## pushing inital state

    while stack.isEmpty() == False:
        (current_state, direction) = stack.pop()
        if problem.isGoalState(current_state):
            return direction
        if current_state not in visited_states:
            visited_states.append(current_state)
            for successor in problem.getSuccessors(current_state):
                newpath =  direction + [successor[1]]
                stack.push((successor[0], newpath))
    return path
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # this priority quque it pick the lowest value in the priority
    fringe = util.PriorityQueue()
    visited_states = []
    direction = []
    path = []
    total_cost  = 0
    start_cost = 0
    ## inital state
    start_state = (problem.getStartState(), direction, start_cost)
    fringe.push(start_state, start_cost)
    while fringe.isEmpty() == False:
        (current_state, direction, cost) = fringe.pop()
        if problem.isGoalState(current_state) == True:
            path = path + direction
            total_cost = total_cost  + cost
            return path
        if current_state not in visited_states:
            visited_states.append(current_state)
            for successor in problem.getSuccessors(current_state):
                newpath =  direction + [successor[1]]
                newcost = cost + successor[2]
                fringe.push((successor[0], newpath, newcost), newcost)
    return path
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    fringe = util.PriorityQueue()
    visited_states = []
    direction = []
    path = []
    total_cost  = 0

    #start_priority = heuristic(problem.getStartState(, problem)
    ## inital state
    start_state = (problem.getStartState(), direction, 0)
    start_priority = heuristic(problem.getStartState(), problem)
    fringe.push(start_state, start_priority)
    while fringe.isEmpty() == False:
        (current_state, direction, cost) = fringe.pop()
        if problem.isGoalState(current_state) == True:
            path = path + direction
            total_cost = total_cost  + cost
            return path
        if current_state not in visited_states:
            visited_states.append(current_state)
            list_of_successors  = problem.getSuccessors(current_state)
            #print("current state: ", current_state, "its successors: ", list_of_successors)
            for successor in list_of_successors:
                if successor[0] not in visited_states:
                    newpath =  direction + [successor[1]]
                    newcost = cost + successor[2]
                    heuristic_value  = heuristic(successor[0], problem)
                    #print("successor: ", successor[0], "its heuristic: ", heuristic_value)
                    priority = cost + successor[2] + heuristic_value
                    fringe.push((successor[0], newpath, newcost), priority)
            #print("###########################################################\n\n")
    return path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
