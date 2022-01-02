#########################################
#                                       #
#                                       #
#  ==  SOKOBAN STUDENT AGENT CODE  ==   #
#                                       #
#      Written by: Abhi Sen             #
#                                       #
#                                       #
#########################################


# SOLVER CLASSES WHERE AGENT CODES GO
from helper import *
import random
import math


# Base class of agent (DO NOT TOUCH!)
class Agent:
    def getSolution(self, state, maxIterations):

        '''
        EXAMPLE USE FOR TREE SEARCH AGENT:


        #expand the tree until the iterations runs out or a solution sequence is found
        while (iterations < maxIterations or maxIterations <= 0) and len(queue) > 0:
            iterations += 1

            [ POP NODE OFF OF QUEUE ]

            [ EVALUATE NODE AS WIN STATE]
                [ IF WIN STATE: BREAK AND RETURN NODE'S ACTION SEQUENCE]

            [ GET NODE'S CHILDREN ]

            [ ADD VALID CHILDREN TO QUEUE ]

            [ SAVE CURRENT BEST NODE ]


        '''


        '''
        EXAMPLE USE FOR EVOLUTION BASED AGENT:
        #expand the tree until the iterations runs out or a solution sequence is found
        while (iterations < maxIterations or maxIterations <= 0) and len(queue) > 0:
            iterations += 1

            [ MUTATE ]

            [ EVALUATE ]
                [ IF WIN STATE: BREAK AND RETURN ]

            [ SAVE CURRENT BEST ]

        '''


        return []       # set of actions


#####       EXAMPLE AGENTS      #####

# Do Nothing Agent code - the laziest of the agents
class DoNothingAgent(Agent):
    def getSolution(self, state, maxIterations):
        if maxIterations == -1:     # RIP your machine if you remove this block
            return []

        #make idle action set
        nothActionSet = []
        for i in range(20):
            nothActionSet.append({"x":0,"y":0})

        return nothActionSet

# Random Agent code - completes random actions
class RandomAgent(Agent):
    def getSolution(self, state, maxIterations):

        #make random action set
        randActionSet = []
        for i in range(20):
            randActionSet.append(random.choice(directions))

        return randActionSet




#####    ASSIGNMENT 1 AGENTS    #####


# BFS Agent code
class BFSAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        intializeDeadlocks(state)
        iterations = 0
        bestNode = None
        queue = [Node(state.clone(), None, None)]
        visited = []
        #expand the tree until the iterations runs out or a solution sequence is found
        while (iterations < maxIterations or maxIterations <= 0) and len(queue) > 0:
            iterations += 1

            # YOUR CODE HERE
            x=queue.pop(0)
            if not bestNode:
                bestNode=x
                x_heuristic=x.getHeuristic()
                depth=x.getCost()
            x_heuristic_check=x.getHeuristic()
            if x_heuristic_check<x_heuristic:
                x_heuristic=x_heuristic_check
                bestNode=x
            if x_heuristic_check==x_heuristic:
                x_heuristic=x_heuristic_check
                new_depth=x.getCost()
                if new_depth<depth:
                    bestNode=x
            
                
            visited.append(x.getHash())
                
            if x.checkWin():
                return x.getActions()
                
            
            
            for i in (x.getChildren()):
                hash_of_child=i.getHash()
                if hash_of_child not in visited:
                    queue.append(i)
                    visited.append(hash_of_child)
        return bestNode.getActions()   #uncomment me



# DFS Agent Code
class DFSAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        intializeDeadlocks(state)
        iterations = 0
        bestNode = None
        queue = [Node(state.clone(), None, None)]
        visited = []
        
        #expand the tree until the iterations runs out or a solution sequence is found
        while (iterations < maxIterations or maxIterations <= 0) and len(queue) > 0:
            iterations += 1

            

            # YOUR CODE HERE
            x=queue[-1]
            del queue[-1]
            
            if not bestNode:
                bestNode=x
                x_heuristic=x.getHeuristic()
                depth=x.getCost()
    
            x_heuristic_check=x.getHeuristic()
            if x_heuristic_check<x_heuristic:
                x_heuristic=x_heuristic_check
                bestNode=x
            if x_heuristic_check==x_heuristic:
                x_heuristic=x_heuristic_check
                new_depth=x.getCost()
                if new_depth<depth:
                    bestNode=x
            if x.checkWin():
                return x.getActions()
            
            visited.append(x.getHash())
            
            queue.extend(child for child in x.getChildren()
                        if child.getHash() not in visited and child not in queue)

        return bestNode.getActions()



# AStar Agent Code
class AStarAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        #setup
        intializeDeadlocks(state)
        iterations = 0
        bestNode = None

        #initialize priority queue
        queue = PriorityQueue()
        queue.put(Node(state.clone(), None, None))
        visited = []

        while (iterations < maxIterations or maxIterations <= 0) and queue.qsize() > 0:
            iterations += 1

            ## YOUR CODE HERE ##

            x=queue.get()
            visited.append(x.getHash())
            
            
            if x.checkWin():
                return x.getActions()
                
            
            
            for i in (x.getChildren()):
                hash_of_child=i.getHash()
                
                if hash_of_child not in visited:
                    f_n=i.getHeuristic()+i.getCost()
                    queue.put(i,f_n)
                    visited.append(hash_of_child)

        return bestNode.getActions()


#####    ASSIGNMENT 2 AGENTS    #####


# Hill Climber Agent code
class HillClimberAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        #setup
        intializeDeadlocks(state)
        iterations = 0
        
        seqLen = 50            # maximum length of the sequences generated
        coinFlip = 0.5          # chance to mutate

        #initialize the first sequence (random movements)
        bestSeq = []
        mod_state = state.clone()
        
        bestnode=Node(mod_state, None, None)
        
        for i in range(seqLen):
            bestSeq.append(random.choice(directions))
            
        for s in bestSeq: 
            mod_state.update(s['x'],s['y'])
            
        hrstc = getHeuristic(mod_state) # CURRENT HEURISTIC = MAIN OR GLOBAL HEURISTIC (WILL COMPARE HEURISTIC AFTER EACH STEP)

        #mutate the best sequence until the iterations runs out or a solution sequence is found
        while (iterations < maxIterations):
            iterations += 1
            
            ## YOUR CODE HERE ##
            curr_seq=bestSeq.copy()
            current_state=mod_state.clone()
            
            
            # UPDATING THE CURRENT SEQUENCE AND ALSO THE CURRENT STATE
            for s in range(len(curr_seq)):
                if random.random() < coinFlip:
                    #mutate 
                    curr_seq[s]=(random.choice(directions))
                    
            current_state=state.clone()
            
            for i in curr_seq:
                current_state.update(i['x'], i['y'])
            #####################################################################
            

            
            #Creating a node to resolve same heuristic clash
            curr_node=Node(current_state.clone(), None, None)
            
            if current_state.checkWin():  # IF WIN RETURN CURRENT SEQUENCE 
                return curr_seq
            
            if getHeuristic(current_state)<hrstc:   # MEANING CURRENT STATE IS BETTER 
                #updating best sequence to current sequence
                bestSeq=curr_seq.copy()
                hrstc=getHeuristic(current_state)  # UPDATE GLOBAL HEURISTIC
                
            # if there is a clash
            if getHeuristic(current_state)==hrstc:
                if curr_node.getCost()>bestnode.getCost():
                    #do nothing, keep the best sequence as it is
                    pass
                else:
                    bestSeq=curr_seq.copy() # UPDATE THE BEST SEQ TO CURRENT SEQ
                    hrstc=getHeuristic(current_state)  # ALSO UPDATE GLOBAL HEURISTIC
                
        #return the best sequence found
        return bestSeq  



# Genetic Algorithm code
class GeneticAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        #setup
        intializeDeadlocks(state)

        iterations = 0
        seqLen = 50             # maximum length of the sequences generated
        popSize = 10            # size of the population to sample from
        parentRand = 0.5        # chance to select action from parent 1 (50/50)
        mutRand = 0.3           # chance to mutate offspring action

        bestSeq = []            #best sequence to use in case iterations max out

        #initialize the population with sequences of POP_SIZE actions (random movements)
        population = []
        for p in range(popSize):
            bestSeq = []
            for i in range(seqLen):
                bestSeq.append(random.choice(directions))
            population.append(bestSeq)
        

        #mutate until the iterations runs out or a solution sequence is found
        while (iterations < maxIterations):
            iterations += 1
            #print(population)
            #1. evaluate the population
            l=[]
            for i in population:
                Current_state=state.clone()
                for s in i: 
                    Current_state.update(s['x'],s['y'])
                    #print(getHeuristic(Current_state))
                
                l.append( list((getHeuristic(Current_state),i)) )

            #2. sort the population by fitness (low to high)
            sorted_pop=sorted(l, key = lambda i: i[0])
            

            #2.1 save bestSeq from best evaluated sequence
            
            
            bestSeq=sorted_pop[0][-1]   #IF NO SOLUTION IS FOUND, THIS ESSENTIALLY IS THE BEST SEQUENCE RETURNED
            Current_state=state.clone()
            #UPDATING THE CURRENT STATE
            for s in bestSeq: 
                Current_state.update(s['x'],s['y'])
                
            #CHECKING FOR WIN WITH THIS BEST SEQ
            if Current_state.checkWin():
                return bestSeq
            


            #3. generate probabilities for parent selection based on fitness
            weight=[]
            length=int(len(sorted_pop)/2)
            sample_space=sum(list(range(1,length+1)))
            
            for i in range((length),0,-1):
                weight.append(i/sample_space)
            
            
            #4. populate by crossover and mutation
            new_pop = []
            for i in range(int(popSize/2)):
                #4.1 select 2 parents sequences based on probabilities generated
                list_of_pop_to_be_mutated=random.choices(sorted_pop[:5],weights=weight,k=2)
                
                par1 = list_of_pop_to_be_mutated[0][1]
                par2 = list_of_pop_to_be_mutated[-1][1]



                #4.2 make a child from the crossover of the two parent sequences
                offspring = []
                for i in range(len(par1)):
                    if random.random()< parentRand:
                        offspring.append(par1[i])
                    else:
                        offspring.append(par2[i])
                

                



                #4.3 mutate the child's actions
                offspring_copy=offspring.copy()
                for K in range(len(offspring)):
                    if random.random() < mutRand:
                    #mutate 
                        offspring_copy[K]=(random.choice(directions))


                #4.4 add the child to the new population
                new_pop.append(list(offspring_copy))


            #5. add top half from last population (mu + lambda)
            for i in range(int(popSize/2)):
                new_pop.append(sorted_pop[i][-1])   #APPENDING PREVIOUS 5 PARENTS TO 5 NEW OFFSPRINGS
                #break           #remove me


            #6. replace the old population with the new one
            population = list(new_pop)

        #return the best found sequence 
        return bestSeq


# MCTS Specific node to keep track of rollout and score
class MCTSNode(Node):
    def __init__(self, state, parent, action, maxDist):
        super().__init__(state,parent,action)
        self.children = []  #keep track of child nodes
        self.n = 0          #visits
        self.q = 0          #score
        self.maxDist = maxDist      #starting distance from the goal (heurstic score of initNode)

    #update get children for the MCTS
    def getChildren(self,visited):
        #if the children have already been made use them
        if(len(self.children) > 0):
            return self.children

        children = []

        #check every possible movement direction to create another child
        for d in directions:
            childState = self.state.clone()
            crateMove = childState.update(d["x"], d["y"])

            #if the node is the same spot as the parent, skip
            if childState.player["x"] == self.state.player["x"] and childState.player["y"] == self.state.player["y"]:
                continue

            #if this node causes the game to be unsolvable (i.e. putting crate in a corner), skip
            if crateMove and checkDeadlock(childState):
                continue

            #if this node has already been visited (same placement of player and crates as another seen node), skip
            if getHash(childState) in visited:
                continue

            #otherwise add the node as a child
            children.append(MCTSNode(childState, self, d, self.maxDist))

        self.children = list(children)    #save node children to generated child

        return children

    #calculates the score the distance from the starting point to the ending point (closer = better = larger number)
    def calcEvalScore(self,state):
        return self.maxDist - getHeuristic(state)

    #compares the score of 2 mcts nodes
    def __lt__(self, other):
        return self.q < other.q

    #print the score, node depth, and actions leading to it
    #for use with debugging
    def __str__(self):
        return str(self.q) + ", " + str(self.n) + ' - ' + str(self.getActions())


# Monte Carlo Tree Search Algorithm code
class MCTSAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        #setup
        intializeDeadlocks(state)
        iterations = 0
        bestNode = None
        initNode = MCTSNode(state.clone(), None, None, getHeuristic(state))

        while(iterations < maxIterations):
            #print("\n\n---------------- ITERATION " + str(iterations+1) + " ----------------------\n\n")
            iterations += 1

            #mcts algorithm
            rollNode = self.treePolicy(initNode)
            score = self.rollout(rollNode)
            self.backpropogation(rollNode, score)

            #if in a win state, return the sequence
            if(rollNode.checkWin()):
                return rollNode.getActions()

            #set current best node
            bestNode = self.bestChildUCT(initNode)

            #if in a win state, return the sequence
            if(bestNode and bestNode.checkWin()):
                return bestNode.getActions()


        #return solution of highest scoring descendent for best node
        #if this line was reached, that means the iterations timed out before a solution was found
        return self.bestActions(bestNode)
        

    #returns the descendent with the best action sequence based
    def bestActions(self, node):
        #no node given - return nothing
        if node == None:
            return []

        bestActionSeq = []
        while(len(node.children) > 0):
            node = self.bestChildUCT(node)

        return node.getActions()


    ####  MCTS SPECIFIC FUNCTIONS BELOW  ####

    #determines which node to expand next
    def treePolicy(self, rootNode):
        current_Node = rootNode
        visited = []

        ## YOUR CODE HERE ##
        while(not current_Node.checkWin()):
            children = current_Node.getChildren(visited)
            for child in children:
                if child.n == 0:
                    return child

            current_Node = self.bestChildUCT(current_Node)

        return current_Node




    # uses the exploitation/exploration algorithm
    def bestChildUCT(self, node):
        c = 1               #c value in the exploration/exploitation equation
        
        bestChild = None

        ## YOUR CODE HERE ##
        
        childrens = node.getChildren([])

        child_with_uct = []
        for child in childrens:
            if child.n != 0:
                expoitation=child.q/child.n
                exploration=math.sqrt((2*(math.log(node.n,2))/child.n))
                uct = expoitation + exploration
                child_with_uct.append((uct,child)) # [(uct value 1,child 1),(uct value 2,child 2) and so on..]

        #RETURNING BEST CHILD BASED ON UCT
        if child_with_uct:
            best_child_with_uct = max(child_with_uct, key=lambda i: i[0])
            bestChild=best_child_with_uct[-1]

        return bestChild



     #simulates a score based on random actions taken
    def rollout(self,node):
        numRolls = 7        #number of times to rollout to

        ## YOUR CODE HERE ##
        curr_state = node.state.clone()
        
        for i in range(numRolls):
            if curr_state.checkWin():
                return node.calcEvalScore(curr_state)

            random_move = random.choice(directions)
            curr_state.update(random_move['x'],random_move['y'])
            
        return node.calcEvalScore(curr_state)



     #updates the score all the way up to the root node
    def backpropogation(self, node, score):
        
        ## YOUR CODE HERE ##
        if not node:
            return
        else:
            node.n += 1
            node.q += score
            self.backpropogation(node.parent, score)    #RECURSIVELY CALLING UNTIL ROOT NODE
                                                        # AND UPDATING N AND SCORE VALUES
        
        
        

        

