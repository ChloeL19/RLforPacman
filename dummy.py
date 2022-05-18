# a function to sort a list of integers
def sort(lst):
    """
    Sort a list of integers
    """
    for i in range(len(lst)):
        for j in range(len(lst) - 1):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
    return lst

# return 1 with 0.5 probability
def chooseOne(prob):
    """
    Return 1 with probability prob
    """
    if random.random() < prob:
        return 1
    else:
        return 0

# import pytorch
import torch

# find the MST of a graph
def findMST(graph):
    # implement Prim's algorithm
    # initialize the MST
    MST = []
    # initialize the set of vertices
    V = set(range(len(graph)))
    # initialize the set of vertices in the MST
    M = set()
    # initialize the set of vertices not in the MST
    notM = V - M
    # initialize the set of vertices with the smallest edge
    S = {0}
    # initialize the set of vertices with the second smallest edge
    notS = V - S
    # initialize the smallest edge
    smallest = graph[0][1]
    # initialize the second smallest edge
    secondSmallest = graph[0][2]
    # while the MST is not complete
    while len(M) < len(graph):
        # for each vertex in the set of vertices not in the MST
        for v in notM:
            # if the edge from the smallest vertex to the current vertex is smaller than the smallest edge
            if graph[S.pop()][v] < smallest:
                # set the second smallest edge to the smallest edge
                secondSmallest = smallest
                # set the smallest edge to the edge from the smallest vertex to the current vertex
                smallest = graph[S.pop()][v]
                # set the second smallest vertex to the current vertex
                notS.add(v)
                # set the smallest vertex to the current vertex
                S.add(v)
            # if the edge from the second smallest vertex to the current vertex is smaller than the second smallest edge
            elif graph[S.pop()][v] < secondSmallest:
                # set the second smallest edge to the edge from the second smallest vertex to the current vertex
                secondSmallest = graph[S.pop()][v]
                # set the second smallest vertex to the current vertex
                notS.add(v)
                # set the smallest vertex to the current vertex
                S.add(v)
        # add the smallest edge to the MST
        MST.append(smallest)
        # add the second smallest edge to the MST
        MST.append(secondSmallest)
        # add the smallest vertex to the MST
        M.add(S.pop())
        # add the second smallest vertex to the MST
        M.add(notS.pop())
        # set the smallest edge to infinity
        smallest = float('inf')
        # set the second smallest edge to infinity
        secondSmallest = float('inf')

# create a unit test for findMST
def testFindMST():
    # create a graph
    graph = [[0, 1, 1], [0, 2, 1], [1, 2, 1], [1, 3, 1], [2, 3, 1]]
    # find the MST of the graph
    MST = findMST(graph)
    # check if the MST is correct
    if MST == [0, 1, 2, 3]:
        return True
    else:
        return False


        

