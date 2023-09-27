import numpy as np

def Action(select_action, matrix, origin, x=116):
    """
    Disconnect a node from a graph represented by an adjacency matrix.

    Parameters:
    - select_action (int or str): The node index to be disconnected.
    - matrix (numpy.array or list): The adj matrix of the graph.
    - origin (numpy.array or list): The adj matrix for backup.
    - x (int, optional): The size of the matrix. Defaults to 116.

    Returns:
    - numpy.array: The updated adjacency matrix with the node disconnected.
    """

    # Ensure the action is an integer
    select_action = int(select_action) 

    # Convert matrix and origin to lists if they aren't already
    if type(matrix) != list:
        List = matrix.tolist()
        backup = origin.tolist()
    else:
        List = matrix
        backup = origin

    # Disconnect the selected node if it's not the default action
    if select_action != 116:
        List[select_action] = [0] * x   
        List = np.array(List).T.tolist()
        List[select_action] = [0] * x

    # Ensure diagonal is set to 1 (self-loop)
    for i in range(x): 
        List[i][i] = 1

    return np.array(List)

if __name__ == "__main__":
    # Sample call or any test case you'd like to add
    # Example:
    # matrix = np.array([[1, 1], [1, 1]])
    # result = Action(0, matrix, matrix)
    # print(result)
    pass
