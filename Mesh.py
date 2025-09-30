import torch
from itertools import cycle


class Mesh:

    def __init__(self,
                 nodes_coords: torch.Tensor,
                 elems_array: torch.Tensor = None,
                 elems_p: torch.Tensor = None,
                 boundary_nodes: torch.Tensor = None) -> None:
        self.nodes_coords = nodes_coords
        self.elems_array = elems_array
        self.elems_p = elems_p
        self.boundary_nodes = boundary_nodes
        
    def dimension(self):
        return self.nodes_coords.size()[1]
        
        
    def plot(self) -> None:
        """2D
        """
        import matplotlib.pyplot as plt
        
        dim = self.dimension()
        
        # MatPlotLib set up
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d') if dim == 3 else fig.add_subplot()
        
        # Scatter points
        ax.scatter(*[self.nodes_coords[:, i] for i in range(dim)])
        
        # Display elements
        if self.elems_p is None or dim == 3:
            plt.show()
            return
        for i in range(len(self.elems_p) - 1):
            start = self.elems_p[i]
            end = self.elems_p[i+1]
            nodes_in_elem = self.elems_array[start: end]
            node_coords_in_elem = self.nodes_coords[nodes_in_elem]
            for k in range(len(node_coords_in_elem)):
                node_coords = node_coords_in_elem[[k, (k+1)%len(node_coords_in_elem)]]
                ax.plot(*[node_coords[:, l] for l in range(dim)])

        plt.show()
        """
        nnodes = numpy.shape(node_coords)[0]
        nelems = numpy.shape(p_elem2nodes)[0]
        for elem in range(0, nelems-1):
            xyz = node_coords[ elem2nodes[p_elem2nodes[elem]:p_elem2nodes[elem+1]], :]
            if xyz.shape[0] == 3:
                matplotlib.pyplot.plot((xyz[0, 0], xyz[1, 0], xyz[2, 0], xyz[0, 0]),
                                    (xyz[0, 1], xyz[1, 1], xyz[2, 1], xyz[0, 1]), color=color)
            else:
                matplotlib.pyplot.plot((xyz[0, 0], xyz[1, 0], xyz[2, 0], xyz[3, 0], xyz[0, 0]),
                                    (xyz[0, 1], xyz[1, 1], xyz[2, 1], xyz[3, 1], xyz[0, 1]), color=color)
        """

        return
        
        
if __name__ == "__main__":
    node_coord = torch.tensor([[0, 0],
                               [1, 0],
                               [1, 1],
                               [0, 1],
                               [2, 0],
                               [2, 1],
                               [2, 2],])
    elem2node = torch.tensor([0, 1, 2, 3,
                              1, 4, 5, 2,
                              2, 5, 6], dtype=torch.int64)
    p_elem2node = torch.tensor([0, 4, 8, 11], dtype=torch.int64)
    mesh_2D = Mesh(nodes_coords=node_coord,
                   elems_array=elem2node,
                   elems_p=p_elem2node)
    mesh_2D.plot()
    
    node_coord = torch.Tensor([[0, 0, 0],
                                [0, 0, 1],
                                [0, 1, 0]])
    mesh_3D = Mesh(nodes_coords=node_coord)
    mesh_3D.plot()
    