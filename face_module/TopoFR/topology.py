'''
Methods for calculating lower-dimensional persistent homology.
'''

import numpy as np


class UnionFind:
    '''
    An implementation of a Union--Find class. The class performs path
    compression by default. It uses integers for storing one disjoint
    set, assuming that vertices are zero-indexed.
    '''

    def __init__(self, n_vertices):
        '''
        Initializes an empty Union--Find data structure for a given
        number of vertices.
        '''

        self._parent = np.arange(n_vertices, dtype=int)

    def find(self, u):
        '''
        Finds and returns the parent of u with respect to the hierarchy.
        '''

        if self._parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        '''
        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.
        '''

        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        '''
        Generator expression for returning roots, i.e. components that
        are their own parents.
        '''

        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex


class PersistentHomologyCalculation:
    def __call__(self, matrix):

        n_vertices = matrix.shape[0]
        uf = UnionFind(n_vertices)

        triu_indices = np.triu_indices_from(matrix)
        edge_weights = matrix[triu_indices]
        edge_indices = np.argsort(edge_weights, kind='stable')

        # 1st dimension: 'source' vertex index of edge
        # 2nd dimension: 'target' vertex index of edge
        persistence_pairs = []

        for edge_index, edge_weight in \
                zip(edge_indices, edge_weights[edge_indices]):

            u = triu_indices[0][edge_index]
            v = triu_indices[1][edge_index]

            younger_component = uf.find(u)
            older_component = uf.find(v)

            # Not an edge of the MST, so skip it
            if younger_component == older_component:
                continue
            elif younger_component > older_component:
                uf.merge(v, u)
            else:
                uf.merge(u, v)

            if u < v:
                persistence_pairs.append((u, v))
            else:
                persistence_pairs.append((v, u))

        # Return empty cycles component
        return np.array(persistence_pairs), np.array([])


class AlephPersistenHomologyCalculation():
    def __init__(self, compute_cycles, sort_selected):
        """Calculate persistent homology using aleph.

        Args:
            compute_cycles: Whether to compute cycles
            sort_selected: Whether to sort the selected pairs using the
                distance matrix (such that they are in the order of the
                filteration)
        """
        self.compute_cycles = compute_cycles
        self.sort_selected = sort_selected

    def __call__(self, distance_matrix):
        """Do PH calculation.

        Args:
            distance_matrix: numpy array of distances

        Returns: tuple(edge_featues, cycle_features)
        """
        import aleph
        if self.compute_cycles:
            pairs_0, pairs_1 = aleph.vietoris_rips_from_matrix_2d(
                distance_matrix)
            pairs_0 = np.array(pairs_0)
            pairs_1 = np.array(pairs_1)
        else:
            pairs_0 = aleph.vietoris_rips_from_matrix_1d(
                distance_matrix)
            pairs_0 = np.array(pairs_0)
            # Return empty cycles component
            pairs_1 = np.array([])

        if self.sort_selected:
            selected_distances = \
                distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]
            indices_0 = np.argsort(selected_distances)
            pairs_0 = pairs_0[indices_0]
            if self.compute_cycles:
                cycle_creation_times = \
                    distance_matrix[(pairs_1[:, 0], pairs_1[:, 1])]
                cycle_destruction_times = \
                    distance_matrix[(pairs_1[:, 2], pairs_1[:, 3])]
                cycle_persistences = \
                    cycle_destruction_times - cycle_creation_times
                # First sort by destruction time and then by persistence of the
                # create cycles in order to recover original filtration order.
                indices_1 = np.lexsort(
                    (cycle_destruction_times, cycle_persistences))
                pairs_1 = pairs_1[indices_1]

        return pairs_0, pairs_1
