"""Implementation of Monte Carlo Tree Search algorithm.

This implementation allows "parallel" multi-root trees.  This does not
enable parallel computation, but reduces significantly the python
overhead thanks to numpy optimized operations.

The search can be configured by providing custom transition and
evaluation functions.

"""

import numpy as np


class MCTS:
    """A multi-rooted tree grown with the Monte Carlo Tree Search algorithm.

    Attributes:
        roots (int): number of roots in the tree
        nodes (int): number of nodes
        height (int): height of the tree
        actions (int): number of actions
        exploration (float): exploration coefficient
        x (ndarray[object]): states
        Q (ndarray[float]): matrix of state-action values
        N (ndarray[int]): matrix of visit counters (nodes, actions)
        C (ndarray[int]): matrix of indices of children (nodes, actions)  (-1 if missing)
        T (ndarray[bool]): flags for terminal nodes
        R (ndarray[float]): returns of nodes
        pi (ndarray[float]): prior probabilities (nodes, actions)

    """

    def __init__(self, root_states, actions, transition_function,
                 evaluation_function, exploration=1, initial_q=0.0):
        """Create and initialize the tree.

        The tree will be initialized with one root for each initial state.

        Args:
            root_states (iterable): initial states
            actions (int): number of actions in each non-terminal node
            transition_function (callable): function computing new states
            evaluation_function (callable): function computing info from states
            exploration (float): exploration coefficient
            initial_q (float): value for non-visited nodes
        
        The transition function will be called with an array of states
        and one of actions.  The result must be the array of states
        obtained by performing the actions in the given states.

        The evaluation function will be called with an array of
        states.  The result must be a triplet of arrays.  The first
        tells which states are terminal; the second provides the
        return for the states; the third is a matrix containing the
        prior probabilities for the states.

        The node states and are not directly processed by this class
        and can be objects of any kind.

        """
        root_states = list(root_states)
        roots = len(root_states)
        self.roots = roots
        self.nodes = roots
        self.height = 1
        self.actions = actions
        self.exploration = exploration
        self.initial_q = initial_q
        self.transition_function = transition_function
        self.evaluation_function = evaluation_function
        self.x = np.empty(roots, dtype=object)
        self.x[:] = root_states
        self.Q = np.full((roots, actions), initial_q, dtype=np.float32)
        self.N = np.zeros((roots, actions), dtype=np.int32)
        self.C = np.full((roots, actions), -1, dtype=np.int32)
        T, R, pi = evaluation_function(self.x)
        self.T = np.array(T, dtype=bool)
        self.R = np.array(R, dtype=np.float32)
        self.pi = np.array(pi, dtype=np.float32)

    def select_by_policy(self, policy):
        """Select leaves in the tree.

        Descend the trees starting from the roots, by applying the
        given policy function until final (terminal or non expanded)
        nodes are found.

        Args:
            policy (callable): the policy used to select the nodes

        Returns:
            paths (ndarray[int]): matrix with indices of the nodes visited
            actions (ndarray[int]): matrix with the actions taken at each step
            lenghts (ndarray[int]): array with the lengths of the paths

        The policy will be called with an array of indices of nodes.
        It must return an array of actions.

        paths[i, j] is the j-th node visited starting from the i-th
        root and actions[i, j] is the action taken from it.  paths[i,
        lengths[i] - 1] is the final node reached from the i-th root.

        actions[i, j] if the action that is taken from path[i, j] to
        reach path[i, j + 1].  actions[i, lengths[i] - 1] is the
        action that would be taken from the last node in the path
        according to he policy and is used by MCTS to decide how to
        expand the leaf.

        """
        idx = np.arange(self.roots)
        active = np.arange(self.roots)
        paths = np.full((self.roots, self.height), -1, dtype=int)
        actions = np.full((self.roots, self.height), -1, dtype=int)
        lengths = np.zeros(self.roots, dtype=int)
        t = 0
        while active.size > 0:
            if 0 and idx[0] == 2 and self.N[idx].sum() == 7:
                breakpoint()
            a = policy(idx)
            paths[active, t] = idx
            actions[active, t] = a
            lengths[active] += 1
            idx = self.C[idx, a]
            sub = (idx >= 0)
            idx = idx[sub]
            active = active[sub]
            t += 1
        return paths, actions, lengths

    def _ucb_policy(self, nodes):
        frac = np.sqrt(np.maximum(1e-8, self.N[nodes].sum(1, keepdims=True))) / (self.N[nodes] + 1)
        ucb = self.Q[nodes] + self.exploration * self.pi[nodes] * frac
        return ucb.argmax(1)

    def select(self):
        """Select the nodes by maximizing the UCB."""
        return self.select_by_policy(self._ucb_policy)

    def expand(self, leaves, actions, lengths):
        """Expand the given leaves by applying the actions.

        Args:
            leaves (ndarray[int]): indices of the nodes to expand
            actions (ndarray[int]): actions to be performed at the leaves
            lengths (ndarray[int]): length of the paths obtained to reach the leaves

        Returns:
            The indices of the new nodes added to the tree.

        Note:
            Leaves that are terminal are not expanded.  In that case
            the index of the leaf itself is returned.

        """
        nodes = np.empty(leaves.size, dtype=int)
        term = self.T[leaves]
        nodes[term] = leaves[term]
        non_terminal = leaves[~term]
        if non_terminal.size > 0:
            assert self.nodes + non_terminal.size <= self.x.size
            new_nodes = np.arange(self.nodes, self.nodes + non_terminal.size)
            nodes[~term] = new_nodes
            self.C[non_terminal, actions[~term]] = new_nodes
            new_states = self.transition_function(self.x[non_terminal], actions[~term])
            new_T, new_R, new_pi = self.evaluation_function(new_states)
            self.x[new_nodes] = new_states
            self.T[new_nodes] = new_T
            self.R[new_nodes] = new_R
            self.pi[new_nodes, :] = new_pi
            self.nodes += new_nodes.size
            self.height = max(self.height, lengths[~term].max() + 1)
        return nodes

    def evaluate(self, nodes):
        """Evaluate the nodes.

        Args:
            nodes (ndarray[int]): indices of the nodes to evaluate

        Returns:
            An array with the evaluations.
        """
        return self.R[nodes]

    def backup(self, paths, actions, depths, values):
        """Update the values and the counters in the tree.

        Args:
            paths (ndarray[int]): indices of the nodes visited
            actions (ndarray[int]): actions performed at the nodes
            depths (ndarray[int]): length of the paths from each root
            values (ndarray[int]): evaluation of the leaves (one for each path)
        """
        p = np.concatenate([paths[i, :d] for i, d in enumerate(depths)])
        a = np.concatenate([actions[i, :d] for i, d in enumerate(depths)])
        v = np.concatenate([[v] * d for v, d in zip(values, depths)])
        self.Q[p, a] = (self.Q[p, a] * self.N[p, a] + v) / (self.N[p, a] + 1)
        self.N[p, a] += 1

    def _grow_array(self, a, n, val):
        b = np.full((n, *a.shape[1:]), val, dtype=a.dtype)
        return np.append(a, b, 0)

    def grow(self, steps):
        """Grow the tree with additional nodes.

        Args:
            steps (int): number of nodes to add under each root
        """
        k = steps * self.roots
        self.Q = self._grow_array(self.Q, k, self.initial_q)
        self.N = self._grow_array(self.N, k, 0)
        self.pi = self._grow_array(self.pi, k, 1 / self.actions)
        self.C = self._grow_array(self.C, k, -1)
        self.T = self._grow_array(self.T, k, False)
        self.R = self._grow_array(self.R, k, np.nan)
        self.x = self._grow_array(self.x, k, None)
        ran = np.arange(self.roots)
        for step in range(steps):
            paths, actions, depths = self.select()
            nodes = self.expand(paths[ran, depths - 1], actions[ran, depths - 1], depths)
            values = self.evaluate(nodes)
            self.backup(paths, actions, depths, values)
        if self.nodes == self.x.size:
            return
        self.Q = self.Q[:self.nodes, :]
        self.N = self.N[:self.nodes, :]
        self.pi = self.pi[:self.nodes, :]
        self.C = self.C[:self.nodes, :]
        self.T = self.T[:self.nodes]
        self.R = self.R[:self.nodes]
        self.x = self.x[:self.nodes]

    def most_visited(self):
        """Select nodes by choosing those with the highest visit count."""
        return self.select_by_policy(lambda nodes: self.N[nodes].argmax(1))

    def _sample_policy(self, nodes, eps=1e-6):
        c = self.N[nodes] + eps
        p = c / c.sum(1, keepdims=True)
        a = [np.random.choice(self.actions, p=p[i]) for i in range(nodes.size)]
        return np.array(a)

    def sample_path(self):
        """Select nodes by randomly descending the trees."""
        return self.select_by_policy(self._sample_policy)

    def descend_tree(self, actions):
        """Move down each root by applying an action.

        Args:
            actions (ndarray): action to apply (one per root).

        Note that roots that are also terminal are not changed.
        """
        new_roots = self.C[np.arange(self.roots), actions]
        active = (new_roots >= 0)
        new_roots[~active] = np.arange(self.roots)[~active]
        descendants = [new_roots]
        active = new_roots[active]
        while active.size:
            active = self.C[active, :].flatten()
            active = active[active >= 0]
            descendants.append(active)
        new_to_old = np.concatenate(descendants)
        self.nodes = new_to_old.size
        self.height = len(descendants) - (descendants[-1].size == 0)
        old_to_new = np.full(self.x.size + 1, -1)
        old_to_new[new_to_old] = np.arange(self.nodes)
        self.x = self.x[new_to_old]
        self.Q = self.Q[new_to_old, :]
        self.N = self.N[new_to_old, :]
        self.C = old_to_new[self.C[new_to_old, :]]
        self.T = self.T[new_to_old]
        self.R = self.R[new_to_old]
        self.pi = self.pi[new_to_old, :]

    def dump(self, filename, root=None):
        """Dump the content of the tree to a dot file.

        Args:
            filename (str): name of the output file
            root (int or None): root of the subtree to dump (or the full tree when None)
        """
        fmtnode = ('  N{} [shape=rect, color="{}", ' +
                   'label="N{}\\n#{:d} / q{:.3f} / r{:.3f}"]')
        fmtedge = '  N{} -- N{} [label="{} [{:.3f}]"]'
        with open(filename, "w") as f:
            print("graph {", file=f)
            print('  rankdir="LR"', file=f)
            roots = (range(self.roots) if root is None else [root])
            for n in roots:
                tot = 1 + self.N[n, :].sum()
                avgq = (self.R[n] + self.Q[n, :] @ self.N[n, :]) / tot
                color = ("red" if self.T[n] else "blue")
                line = fmtnode.format(n, color, n, tot, avgq, self.R[n])
                print(line, file=f)
            nodes = (range(self.nodes) if root is None else self.subtree(root))
            for n in nodes:
                for a in range(self.N.shape[1]):
                    c = self.C[n, a]
                    if c < 0:
                        continue
                    color = ("red" if self.T[c] else "blue")
                    line = fmtnode.format(c, color, c, self.N[n, a],
                                          self.Q[n, a], self.R[c])
                    print(line, file=f)
                    line = fmtedge.format(n, c, a, self.pi[n, a])
                    print(line, file=f)
            print("}", file=f)

    def subtree(self, root_index):
        """Return the indices of nodes descending from root_index."""
        nodes = [root_index]
        subtree = []
        while nodes:
            n = nodes.pop()
            subtree.append(n)
            nodes.extend(c for c in self.C[n, :] if c >= 0)
        subtree.sort()
        return subtree


def _test():
    def evaluation(state):
        state = np.array(state)
        return state == 0, -np.abs(state), np.ones((state.size, 2)) / 2

    def transition(state, actions):
        return state + (2 * actions - 1)

    tree = MCTS(np.arange(-3, 4), 2, transition, evaluation)
    tree.grow(100)
    p, a, d = tree.most_visited()
    for aa, dd in zip(a, d):
        print(aa[:dd])
    print()
    for aa, dd in zip(p, d):
        print(aa[:dd])
    print(tree.nodes, tree.height)
    tree.descend_tree(a[:, 0])
    print(tree.nodes, tree.height)


if __name__ == "__main__":
    _test()
