//! A directed graph of nodes connected publisher-to-subscriber, with its roots
//! and leaves precomputed. The dataflow's operator graph is one; structures
//! that must mirror an operator graph's layout, such as the memory context's
//! per-dataflow eviction tree, build another over the same edges.

use std::ops::ControlFlow;

/// Nodes with zero or more downstream subscribers (`edges`) and zero or more
/// upstream publishers (`back_edges`). Several subscribers means one node's
/// output is read in full by each of them, as a CTE's is.
///
/// Two traversal orders are offered:
/// - **Backwards** (leaf-to-root): downstream nodes first.
/// - **Forwards** (root-to-leaf): upstream nodes first.
///
/// Both are depth-first walks. A node read by several subscribers is reached
/// once through each of them, so the closure runs on it that many times.
pub(crate) struct Graph<N> {
    nodes: Vec<N>,
    leafs: Vec<usize>,
    roots: Vec<usize>,
    /// Subscriber indices for each node; empty for a leaf.
    edges: Vec<Vec<usize>>,
    /// Publisher indices for each node; empty for a root.
    back_edges: Vec<Vec<usize>>,
}

impl<N> Graph<N> {
    /// Connect `nodes` by `publisher_to_subscriber` pairs of indices into it.
    pub(crate) fn from_edges(nodes: Vec<N>, publisher_to_subscriber: &[(usize, usize)]) -> Self {
        let n = nodes.len();
        let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut back_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(publisher, subscriber) in publisher_to_subscriber {
            edges[publisher].push(subscriber);
            back_edges[subscriber].push(publisher);
        }
        let roots = (0..n).filter(|&i| back_edges[i].is_empty()).collect();
        let leafs = (0..n).filter(|&i| edges[i].is_empty()).collect();
        Self {
            nodes,
            leafs,
            roots,
            edges,
            back_edges,
        }
    }

    pub(crate) fn nodes(&self) -> &[N] {
        &self.nodes
    }

    pub(crate) fn node(&self, idx: usize) -> &N {
        &self.nodes[idx]
    }

    pub(crate) fn node_mut(&mut self, idx: usize) -> &mut N {
        &mut self.nodes[idx]
    }

    pub(crate) fn roots(&self) -> &[usize] {
        &self.roots
    }

    /// The nodes reading `idx`'s output.
    pub(crate) fn edges(&self, idx: usize) -> &[usize] {
        &self.edges[idx]
    }

    /// The nodes whose output `idx` reads.
    pub(crate) fn back_edges(&self, idx: usize) -> &[usize] {
        &self.back_edges[idx]
    }

    /// Walk leaf-to-root via back-edges, running `f` on every node reached and
    /// stopping at the first `Break`.
    pub(crate) fn traverse_backwards<T, E>(
        &mut self,
        mut f: impl FnMut(usize, &mut N) -> Result<ControlFlow<T>, E>,
    ) -> Result<ControlFlow<T>, E> {
        let mut stack: Vec<usize> = self.leafs.clone();
        while let Some(idx) = stack.pop() {
            let result = f(idx, &mut self.nodes[idx])?;
            if matches!(result, ControlFlow::Break(..)) {
                return Ok(result);
            }
            stack.extend(self.back_edges[idx].iter().copied());
        }
        Ok(ControlFlow::Continue(()))
    }

    /// The root-to-leaf twin of [`traverse_backwards`](Self::traverse_backwards).
    pub(crate) fn traverse_forwards<T, E>(
        &mut self,
        mut f: impl FnMut(usize, &mut N) -> Result<ControlFlow<T>, E>,
    ) -> Result<ControlFlow<T>, E> {
        let mut stack: Vec<usize> = self.roots.clone();
        while let Some(idx) = stack.pop() {
            let result = f(idx, &mut self.nodes[idx])?;
            if matches!(result, ControlFlow::Break(..)) {
                return Ok(result);
            }
            stack.extend(self.edges[idx].iter().copied());
        }
        Ok(ControlFlow::Continue(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    fn visit_order(graph: &mut Graph<&'static str>, backwards: bool) -> Vec<&'static str> {
        let mut seen = Vec::new();
        let walk = |_: usize, node: &mut &'static str| {
            seen.push(*node);
            Ok::<ControlFlow<()>, Infallible>(ControlFlow::Continue(()))
        };
        let Ok(_) = if backwards {
            graph.traverse_backwards(walk)
        } else {
            graph.traverse_forwards(walk)
        };
        seen
    }

    #[test]
    fn backwards_walk_starts_at_the_leaf() {
        let mut graph = Graph::from_edges(vec!["scan", "filter", "output"], &[(0, 1), (1, 2)]);

        let order = visit_order(&mut graph, true);

        assert_eq!(order, ["output", "filter", "scan"]);
    }

    #[test]
    fn forwards_walk_starts_at_the_root() {
        let mut graph = Graph::from_edges(vec!["scan", "filter", "output"], &[(0, 1), (1, 2)]);

        let order = visit_order(&mut graph, false);

        assert_eq!(order, ["scan", "filter", "output"]);
    }

    #[test]
    fn break_stops_the_walk() {
        let mut graph = Graph::from_edges(vec![1, 2, 3], &[(0, 1), (1, 2)]);

        let Ok(found) = graph.traverse_backwards(|_, node| {
            Ok::<_, Infallible>(if *node == 2 {
                ControlFlow::Break(*node)
            } else {
                ControlFlow::Continue(())
            })
        });

        assert_eq!(found, ControlFlow::Break(2));
    }

    #[test]
    fn disconnected_roots_are_all_walked() {
        let mut graph = Graph::from_edges(vec!["a", "b", "c"], &[(0, 2)]);

        let order = visit_order(&mut graph, false);

        assert_eq!(graph.roots(), [0, 1]);
        assert_eq!(order.len(), 3);
    }
}
