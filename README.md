# Graph-Neural-Network

A Graph Neural Network (GNN) is a type of neural network specifically designed to work with graph-structured data. The basic form of a GNN typically involves three key steps: message passing (or aggregation), update, and readout.

### Basic Form of a Graph Neural Network

1. **Message Passing (Aggregation):**
   - **Input:** Graph \( G = (V, E) \) where \( V \) is a set of nodes and \( E \) is a set of edges.
   - For each node \( v \) in the graph, aggregate information from its neighbors \( N(v) \). This can be done using various aggregation functions such as sum, mean, max, etc.
   - **Aggregation Function:** 
     \[
     m_v^{(k)} = \text{AGGREGATE}^{(k)} \left( \{ h_u^{(k-1)} : u \in N(v) \} \right)
     \]
   - \( h_u^{(k-1)} \) represents the feature vector of node \( u \) at layer \( k-1 \), and \( m_v^{(k)} \) is the aggregated message for node \( v \) at layer \( k \).

2. **Update:**
   - Update the node's feature vector using the aggregated message and the node's current state.
   - **Update Function:**
     \[
     h_v^{(k)} = \text{UPDATE}^{(k)} \left( h_v^{(k-1)}, m_v^{(k)} \right)
     \]
   - This is typically done using a neural network layer, such as a fully connected layer followed by a non-linear activation function.

3. **Readout:**
   - After several rounds of message passing and updating (typically \( K \) layers), a readout function is applied to obtain the final representation or prediction.
   - **Readout Function:**
     \[
     \hat{y} = \text{READOUT} \left( \{ h_v^{(K)} : v \in V \} \right)
     \]
   - The readout function can be as simple as a sum or average of node features, or a more complex pooling mechanism depending on the task (e.g., node classification, graph classification, link prediction).

### Example Architecture

For example, a simple GNN architecture might look like this:

1. **Input Layer:** 
   - Each node is initially represented by a feature vector \( h_v^{(0)} \).

2. **Hidden Layers:** 
   - Apply message passing and update steps for \( K \) layers.
     \[
     h_v^{(k)} = \sigma \left( W^{(k)} \cdot \text{AGGREGATE} \left( \{ h_u^{(k-1)} : u \in N(v) \} \right) \right)
     \]
   - Here, \( W^{(k)} \) are learnable weight matrices and \( \sigma \) is a non-linear activation function like ReLU.

3. **Output Layer:**
   - Apply a readout function to get the final graph-level or node-level output.

### Example Pseudocode
```python
for k in range(K):
    for v in V:
        m_v = aggregate([h_u for u in neighbors(v)])
        h_v = update(h_v, m_v)
output = readout([h_v for v in V])
```

### Summary

- **Message Passing:** Aggregate information from neighbors.
- **Update:** Update node features using aggregated messages.
- **Readout:** Compute final output from node features.
