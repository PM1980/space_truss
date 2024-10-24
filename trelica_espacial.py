import streamlit as st
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
import pandas as pd

@dataclass
class TrussData:
    coord: np.ndarray
    conec: np.ndarray
    E: float
    A: np.ndarray
    fixed_dofs: list
    loads: dict

def solve_truss(data):
    # Your existing solver logic here, converted from MATLAB
    # Returns displacements, reactions, and member forces
    nn = len(data.coord)
    ndof = nn * 3
    K = np.zeros((ndof, ndof))
    
    # Assembly K matrix
    for i in range(len(data.conec)):
        n1, n2 = data.conec[i]
        x1, y1, z1 = data.coord[n1-1]
        x2, y2, z2 = data.coord[n2-1]
        
        L = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        Cx = (x2-x1)/L
        Cy = (y2-y1)/L
        Cz = (z2-z1)/L
        
        T = np.array([[Cx, Cy, Cz, 0, 0, 0],
                      [0, 0, 0, Cx, Cy, Cz]])
        
        ke = data.E * data.A[i] / L * np.array([[1, -1],
                                               [-1, 1]])
        
        Ke = T.T @ ke @ T
        
        dofs = [3*n1-3, 3*n1-2, 3*n1-1, 3*n2-3, 3*n2-2, 3*n2-1]
        for j in range(6):
            for k in range(6):
                K[dofs[j], dofs[k]] += Ke[j,k]
    
    # Apply boundary conditions and solve
    free_dofs = [i for i in range(ndof) if i not in data.fixed_dofs]
    F = np.zeros(ndof)
    for dof, load in data.loads.items():
        F[dof] = load
        
    u = np.zeros(ndof)
    u[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], 
                                  F[free_dofs])
    
    return u

def create_3d_plot(data, displacements=None, scale=1.0):
    fig = go.Figure()
    
    # Plot nodes
    fig.add_trace(go.Scatter3d(
        x=data.coord[:, 0],
        y=data.coord[:, 1],
        z=data.coord[:, 2],
        mode='markers',
        marker=dict(size=8, color='blue'),
        name='Nodes'
    ))
    
    # Plot elements
    for elem in data.conec:
        n1, n2 = elem
        x = [data.coord[n1-1, 0], data.coord[n2-1, 0]]
        y = [data.coord[n1-1, 1], data.coord[n2-1, 1]]
        z = [data.coord[n1-1, 2], data.coord[n2-1, 2]]
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='black', width=3),
            name=f'Element {n1}-{n2}'
        ))
    
    # Plot supports
    for dof in data.fixed_dofs:
        node = dof // 3
        x = data.coord[node, 0]
        y = data.coord[node, 1]
        z = data.coord[node, 2]
        
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers',
            marker=dict(
                size=12,
                symbol='diamond',
                color='red'
            ),
            name='Support'
        ))
    
    # Plot loads
    for dof, load in data.loads.items():
        node = dof // 3
        x = data.coord[node, 0]
        y = data.coord[node, 1]
        z = data.coord[node, 2]
        
        # Add arrow for load
        dir_vector = np.zeros(3)
        dir_vector[dof % 3] = np.sign(load)
        
        fig.add_trace(go.Cone(
            x=[x], y=[y], z=[z],
            u=[dir_vector[0]], 
            v=[dir_vector[1]], 
            w=[dir_vector[2]],
            colorscale='Reds',
            sizemode='absolute',
            sizeref=2,
            name='Load'
        ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        title='3D Truss Analysis'
    )
    
    return fig

def main():
    st.set_page_config(layout="wide")
    
    st.title("🏗️ 3D Truss Analysis")
    st.markdown("""
    This app performs 3D truss analysis using the finite element method.
    Enter the structure details below and visualize the results!
    """)
    
    with st.sidebar:
        st.header("Input Parameters")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Manual Input", "Sample Problem"]
        )
        
        if input_method == "Sample Problem":
            # Load sample problem
            coord = np.array([
                [72, 0, 0],
                [0, 36, 0],
                [0, 36, 72],
                [0, 0, -48]
            ])
            
            conec = np.array([
                [1, 2],
                [1, 3],
                [1, 4]
            ])
            
            E = 1.2e6
            A = np.array([0.302, 0.729, 0.187])
            fixed_dofs = [3, 4, 5, 6, 7, 8, 9, 10, 11]
            loads = {2: -1000}
            
        else:
            # Manual input interface
            st.subheader("Node Coordinates")
            coord_data = st.text_area(
                "Enter coordinates (x,y,z per line):",
                "72,0,0\n0,36,0\n0,36,72\n0,0,-48"
            )
            coord = np.array([list(map(float, line.split(','))) 
                            for line in coord_data.strip().split('\n')])
            
            st.subheader("Element Connectivity")
            conec_data = st.text_area(
                "Enter connectivity (node1,node2 per line):",
                "1,2\n1,3\n1,4"
            )
            conec = np.array([list(map(int, line.split(','))) 
                            for line in conec_data.strip().split('\n')])
            
            st.subheader("Material Properties")
            E = st.number_input("Young's modulus (E):", value=1.2e6)
            A_data = st.text_input("Cross-sectional areas:", "0.302,0.729,0.187")
            A = np.array(list(map(float, A_data.split(','))))
            
            st.subheader("Boundary Conditions")
            fixed_dofs_data = st.text_input(
                "Fixed DOFs (comma-separated):",
                "3,4,5,6,7,8,9,10,11"
            )
            fixed_dofs = list(map(int, fixed_dofs_data.split(',')))
            
            st.subheader("Loads")
            load_dof = st.number_input("Load DOF:", value=2)
            load_value = st.number_input("Load value:", value=-1000.0)
            loads = {load_dof: load_value}
    
    # Create truss data object
    truss_data = TrussData(coord, conec, E, A, fixed_dofs, loads)
    
    # Create two columns for visualization and results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("3D Visualization")
        fig = create_3d_plot(truss_data)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Analysis Results")
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Analyzing structure..."):
                # Solve the truss
                displacements = solve_truss(truss_data)
                
                # Display results
                st.success("Analysis completed!")
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'DOF': range(1, len(displacements) + 1),
                    'Displacement': displacements
                })
                
                st.dataframe(
                    results_df.style.highlight_max(axis=0)
                             .highlight_min(axis=0),
                    hide_index=True
                )
                
                # Plot displacement graph
                st.subheader("Displacement Plot")
                st.bar_chart(results_df.set_index('DOF')['Displacement'])

if __name__ == "__main__":
    main()
