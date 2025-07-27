#!/usr/bin/env python3
"""
Demo script để test PDG2Seq với Hypergraph
"""
import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append('.')

from model.PDG2Seq_DGCN import PDG2Seq_HyperGCN, HyperedgeBuilder

def test_hyperedge_builder():
    """Test HyperedgeBuilder với dữ liệu giả"""
    print("Testing HyperedgeBuilder...")
    
    # Tạo dữ liệu giả
    num_nodes = 10
    num_time = 100
    pick = np.random.rand(num_time, num_nodes)
    drop = np.random.rand(num_time, num_nodes)
    distance_matrix = np.random.rand(num_nodes, num_nodes)
    
    # Test các loại hyperedge
    print("\n1. Testing pick_drop similarity edges...")
    edges = HyperedgeBuilder.build_pick_drop_similarity_edges(pick, drop, threshold=0.5)
    print(f"Pick-drop similarity edges shape: {edges.shape}")
    
    print("\n2. Testing geographical edges...")
    edges = HyperedgeBuilder.build_geographical_edges(distance_matrix, threshold=0.5)
    print(f"Geographical edges shape: {edges.shape}")
    
    print("\n3. Testing temporal change edges...")
    edges = HyperedgeBuilder.build_temporal_change_edges(pick, drop, threshold=0.5)
    print(f"Temporal change edges shape: {edges.shape}")
    
    print("\n4. Testing correlation edges...")
    edges = HyperedgeBuilder.build_correlation_edges(pick, drop, threshold=0.5)
    print(f"Correlation edges shape: {edges.shape}")
    
    print("\n5. Testing temporal pattern edges...")
    edges = HyperedgeBuilder.build_temporal_pattern_edges(pick, drop, window=5, threshold=0.5)
    print(f"Temporal pattern edges shape: {edges.shape}")

def test_hypergcn_model():
    """Test PDG2Seq_HyperGCN model"""
    print("\n\nTesting PDG2Seq_HyperGCN model...")
    
    # Model parameters
    dim_in = 2
    dim_out = 16
    cheb_k = 2
    embed_dim = 16
    time_dim = 8
    batch_size = 4
    num_nodes = 10
    
    # Khởi tạo model
    model = PDG2Seq_HyperGCN(
        dim_in=dim_in, 
        dim_out=dim_out, 
        cheb_k=cheb_k, 
        embed_dim=embed_dim, 
        time_dim=time_dim,
        dataset_name='test',
        hyperedge_types=['pick_drop', 'geo']
    )
    
    # Test input
    x = torch.randn(batch_size, num_nodes, dim_in)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    try:
        output = model(x)
        print(f"Output shape: {output.shape}")
        print("✓ Model forward pass successful!")
    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")

def test_with_real_data():
    """Test với dữ liệu thật nếu có"""
    print("\n\nTesting with real NYC-Bike data...")
    
    try:
        # Load real data
        pick, drop = HyperedgeBuilder.load_pick_drop_data('NYC-Bike')
        distance_matrix = HyperedgeBuilder.load_distance_matrix('NYC-Bike')
        
        print(f"Pick data shape: {pick.shape}")
        print(f"Drop data shape: {drop.shape}")
        if distance_matrix is not None:
            print(f"Distance matrix shape: {distance_matrix.shape}")
        
        # Test hyperedge building
        edges = HyperedgeBuilder.build_pick_drop_similarity_edges(pick, drop, threshold=0.9)
        print(f"Real data pick-drop edges: {edges.shape}")
        
        if distance_matrix is not None:
            edges = HyperedgeBuilder.build_geographical_edges(distance_matrix, threshold=0.1)
            print(f"Real data geographical edges: {edges.shape}")
        
        print("✓ Real data test successful!")
        
    except Exception as e:
        print(f"✗ Real data test failed: {e}")
        print("This is expected if NYC-Bike data is not available")

if __name__ == '__main__':
    print("=== PDG2Seq Hypergraph Demo ===")
    
    # Test 1: HyperedgeBuilder
    test_hyperedge_builder()
    
    # Test 2: Model
    test_hypergcn_model()
    
    # Test 3: Real data (if available)
    test_with_real_data()
    
    print("\n=== Demo completed ===")
    print("\nTo run full training:")
    print("python run_hypergraph.py --dataset NYC-Bike --model PDG2Seq_HyperGCN --mode train")
