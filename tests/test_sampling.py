"""Tests for the sampling module."""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pytest

from dvid_point_cloud.client import DVIDClient
from dvid_point_cloud.parse import parse_rles, rles_to_points
from dvid_point_cloud.sampling import uniform_sample

def test_parse_rles(create_sparse_volume):
    """Test that RLE format is correctly parsed."""
    # Create a sparse volume with known runs
    runs = [
        (10, 20, 30, 5),   # Run of 5 voxels starting at (10, 20, 30)
        (15, 20, 30, 10),  # Run of 10 voxels starting at (15, 20, 30)
        (0, 0, 0, 3)       # Run of 3 voxels starting at (0, 0, 0)
    ]
    sparse_vol_data = create_sparse_volume(runs)
    
    # Parse RLEs
    starts_zyx, lengths = parse_rles(sparse_vol_data)
    
    # Check that parsed runs match input runs
    assert len(starts_zyx) == len(runs)
    assert len(lengths) == len(runs)
    
    # Check that the values match (note: starts_zyx is in ZYX order)
    for i, (x, y, z, length) in enumerate(runs):
        assert starts_zyx[i, 0] == z
        assert starts_zyx[i, 1] == y
        assert starts_zyx[i, 2] == x
        assert lengths[i] == length


def test_rles_to_points():
    """Test conversion of RLEs to points based on sample indices."""
    # Create runs
    runs = [
        (10, 20, 30, 5),   # Run of 5 voxels at indices 0-4
        (15, 20, 30, 10),  # Run of 10 voxels at indices 5-14
        (0, 0, 0, 3)       # Run of 3 voxels at indices 15-17
    ]
    
    # Sample indices (one from each run)
    sample_indices = np.array([2, 10, 16])
    
    # Convert to points
    points = rles_to_points(runs, 18, sample_indices)
    
    # Expected points
    expected = np.array([
        [12, 20, 30],  # Run 1, offset 2
        [20, 20, 30],  # Run 2, offset 5
        [1, 0, 0]      # Run 3, offset 1
    ])
    
    # Check that points match expected
    np.testing.assert_array_equal(points, expected)

def test_uniform_sample_integration(mock_server, create_sparse_volume, generate_sparse_volume):
    """Integration test for uniform_sample function."""
    # Set up test parameters
    server = "http://test-server"
    uuid = "test-uuid"
    instance = "segmentation"
    label_id = 42
    density = 0.1
    
    # Generate a random sparse volume
    size = (128, 128, 128)
    _, runs, total_voxels = generate_sparse_volume(size, label_id, density=0.2, seed=42)
    
    # Create sparse volume data
    sparse_vol_data = create_sparse_volume(runs)
    
    # Configure mock server
    mock_server.get(f"{server}/api/node/{uuid}/{instance}/sparsevol/{label_id}?format=rles",
                   content=sparse_vol_data)
    
    # Call uniform_sample
    point_cloud = uniform_sample(server, uuid, label_id, density, instance)
    
    # Check that the point cloud has the expected shape
    expected_samples = max(1, int(round(total_voxels * density)))
    assert point_cloud.shape == (expected_samples, 3)
    
    # Ensure all points are within the volume bounds
    assert np.all(point_cloud[:, 0] >= 0)
    assert np.all(point_cloud[:, 1] >= 0)
    assert np.all(point_cloud[:, 2] >= 0)
    assert np.all(point_cloud[:, 0] < size[0])
    assert np.all(point_cloud[:, 1] < size[1])
    assert np.all(point_cloud[:, 2] < size[2])


def test_uniform_sample_fuzz(mock_server, create_sparse_volume, generate_sparse_volume):
    """Fuzz test for uniform_sample with different random seeds."""
    # Set up test parameters
    server = "http://test-server"
    uuid = "test-uuid"
    instance = "segmentation"
    label_id = 42
    
    # Try multiple random seeds
    for seed in range(5):
        # Generate a random sparse volume
        size = (64, 64, 64)
        _, runs, total_voxels = generate_sparse_volume(size, label_id, density=0.3, seed=seed)
        
        # Create sparse volume data
        sparse_vol_data = create_sparse_volume(runs)
        
        # Configure mock server
        mock_server.get(f"{server}/api/node/{uuid}/{instance}/sparsevol/{label_id}?format=rles",
                       content=sparse_vol_data)
        
        # Try different densities
        for density in [0.01, 0.05, 0.1]:
            # Call uniform_sample
            point_cloud = uniform_sample(server, uuid, label_id, density, instance)
            
            # Check that the point cloud has the expected shape
            expected_samples = max(1, int(round(total_voxels * density)))
            assert point_cloud.shape == (expected_samples, 3)
            
            # Ensure all points are within the volume bounds
            assert np.all(point_cloud[:, 0] >= 0)
            assert np.all(point_cloud[:, 1] >= 0)
            assert np.all(point_cloud[:, 2] >= 0)
            assert np.all(point_cloud[:, 0] < size[0])
            assert np.all(point_cloud[:, 1] < size[1])
            assert np.all(point_cloud[:, 2] < size[2])