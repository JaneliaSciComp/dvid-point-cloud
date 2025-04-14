"""Tests for the client module."""

import json
import numpy as np
import pytest

from dvid_point_cloud.client import DVIDClient


def test_dvid_client_initialization():
    """Test that the DVIDClient is initialized correctly."""
    server = "http://test-server.org"
    client = DVIDClient(server)
    
    assert client.server == server
    assert client.timeout == 60  # Default timeout
    
    # Test with custom timeout
    client = DVIDClient(server, timeout=120)
    assert client.timeout == 120


def test_label_at_point():
    """Test that the label at a point is correctly returned."""
    server = "https://hemibrain-dvid.janelia.org"
    uuid = "15aee239283143c08b827177ebee01b3"
    instance = "segmentation"
    point = (20000, 30000, 33000)
    client = DVIDClient(server, timeout=120)
    label = client.get_label(uuid, instance, point)
    assert label == 2351004142

def test_labels_at_multiple_points():
    """Test that the label at a point is correctly returned."""
    server = "https://hemibrain-dvid.janelia.org"
    uuid = "15aee239283143c08b827177ebee01b3"
    instance = "segmentation"
    points = [[20000, 30000, 33000], [20000, 30000, 35000]]
    client = DVIDClient(server, timeout=120)
    labels = client.get_labels(uuid, instance, points)
    assert labels == [2351004142, 2261637853]

def test_supervoxel_at_point():
    """Test that the label at a point is correctly returned."""
    server = "https://hemibrain-dvid.janelia.org"
    uuid = "15aee239283143c08b827177ebee01b3"
    instance = "segmentation"
    point = (20000, 30000, 35000)
    client = DVIDClient(server, timeout=120)
    label = client.get_label(uuid, instance, point, supervoxels=True)
    assert label == 2353058939

def test_sample_supervoxels(mock_server):
    """Test that supervoxel sampling works correctly."""
    # Set up test parameters
    server = "http://test-server"
    uuid = "test-uuid"
    instance = "segmentation"
    body_id = 42
    
    # Create mock response content
    mock_supervoxels = [2351004142, 2353058939]
    mock_content = json.dumps(mock_supervoxels).encode('utf-8')
    
    # Configure mock server
    mock_server.get(f"{server}/api/node/{uuid}/{instance}/supervoxels/{body_id}",
                   content=mock_content)
    
    # Call sample_supervoxels
    client = DVIDClient(server, timeout=120)
    supervoxels = client.get_supervoxels(uuid, instance, body_id)
    
    # Check that the response is correct
    assert isinstance(supervoxels, np.ndarray)
    assert supervoxels.dtype == np.int64
    np.testing.assert_array_equal(supervoxels, np.array(mock_supervoxels))
    
    # Check that the request was made correctly
    assert mock_server.called
    assert mock_server.call_count == 1
    
    request = mock_server.request_history[0]
    assert request.method == "GET"
    assert request.url == f"{server}/api/node/{uuid}/{instance}/supervoxels/{body_id}"


def test_get_sparse_vol(mock_server):
    """Test that the get_sparse_vol method makes the correct HTTP request."""
    server = "http://test-server.org"
    uuid = "abc123"
    instance = "segmentation"
    label_id = 42
    
    # Create mock response content
    mock_content = b"mock sparse volume data"
    
    # Configure mock server
    mock_server.get(f"{server}/api/node/{uuid}/{instance}/sparsevol/{label_id}?format=rles",
                  content=mock_content)
    
    # Call the method
    client = DVIDClient(server)
    response = client.get_sparse_vol(uuid, instance, label_id, format="rles")
    
    # Check that the response is correct
    assert response == mock_content
    
    # Check that the request was made correctly
    assert mock_server.called
    assert mock_server.call_count == 1
    
    request = mock_server.request_history[0]
    assert request.method == "GET"
    assert request.url == f"{server}/api/node/{uuid}/{instance}/sparsevol/{label_id}?format=rles"


def test_get_info(mock_server):
    """Test that the get_info method makes the correct HTTP request."""
    server = "http://test-server.org"
    uuid = "abc123"
    instance = "segmentation"
    
    # Create mock response content
    mock_info = {
        "Base": {
            "TypeName": "labelmap",
            "Name": "segmentation"
        },
        "Extended": {
            "BlockSize": [64, 64, 64],
            "VoxelSize": [8, 8, 8],
            "VoxelUnits": ["nanometers", "nanometers", "nanometers"]
        }
    }
    
    # Configure mock server
    mock_server.get(f"{server}/api/node/{uuid}/{instance}/info",
                  json=mock_info)
    
    # Call the method
    client = DVIDClient(server)
    response = client.get_info(uuid, instance)
    
    # Check that the response is correct
    assert response == mock_info
    
    # Check that the request was made correctly
    assert mock_server.called
    assert mock_server.call_count == 1
    
    request = mock_server.request_history[0]
    assert request.method == "GET"
    assert request.url == f"{server}/api/node/{uuid}/{instance}/info"


def test_get_label_blocks(mock_server):
    """Test that the get_label_blocks method makes the correct HTTP request."""
    server = "http://test-server.org"
    uuid = "abc123"
    instance = "segmentation"
    block_coords = "0,0,0,1,1,1"
    
    # Create mock response content
    mock_content = b"mock block data"
    
    # Configure mock server
    mock_server.get(f"{server}/api/node/{uuid}/{instance}/specificblocks?blocks={block_coords}&scale=0",
                  content=mock_content)
    
    # Call the method
    client = DVIDClient(server)
    response = client.get_label_blocks(uuid, instance, block_coords)
    
    # Check that the response is correct
    assert response == mock_content
    
    # Check that the request was made correctly
    assert mock_server.called
    assert mock_server.call_count == 1
    
    request = mock_server.request_history[0]
    assert request.method == "GET"
    # Use requests' unquote to handle URL encoding differences
    import urllib.parse
    request_url = urllib.parse.unquote(request.url)
    expected_url = f"{server}/api/node/{uuid}/{instance}/specificblocks?blocks={block_coords}&scale=0"
    assert request_url == expected_url
    
    # Test with custom scale and supervoxels params
    mock_server.get(f"{server}/api/node/{uuid}/{instance}/specificblocks?blocks={block_coords}&scale=2&supervoxels=true",
                  content=mock_content)
    
    response = client.get_label_blocks(uuid, instance, block_coords, scale=2, supervoxels=True)
    
    assert response == mock_content
    assert mock_server.call_count == 2
    
    request = mock_server.request_history[1]
    assert request.method == "GET"
    import urllib.parse
    request_url = urllib.parse.unquote(request.url)
    expected_url = f"{server}/api/node/{uuid}/{instance}/specificblocks?blocks={block_coords}&scale=2&supervoxels=true"
    assert request_url == expected_url