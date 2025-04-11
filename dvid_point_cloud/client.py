"""Client for interacting with DVID HTTP API."""

import logging
from typing import Any, Dict, List
from dataclasses import dataclass
import json

import requests

logger = logging.getLogger(__name__)

@dataclass
class SparseVolumeStats:
    """Class to hold statistics about a sparse volume."""
    num_voxels: int
    num_blocks: int
    min_voxel: tuple
    max_voxel: tuple

class DVIDClient:
    """Client for making HTTP requests to DVID server."""

    def __init__(self, server: str, timeout: int = 60):
        """
        Initialize DVID client.

        Args:
            server: Base URL for the DVID server (without trailing slash)
            timeout: Request timeout in seconds
        """
        self.server = server.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()


    def get_info(self, uuid: str, instance: str) -> Dict[str, Any]:
        """
        Get instance info from DVID.
        
        Args:
            uuid: UUID of the DVID node
            instance: Name of the data instance

        Returns:
            Dictionary of instance info
        """
        url = f"{self.server}/api/node/{uuid}/{instance}/info"
        logger.debug(f"GET request to {url}")
        
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        
        return response.json()

    def get_sparse_vol_stats(self, uuid: str, instance: str, label_id: int) -> SparseVolumeStats:
        """
        Get sparse volume statistics for a specific label ID.
        
        Args:
            uuid: UUID of the DVID node
            instance: Name of the labelmap instance (usually 'segmentation')
            label_id: Label ID to query

        Returns:
            SparseVolumeStats object containing # voxels, # blocks, min/max voxel coords
        """
        url = f"{self.server}/api/node/{uuid}/{instance}/sparsevol-size/{label_id}"
        
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        return SparseVolumeStats(
            num_voxels=data["voxels"],
            num_blocks=data["numblocks"],
            min_voxel=tuple(data["minvoxel"]),
            max_voxel=tuple(data["maxvoxel"])
        )
    
    def get_label(self, uuid: str, instance: str, point: tuple[int, int, int], supervoxels: bool = False) -> bytes:
        """
        Get label data for a specific point.
        
        Args:
            uuid: UUID of the DVID node
            instance: Name of the labelmap instance (usually 'segmentation')
            point: Tuple of (x, y, z) coordinates
            supervoxels: If True, returns supervoxel data instead of agglomerated body data

        Returns:
            Binary encoded label data

        """
        url = f"{self.server}/api/node/{uuid}/{instance}/label/{point[0]}_{point[1]}_{point[2]}"

        params = {}
        if supervoxels:
            params["supervoxels"] = "true"

        logger.debug(f"GET request to {url} with params {params}")
        
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        return response.content

    def get_labels(self, uuid: str, instance: str, points: List[List[int]], supervoxels: bool = False) -> List[int]:
        """
        Get label data for multiple points.
        
        Args:
            uuid: UUID of the DVID node
            instance: Name of the labelmap instance (usually 'segmentation')
            points: List of Lists of (x, y, z) coordinates
            supervoxels: If True, returns supervoxel data instead of agglomerated body data

        Returns:
            List of label IDs for each point
        """
        url = f"{self.server}/api/node/{uuid}/{instance}/labels"
        
        params = {}
        if supervoxels:
            params["supervoxels"] = "true"
            
        logger.debug(f"GET request to {url} with params {params}")
        
        response = self.session.get(url, params=params, json=points, timeout=self.timeout)
        response.raise_for_status()
        
        return response.json()
        
        

    def get_sparse_vol(self, uuid: str, instance: str, label_id: int, 
                   format: str = "rles", scale: int = 0, supervoxels: bool = False) -> bytes:
        """
        Get sparse volume data for a specific label ID.
        
        Args:
            uuid: UUID of the DVID node
            instance: Name of the labelmap instance (usually 'segmentation')
            label_id: Label ID to query
            format: Format of the sparse volume ('rles' or 'blocks')
            scale: Resolution scale (0 is highest resolution)
            supervoxels: If True, returns supervoxel data instead of agglomerated body data

        Returns:
            Binary encoded sparse volume data
        """
        url = f"{self.server}/api/node/{uuid}/{instance}/sparsevol/{label_id}"
        params = {"format": format}
        
        if scale > 0:
            params["scale"] = scale
            
        if supervoxels:
            params["supervoxels"] = "true"
            
        logger.debug(f"GET request to {url} with params {params}")
        
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        return response.content
        
    def get_label_blocks(self, uuid: str, instance: str, block_coords: str, 
                          scale: int = 0, supervoxels: bool = False) -> bytes:
        """
        Get blocks of label data for specific block coordinates.
        
        Args:
            uuid: UUID of the DVID node
            instance: Name of the labelmap instance (usually 'segmentation')
            block_coords: Comma-separated string of block coordinates (e.g., "10,11,12,13,14,15")
            scale: Resolution scale (0 is highest resolution)
            supervoxels: If True, returns unmapped supervoxels instead of agglomerated labels

        Returns:
            Binary encoded block data
        """
        url = f"{self.server}/api/node/{uuid}/{instance}/specificblocks"
        params = {
            "blocks": block_coords,
            "scale": scale
        }
        if supervoxels:
            params["supervoxels"] = "true"
            
        logger.debug(f"GET request to {url} with params {params}")
        
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        return response.content