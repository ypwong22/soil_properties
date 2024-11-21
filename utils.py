import os
import numpy as np
import rasterio as rio
from typing import Union, List


def vert_interp(
    target_nodes: Union[List[float], np.ndarray],
    input_nodes: Union[List[float], np.ndarray],
    input_data: np.ndarray,
    target_single_level: bool = False,
    target_interfaces: Union[List[float], np.ndarray, None] = None,
    input_interfaces: Union[List[float], np.ndarray, None] = None,
) -> np.ndarray:
    """
    Linearly interpolate soil moisture/soil temperature from input_nodes to target_nodes. 
    If the target depths are single-level, returns weighted average based on the distance 
        between the input nodes and target node. 
    If the target depths are defined by bounds (target_interfaces != None), returns weighted
        average based on the overlapping lengths between the target_interfaces and 
        input_interface. 

    Parameters:
    -----------
    target_nodes : Union[List[float], np.ndarray]
        List or numpy array of target node depths in meters.
    
    input_nodes : Union[List[float], np.ndarray]
        List or numpy array of input node depths in meters.

    input_data : np.ndarray
        2D numpy array of input data with shape (time, len(input_nodes)).

    target_single_level : bool
        Indicates whether the target nodes are single level. If true, target_interface and input_interfaces are un-used. 

    Returns:
    --------
    np.ndarray
        Processed data as a 2D numpy array with shape (time, len(target_nodes)).
    """
    # unifying data types
    target_nodes = np.array(target_nodes)
    input_nodes = np.array(input_nodes)

    # sanity checks
    if not input_data.shape[1] == len(input_nodes):
        raise Exception('Mismatch between specified inputs depths and available data')
    if not target_single_level:
        if (target_interfaces is None or input_interfaces is None):
            raise Exception('Must specify depth bounds if target is not single level')
        if not ((len(target_interfaces) - len(target_nodes)) == 1):
            raise Exception('Number of soil layers mismatched between target interface and node depths')
        if not ((len(input_interfaces) - len(input_nodes)) == 1):
            raise Exception('Number of soil layers mismatched between input interface and node depths')
        # unifying data types
        target_interfaces = np.array(target_interfaces)
        input_interfaces = np.array(input_interfaces)

    # actual calculations
    output_data = np.full([input_data.shape[0], len(target_nodes)], np.nan)
    if target_single_level:
        for i, d in enumerate(target_nodes):
            if d < input_nodes[0]:
                output_data[:, i] = input_data[:, 0]
            elif d > input_nodes[-1]:
                output_data[:, i] = input_data[:, -1]
            else:
                d_matched = np.where(np.isclose(input_nodes, d))[0]
                if len(d_matched) > 1:
                    raise Exception('Input nodes have duplicate values')
                elif len(d_matched) == 1:
                    # just apply the nearest node
                    output_data[:, i] = input_data[:, d_matched[0]]
                else:
                    # interpolate between two nearby nodes
                    d_up = np.where(input_nodes < d)[0][-1]
                    d_down = np.where(input_nodes > d)[0][0]
                    f1 = (input_nodes[d_down] - d) / (input_nodes[d_down] - input_nodes[d_up]) 
                    f2 = (d - input_nodes[d_up]) / (input_nodes[d_down] - input_nodes[d_up])
                    output_data[:, i] = input_data[:, d_up] * f1 + input_data[:, d_down] * f2
    else:
        for i, d1 in enumerate(target_interfaces[:-1]):
            d2 = target_interfaces[i+1]
            if d2 <= input_interfaces[1]:
                output_data[:, i] = input_data[:, 0]
            elif d1 >= input_interfaces[-2]:
                output_data[:, i] = input_data[:, -1]
            else:
                output_data[:, i] = 0.
                sum_weight = 0.
                for j, dd1 in enumerate(input_interfaces[:-1]):
                    dd2 = input_interfaces[j+1]
                    if (dd2 <= d1) or (dd1 >= d2):
                        continue
                    else:
                        if (dd1 >= d1):
                            if (dd2 <= d2):
                                sum_weight += (dd2 - dd1)
                                output_data[:, i] += input_data[:, j] * (dd2 - dd1)
                            else:
                                sum_weight += (d2 - dd1)
                                output_data[:, i] += input_data[:, j] * (d2 - dd1)
                        else:
                            if (dd2 <= d2):
                                sum_weight += (dd2 - d1)
                                output_data[:, i] += input_data[:, j] * (dd2 - d1)
                            else:
                                sum_weight = 1.
                                output_data[:, i] = input_data[:, j]
                                break
                output_data[:, i] /= sum_weight
                # print(i, d1, d2, sum_weight)

    return output_data