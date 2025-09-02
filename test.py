import numpy as np
import scipy
import time

import scipy.spatial
from collections import defaultdict
def find_close_pairs(points, d):
    """
    查找3D点集中所有距离小于d的点对
    
    参数:
        points: numpy数组，形状为[n, 3]，表示n个3D点的坐标
        d: 距离阈值，小于此距离的点对会被返回
    
    返回:
        numpy数组，形状为[2, m]，其中m是符合条件的点对数量，
        每行表示点对的索引(i, j)，满足i < j且两点距离 < d
    """
    # 输入验证
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("输入必须是形状为[n, 3]的numpy数组")
    
    n = points.shape[0]
    if n < 2:
        return np.empty((2, 0), dtype=int)  # 点数量不足，返回空数组
    
    # 构建KDTree以加速近邻搜索
    kdtree = scipy.spatial.cKDTree(points)
    

    pairs = kdtree.query_pairs(d, output_type='ndarray')

    return pairs.T.astype(np.int64)


# 示例用法
if __name__ == "__main__":
    # 生成随机的3D点集（1000个点）
    np.random.seed(42)
    num_points = 100000
    points = np.random.rand(num_points, 3)  # 坐标在[0,1)范围内
    
    # 设置距离阈值
    distance_threshold = 0.01
    
    # 查找近距离点对
    start_time = time.time()
    close_pairs = find_close_pairs(points, distance_threshold)
    end_time = time.time()

    print(f"查找近距离点对耗时: {end_time - start_time:.4f}秒")
    
    
    # # 输出结果信息
    # print(f"点集规模: {num_points}个3D点")
    # print(f"距离阈值: {distance_threshold}")
    # print(f"找到的近距离点对数量: {close_pairs.shape[1]}")
    
    
    # # 打印前10个点对（如果有的话）
    # if close_pairs.size > 0:
    #     print("前10个点对索引:")
    #     print(close_pairs[:, :10].T)
        
    #     # 验证第一个点对的距离
    #     i, j = close_pairs[:, 0]
    #     dist = np.linalg.norm(points[i] - points[j])
    #     print(f"第一个点对的实际距离: {dist:.6f}")
    # print(f"空间网格算法耗时: {end_time - start_time:.4f}秒")
    
    
    # # 输出结果信息
    # print(f"点集规模: {num_points}个3D点")
    # print(f"距离阈值: {distance_threshold}")
    # print(f"找到的近距离点对数量: {close_pairs.shape[1]}")
    
    
    # # 打印前10个点对（如果有的话）
    # if close_pairs.size > 0:
    #     print("前10个点对索引:")
    #     print(close_pairs[:, :10].T)
        
    #     # 验证第一个点对的距离
    #     i, j = close_pairs[:, 0]
    #     dist = np.linalg.norm(points[i] - points[j])
    #     print(f"第一个点对的实际距离: {dist:.6f}")
    
    # # 输出结果信息
    # print(f"点集规模: {num_points}个3D点")
    # print(f"距离阈值: {distance_threshold}")
    # print(f"找到的近距离点对数量: {close_pairs.shape[1]}")
    
    
    # # 打印前10个点对（如果有的话）
    # if close_pairs.size > 0:
    #     print("前10个点对索引:")
    #     print(close_pairs[:, :10].T)
        
    #     # 验证第一个点对的距离
    #     i, j = close_pairs[:, 0]
    #     dist = np.linalg.norm(points[i] - points[j])
    #     print(f"第一个点对的实际距离: {dist:.6f}")
        
    #     # 验证第一个点对的距离
    #     i, j = close_pairs[:, 0]
    #     dist = np.linalg.norm(points[i] - points[j])
    #     print(f"第一个点对的实际距离: {dist:.6f}")
