import numpy as np
import bisect
import tf_transformations
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
import bisect

def align_timestamps(data1, data2):
    """
    将两个数据集按照时间戳对齐，以data1为基准
    
    参数:
    data1 (list): 第一个数据集，元素为 (timestamp, message)
    data2 (list): 第二个数据集，元素为 (timestamp, message)
    
    返回:
    tuple: 包含两个对齐后的数据集
        aligned_data1 (list): 与输入data1顺序相同的时间戳对齐数据
        aligned_data2 (list): 与aligned_data1对应的时间戳对齐数据
    """
    # 确保输入数据按时间排序（虽然read_bag返回的数据默认已排序）
    data1_sorted = sorted(data2, key=lambda x: x[0])  # 注意：参数顺序修正
    data2_sorted = sorted(data1, key=lambda x: x[0])  # 注意：参数顺序修正
    
    # 修正：原参数顺序有误，data1应为基准
    # 正确排序应保持data1的原始顺序，仅排序data2
    # 正确实现：
    data1_sorted = sorted(data1, key=lambda x: x[0])
    data2_sorted = sorted(data2, key=lambda x: x[0])
    
    aligned_data1 = []
    aligned_data2 = []
    
    # 提取data2的时间戳列表用于快速查找
    data2_timestamps = [item[0] for item in data2_sorted]
    
    for idx1, (t1, msg1) in enumerate(data1_sorted):
        # 使用bisect找到插入位置
        pos = bisect.bisect_left(data2_timestamps, t1)
        
        # 处理边界情况
        if pos == 0:
            closest = data2_sorted[0]
        elif pos >= len(data2_sorted):
            closest = data2_sorted[-1]
        else:
            # 比较前后两个时间戳
            before = data2_sorted[pos-1]
            after = data2_sorted[pos]
            if (t1 - before[0]) <= (after[0] - t1):
                closest = before
            else:
                closest = after
        
        # 添加匹配结果（保持data1原始顺序）
        aligned_data1.append((t1, msg1))
        aligned_data2.append(closest)
    
    return aligned_data1, aligned_data2
def read_bag(bag_path, topic_name):
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}
    
    storage_filter = rosbag2_py.StorageFilter([topic_name])
    reader.set_filter(storage_filter)
    
    data = []
    while reader.has_next():
        (topic, msg_bytes, t) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(msg_bytes, msg_type)
        data.append((t, msg))
    return data

def get_transform_matrix(msg):
    position = msg.pose.pose.position
    orientation = msg.pose.pose.orientation
    translation = [position.x, position.y, position.z]
    rotation = [orientation.x, orientation.y, orientation.z, orientation.w]
    translation_matrix = tf_transformations.translation_matrix(translation)
    rotation_matrix = tf_transformations.quaternion_matrix(rotation)
    return tf_transformations.concatenate_matrices(translation_matrix, rotation_matrix)

def calculate_total_distance(msgs):
    total_distance = 0.0
    prev_pos = None
    for _, msg in msgs:
        pos = msg.pose.pose.position
        if prev_pos is not None:
            dx = pos.x - prev_pos.x
            dy = pos.y - prev_pos.y
            total_distance += np.sqrt(dx**2 + dy**2)
        prev_pos = pos
    return total_distance
import numpy as np
import matplotlib.pyplot as plt
from tf_transformations import quaternion_matrix, translation_matrix, concatenate_matrices, inverse_matrix, translation_from_matrix

def transform_and_visualize(aligned_data1, aligned_data2):
    """
    将第二个数据集转换到第一个数据集的坐标系下，并可视化轨迹
    
    参数:
    aligned_data1 (list): 对齐后的基准数据集（时间戳已对齐）
    aligned_data2 (list): 对齐后的待转换数据集（时间戳已对齐）
    
    返回:
    list: 转换后的数据集（仅包含x,y坐标）
    """
    if not aligned_data1 or not aligned_data2:
        raise ValueError("输入数据不能为空")

    # 获取初始位姿的变换矩阵
    def get_transform_matrix(msg):
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        trans_mat = translation_matrix([pos.x, pos.y, pos.z])
        rot_mat = quaternion_matrix([quat.x, quat.y, quat.z, quat.w])
        return concatenate_matrices(trans_mat, rot_mat)

    T1 = get_transform_matrix(aligned_data1[0][1])
    T2_initial = get_transform_matrix(aligned_data2[0][1])
    
    # 计算坐标系转换矩阵
    transform_mat = concatenate_matrices(T1, inverse_matrix(T2_initial))

    # 转换数据并收集轨迹
    transformed_data = []
    original_traj1 = []
    original_traj2 = []
    transformed_traj = []

    for (t1, msg1), (t2, msg2) in zip(aligned_data1, aligned_data2):
        # 原始轨迹数据
        original_traj1.append((msg1.pose.pose.position.x, msg1.pose.pose.position.y))
        original_traj2.append((msg2.pose.pose.position.x, msg2.pose.pose.position.y))
        
        # 计算转换后的位姿
        T2_current = get_transform_matrix(msg2)
        transformed_T = concatenate_matrices(transform_mat, T2_current)
        trans = translation_from_matrix(transformed_T)
        transformed_traj.append((trans[0], trans[1]))
        transformed_data.append((t2, trans))  # 保留时间戳和转换后的坐标

    # 可视化
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制原始轨迹
    x1, y1 = zip(*original_traj1)
    ax.plot(x1, y1, label='base trah', color='blue', linewidth=2)
    
    # 绘制转换前轨迹
    x2_orig, y2_orig = zip(*original_traj2)
    ax.plot(x2_orig, y2_orig, label='origin VIO', color='red', linestyle='--', alpha=0.7)
    
    # 绘制转换后轨迹
    x_trans, y_trans = zip(*transformed_traj)
    ax.plot(x_trans, y_trans, label='transed VIO', color='green', linewidth=1.5)
    
    ax.set_title('traj compare', fontsize=14)
    ax.set_xlabel('X  (m)', fontsize=12)
    ax.set_ylabel('Y  (m)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()

    return transformed_data


import numpy as np
import matplotlib.pyplot as plt
from tf_transformations import quaternion_matrix, translation_matrix, concatenate_matrices, inverse_matrix, translation_from_matrix

import numpy as np
import matplotlib.pyplot as plt
from tf_transformations import quaternion_matrix, translation_matrix, concatenate_matrices, inverse_matrix, translation_from_matrix

import numpy as np



def cal_scale_factor(aligned_data1, aligned_data2, rate=(0.5, 0.01), times=100, d=0.5):
    """
    计算最优缩放因子
    
    参数:
    aligned_data1 (list): 对齐后的待缩放数据集（时间戳已对齐）
    aligned_data2 (list): 基准数据集
    rate (tuple): 初始调整速率和最终调整速率（例如：(0.5, 0.01)）
    times (int): 最大迭代次数
    d (float): 允许的最大点距阈值（米）
    
    返回:
    float: 最优缩放因子
    """

    # 计算几何中心
    def calculate_geometric_center(data):
        xs = [x for _, (x, _) in data]
        ys = [y for _, (_, y) in data]
        return (np.mean(xs), np.mean(ys))
    
    center_x, center_y = calculate_geometric_center(aligned_data1)
    
    def calculate_center_distances(scaling_factor):
        """基于几何中心计算缩放后的点距"""
        passed = 0
        min_dist,max_dist = 0,0
        for (t1, (x1, y1)), (_, msg2) in zip(aligned_data1, aligned_data2):
            # 计算缩放后的坐标
            x_scaled = center_x + (x1 - center_x) * scaling_factor
            y_scaled = center_y + (y1 - center_y) * scaling_factor
            
            # 获取基准坐标
            x2 = msg2.pose.pose.position.x
            y2 = msg2.pose.pose.position.y
            dist = np.sqrt((x_scaled - x2)**2 + (y_scaled - y2)**2)
            if dist > max_dist:
                max_dist = dist
            if dist < min_dist or min_dist == 0:
                min_dist = dist
            if dist <= d:
                passed += 1
        return passed / len(aligned_data1)
    # aligned_data1为((t1, (x,y))、aligned_data2为(timestamp, msg)格式的列表
    def calculate_distances(scaling_factor):
        """计算缩放后的点距"""
        passed = 0
        # 遍历aligned_data1和aligned_data2，计算缩放后的点距
        for (t1, (x1, y1)), (_, msg2) in zip(aligned_data1, aligned_data2):
            x2 = msg2.pose.pose.position.x
            y2 = msg2.pose.pose.position.y
            dist = np.sqrt((x1*scaling_factor - x2)**2 + (y1*scaling_factor - y2)**2)
            if dist <= d:
                passed += 1
        return passed / len(aligned_data1)
    
    def calculate_total_distance(data):
        """计算轨迹总距离"""
        total = 0.0
        prev = None
        for _, msg in data:
            pos = msg.pose.pose.position
            if prev is not None:
                dx = pos.x - prev.x
                dy = pos.y - prev.y
                total += np.sqrt(dx**2 + dy**2)
            prev = pos
        return total
    def calculate_data1_distances(aligned_data1,factor=1.0):
        """计算aligned_data1的总距离"""
        total = 0.0
        prev = None
        for t1, (x1, y1) in aligned_data1:
            if prev is not None:
                dx = x1 * factor - prev[0]
                dy = y1 * factor - prev[1]
                total += np.sqrt(dx**2 + dy**2)
            prev = (x1 * factor, y1 * factor)
        return total
    
    # 初始检查
    initial_pass_rate = calculate_center_distances(1.0)
    if initial_pass_rate >= 0.8:
        return 1.0
    
    # 初始化参数
    scale = 1.0
    distance1 = calculate_data1_distances(aligned_data1)
    distance2 = calculate_total_distance(aligned_data2)
    rate_step = (rate[0] - rate[1]) / times
    best_scale,best_pass_rate = scale,0
    for i in range(times):
        current_rate = rate[0] - rate_step * i
        
        # 调整缩放因子
        if distance1 < distance2:
            scale *= (1 + current_rate)
        else:
            scale *= (1 - current_rate)
        # 检查是否满足条件
        distance1 = calculate_data1_distances(aligned_data1, scale)
        
        pass_rate = calculate_center_distances(scale)
        print(f"当前缩放因子: {scale:.2f}, 当前点距: {distance1:.2f}")
        print(f"当前通过率: {pass_rate:.2f}")
        if best_pass_rate< pass_rate:
            best_pass_rate = pass_rate
            best_scale = scale
        if pass_rate >= 0.95:
            print(f"缩放因子: {scale:.2f}, 通过率: {pass_rate:.2f}")
            print("current time:", i)
            break
    if pass_rate < 0.8:
        print("未找到合适的缩放因子")
    return best_scale
def transform_and_visualize_pro(aligned_data1, aligned_data2, scale_enabled=False):
    """
    将第一个数据集转换到第二个数据集的坐标系下，并可视化轨迹
    
    参数:
    aligned_data1 (list): 对齐后的待转换数据集（时间戳已对齐）
    aligned_data2 (list): 对齐后的基准数据集（目标坐标系）
    scale_enabled (bool): 是否启用尺度缩放功能
    
    返回:
    list: 转换后的数据集（包含时间戳和(x,y)坐标）
    """
    if not aligned_data1 or not aligned_data2:
        raise ValueError("输入数据不能为空")

    # 计算总距离函数
    def calculate_total_distance(msgs):
        total = 0.0
        prev = None
        for _, msg in msgs:
            pos = msg.pose.pose.position
            if prev is not None:
                dx = pos.x - prev.x
                dy = pos.y - prev.y
                total += np.sqrt(dx**2 + dy**2)
            prev = pos
        return total

    # 获取变换矩阵
    def get_transform_matrix(msg):
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        trans_mat = translation_matrix([pos.x, pos.y, pos.z])
        rot_mat = quaternion_matrix([quat.x, quat.y, quat.z, quat.w])
        return concatenate_matrices(trans_mat, rot_mat)

    

    # 先对aligned_data1进行放缩
    # if scale_enabled:
    #     for i in range(len(aligned_data1)):
    #         pos = aligned_data1[i][1].pose.pose.position
    #         pos.x *= scale_factor
    #         pos.y *= scale_factor
    #         aligned_data1[i][1].pose.pose.position = pos

    # 构建基础变换矩阵（关键修正）
    T1_initial = get_transform_matrix(aligned_data1[0][1])  # 数据集1初始位姿
    T2_initial = get_transform_matrix(aligned_data2[0][1])  # 数据集2初始位姿

    # 初始变换矩阵：将数据集1的初始位姿转换到数据集2的坐标系下
    T_base = concatenate_matrices(inverse_matrix(T2_initial), T1_initial)

    # 转换数据并收集轨迹
    transformed_data = []
    traj2, traj1_orig, traj1_trans = [], [], []

    for (t1, msg1), (t2, msg2) in zip(aligned_data1, aligned_data2):
        # 基准轨迹数据（aligned_data2）
        pos2 = msg2.pose.pose.position
        traj2.append((pos2.x, pos2.y))
        
        # 原始aligned_data1轨迹
        pos1 = msg1.pose.pose.position
        traj1_orig.append((pos1.x, pos1.y))
        
        # 计算当前数据集1的位姿矩阵
        T1_current = get_transform_matrix(msg1)
        
        # 计算相对位姿变化（相对于数据集1的初始位姿）
        delta_T1 = concatenate_matrices(inverse_matrix(T1_initial), T1_current)
        
        # 应用基础变换矩阵，得到在数据集2坐标系下的位姿
        T_transformed = concatenate_matrices(T2_initial, delta_T1)
        
        # 提取平移向量
        trans = translation_from_matrix(T_transformed)
        
        # 应用缩放（仅缩放平移部分）
        # scaled_trans = trans * np.array([scale_factor, scale_factor, 1.0]) if scale_enabled else trans
        
        # 记录转换后的轨迹点
        traj1_trans.append((trans[0], trans[1]))
        transformed_data.append((t1, (trans[0], trans[1])))
    def calculate_geometric_center(traj1_trans):
        xs = [x for (x,y) in traj1_trans]
        ys = [y for (x,y) in traj1_trans]
        return (np.mean(xs), np.mean(ys))
    
    center_x, center_y = calculate_geometric_center(traj1_trans)
    # 放缩
    if scale_enabled:
        scale_factor = cal_scale_factor(
                            transformed_data,
                            aligned_data2,
                            rate=(0.1, 0.01),  # 调整速率从50%线性降到1%
                            times=100,         # 最大迭代100次
                            d=1.3              # 允许1.3cm的点距误差
                        )
        print(f"缩放因子: {scale_factor:.2f}")
    else:
        scale_factor = 1.0
    
    # 基于几何中心计算缩放缩放,遍历traj1_trans
    for i in range(len(traj1_trans)):
        x1,y1 = traj1_trans[i][0],traj1_trans[i][1]
        x_scaled = center_x + (x1 - center_x) * scale_factor
        y_scaled = center_y + (y1 - center_y) * scale_factor
        traj1_trans[i] = (x_scaled, y_scaled)
    # 可视化
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制基准轨迹（aligned_data2）
    x2, y2 = zip(*traj2)
    ax.plot(x2, y2, label=f'base (dis {calculate_total_distance(aligned_data2):.2f}m)', 
            color='blue', linewidth=2)
    
    # 绘制原始aligned_data1轨迹
    x1_orig, y1_orig = zip(*traj1_orig)
    ax.plot(x1_orig, y1_orig, label=f'vio (dis {calculate_total_distance(aligned_data1):.2f}m)', 
            color='red', linestyle='--', alpha=0.7)
    
    # 绘制转换后轨迹
    label = 'transed vio'
    if scale_enabled:
        label += f' (scale {scale_factor:.2f}x)'
    x_trans, y_trans = zip(*traj1_trans)
    ax.plot(x_trans, y_trans, label=f'{label} (dis {calculate_total_distance(aligned_data2):.2f}m)', 
            color='green', linewidth=1.5)
    
    # 添加图注
    ax.set_title('compare', fontsize=14)
    ax.set_xlabel('X  (m)', fontsize=12)
    ax.set_ylabel('Y  (m)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axis('equal')
    
    # 显示缩放信息
    # if scale_enabled:
    #     plt.text(0.05, 0.95, f'scale: {scale_factor:.3f}\n'
    #              f'vio dis: {source_distance:.2f}m\n'
    #              f'targe dis: {target_distance:.2f}m',
    #              transform=ax.transAxes, fontsize=10, 
    #              bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

    return transformed_data


    
def main(bag_path, gt_topic, vio_topic):
    gt_msgs = read_bag(bag_path, gt_topic)
    vio_msgs = read_bag(bag_path, vio_topic)
    
    if not gt_msgs or not vio_msgs:
        print("Error: One of the topics has no messages.")
        return
    
    def get_frequency(msgs):
        if len(msgs) < 2:
            return 0.0
        times = [t for t, _ in msgs]
        duration = times[-1] - times[0]
        return len(times) / duration
    
    # freq_gt = get_frequency(gt_msgs)
    # freq_vio = get_frequency(vio_msgs)
    # times_gt = [t for t, _ in gt_msgs]
    # times_vio = [t for t, _ in vio_msgs]
    base_topic, base_msgs, other_topic, other_msgs = (
        (gt_topic, gt_msgs, vio_topic, vio_msgs) if len(gt_msgs) < len(vio_msgs)
        else (vio_topic, vio_msgs, gt_topic, gt_msgs)
    )

    real_total_distance = calculate_total_distance(gt_msgs)
    vio_total_distance = calculate_total_distance(vio_msgs)
    aligned_base_msgs, aligned_other_msgs = align_timestamps(base_msgs, other_msgs)

    transformed_other_msg = transform_and_visualize_pro(aligned_base_msgs, aligned_other_msgs,True)
    
    
    if len(base_msgs) == 0 or len(transformed_other_msg) == 0:
        print("Error: One of the topics has no messages.")
        return
    
   
    # transform_matrix = np.dot(base_first_matrix, np.linalg.inv(other_first_matrix))
    
    other_timestamps = [t for t, _ in other_msgs]
    cumulative_error = 0.0
    # 遍历aligned_base_msgs和transformed_other_msg，计算累计误差
    for id in range(len(aligned_other_msgs)):
        msg_base = aligned_other_msgs[id][1]
        msg_transformed = transformed_other_msg[id][1]
        x_base, y_base, z_base = msg_base.pose.pose.position.x, msg_base.pose.pose.position.y, msg_base.pose.pose.position.z
        x_transformed, y_transformed = msg_transformed[0], msg_transformed[1]
        
        
        cumulative_error += np.sqrt((x_base - x_transformed) ** 2 + (y_base - y_transformed) ** 2)
    
    drift_rate = (cumulative_error / real_total_distance) * 1000
    
    print(f"累计欧氏距离误差: {cumulative_error:.4f} 米")
    print(f"漂移率: {drift_rate:.2f} ‰")
    print(f"VIO总距离: {vio_total_distance:.4f} 米")
    print(f"真实总距离: {real_total_distance:.4f} 米")

if __name__ == "__main__":
    # 参数配置（修改以下值即可）
    BAG_PATH = "/home/autoware/vins_ws/bags/result_bags/0423_b_stereo_key10"  # 替换为实际ROS2包路径
    GT_TOPIC = "/fixposition/odometry_enu"    # 替换为真实参考里程计话题
    VIO_TOPIC = "/odometry_rect"            # 替换为VIO里程计话题
    
    main(BAG_PATH, GT_TOPIC, VIO_TOPIC)