
import numpy as np
from threading import Thread
import torch




class FEA_INP():

    class Parts():
        def __init__(self) -> None:
            self.elems: dict['str', np.ndarray]
            self.nodes: torch.Tensor
            self.sections: list
            self.sets_nodes: dict['str', set]
            self.sets_elems: dict['str', set]
            self.surfaces: dict[str, list[tuple[np.ndarray, int]]]
            
            self.num_elems_3D: int
            self.num_elems_2D: int
            self.elems_material: torch.Tensor
            """
            0: index of the element\n
            1: density of the element\n
            2: type of the element\n
            3-: parameter of the element
            """
            # self.sets_nodes = {}
            # self.sets_elems = {}
            # self.num_elems_3D = 0
            # self.num_elems_2D = 0
            # section = []

        def read(self, origin_data: list[str], ind):

            self.elems = {}
            self.sets_nodes = {}
            self.sets_elems = {}
            self.surfaces = {}
            self.num_elems_3D = 0
            self.num_elems_2D = 0
            section = []
            while ind < len(origin_data):
                now = origin_data[ind]
                if len(now) > 9 and now[0:9] == '*End Part':
                    break
                # case element
                if len(now) == 22 and now[0:21] == '*Element, type=C3D10H':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['C3D10H'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue

                if len(now) == 21 and now[0:20] == '*Element, type=C3D10':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['C3D10'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue

                if len(now) == 21 and now[0:20] == '*Element, type=C3D15':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['C3D15'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue
                
                if len(now) == 21 and now[0:20] == '*Element, type=C3D20':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    
                    num_line = round(len(datalist)/2)
                    datalist = [datalist[2*i]+datalist[2*i+1] for i in range(num_line)]
                    self.elems['C3D20'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue
                
                if len(now) == 21 and now[0:20] == '*Element, type=C3D8R':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['C3D8R'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue

                if len(now) == 21 and now[0:20] == '*Element, type=C3D4H':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['C3D4H'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue

                if len(now) == 20 and now[0:19] == '*Element, type=C3D4':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['C3D4'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue
                
                if len(now) == 20 and now[0:19] == '*Element, type=C3D8':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['C3D8'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue

                if len(now) == 20 and now[0:19] == '*Element, type=C3D6':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['C3D6'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue

                if len(now) >= 17 and now[0:17] == '*Element, type=S3':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['S3'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_2D += ind1 - ind0
                    continue

                # case node set
                if len(now
                    ) >= 12 and now[0:12] == '*Nset, nset=':
                    # name = now[12:].replace('\n', '').strip()
                    data_now = now.split('=')[1].split(',')
                    for ii in range(len(data_now)):
                        data_now[ii] = data_now[ii].strip()
                    name = data_now[0].strip()
                    ind += 1
                    if not 'generate' in data_now:
                        ind0 = ind
                        now = origin_data[ind]
                        while now[0] != '*':
                            ind += 1
                            now = origin_data[ind]
                            ind1 = ind
                        datalist = [[
                            int(i) for i in row.replace('\n', '').replace(
                                ',', ' ').strip().split()
                        ] for row in origin_data[ind0:ind1]]
                        datalist = [
                            element for sublist in datalist for element in sublist
                        ]
                        self.sets_nodes[name] = set(
                            (torch.tensor(datalist, dtype=torch.int) - 1).tolist())
                    else:
                        now = list(map(int, origin_data[ind].split(',')))
                        self.sets_nodes[name] = set(
                            (torch.arange(now[0], now[1]+1, now[2]) - 1).tolist())
                    continue

                # case element set
                if len(now) >= 14 and now[
                        0:14] == '*Elset, elset=':
                    # name = now[14:].replace('\n', '').strip()
                    data_now = now.split('=')[1].split(',')
                    for ii in range(len(data_now)):
                        data_now[ii] = data_now[ii].strip()
                    name = data_now[0].strip()
                    ind += 1
                    if not 'generate' in data_now:
                        ind0 = ind
                        now = origin_data[ind]
                        while now[0] != '*':
                            ind += 1
                            now = origin_data[ind]
                            ind1 = ind
                        datalist = [[
                            int(i) for i in row.replace('\n', '').replace(
                                ',', ' ').strip().split()
                        ] for row in origin_data[ind0:ind1]]
                        datalist = [
                            element for sublist in datalist for element in sublist
                        ]
                        self.sets_elems[name] = set(
                            (torch.tensor(datalist, dtype=torch.int) - 1).tolist())
                        continue
                    else:
                        now = list(map(int, origin_data[ind].split(',')))
                        self.sets_elems[name] = set(
                            (torch.arange(now[0], now[1]+1, now[2]) - 1).tolist())
                        continue
                
                # case surfaces
                if len(now) >= 8 and now[0:8] == '*Surface':
                    data_now = now.split('=')
                    ind += 1
                    if len(self.elems.keys()) == 0:
                        continue
                    if data_now[1].split(',')[0].strip()[:7] == 'ELEMENT':
                        name = data_now[2].strip()
                        self.surfaces[name] = []
                        surfaceList = []
                        while origin_data[ind][0] != '*':
                            data_now = origin_data[ind].split(',')
                            ind+=1
                            elem_set_name = data_now[0].strip()
                            surface_index = int(data_now[1].strip()[1:])
                            for key in list(self.elems.keys()):
                                elem_now = self.elems[key]
                                elem_index = np.where(np.isin(elem_now[:, 0],
                                                        list(self.sets_elems[elem_set_name])))[0]
                                elem = elem_now[elem_index]
                                if elem.shape[1] == 5:
                                    if surface_index == 1:
                                        surfaceList.append(elem[:, [1,3,2]])
                                    elif surface_index == 2:
                                        surfaceList.append(elem[:, [1,2,4]])
                                    elif surface_index == 3:
                                        surfaceList.append(elem[:, [2,3,4]])
                                    elif surface_index == 4:
                                        surfaceList.append(elem[:, [3,1,4]])
                                elif elem.shape[1] == 11:
                                    if surface_index == 1:
                                        surfaceList.append(elem[:, [1,7,5]])
                                        surfaceList.append(elem[:, [2,5,6]])
                                        surfaceList.append(elem[:, [3,6,7]])
                                        surfaceList.append(elem[:, [5,7,6]])
                                    elif surface_index == 2:
                                        surfaceList.append(elem[:, [1,5,8]])
                                        surfaceList.append(elem[:, [2,9,5]])
                                        surfaceList.append(elem[:, [4,8,9]])
                                        surfaceList.append(elem[:, [5,9,8]])
                                    elif surface_index == 3:
                                        surfaceList.append(elem[:, [2,6,9]])
                                        surfaceList.append(elem[:, [3,10,6]])
                                        surfaceList.append(elem[:, [4,9,10]])
                                        surfaceList.append(elem[:, [6,10,9]])
                                    elif surface_index == 4:
                                        surfaceList.append(elem[:, [1,8,7]])
                                        surfaceList.append(elem[:, [3,7,10]])
                                        surfaceList.append(elem[:, [4,10,8]])
                                        surfaceList.append(elem[:, [8,10,7]])
                                        
                                self.surfaces[name].append((elem_now[elem_index, 0], surface_index-1))
                        try:
                            self.surfaces_tri[name] = np.concatenate(surfaceList, axis=0)
                        except:
                            pass
                    continue

                # case node
                if len(now) >= 5 and now[0:5] == '*Node':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        float(i) for i in row.replace('\n', '').strip().split(',')
                    ] for row in origin_data[ind0:ind1]]
                    self.nodes = torch.tensor(datalist)
                    self.nodes[:, 0] -= 1
                    continue

                # case section
                if len(now) >= 11 and now[0:11] == '** Section:':
                    name = now.split(':')[1].strip()
                    ind += 1
                    now = origin_data[ind]
                    data = now.split(',')
                    section_set = data[1].split('=')[1].strip()
                    section_material = data[2].split('=')[1].strip()
                    section.append([section_set, section_material])

                # case finished
                if len(now) >= 9 and now[0:9] == '*End Part':
                    break

                ind += 1

            self.sections = section
            self.elems_material = -torch.ones(
                [self.num_elems_2D + self.num_elems_3D, 5])

    class Assembly():
        def __init__(self) -> None:
            self.instances: dict['str', tuple[str, np.ndarray]]
            self.sets_nodes: dict['str', set]
            self.sets_elems: dict['str', set]
            self.nodes: torch.Tensor
            self.sets_elems: dict['str', set]
            self.sets_nodes: dict['str', set]

        def read(self, origin_data: list[str], ind):
            self.instances = {}
            self.sets_nodes = {}
            self.sets_elems = {}
            
            while ind < len(origin_data):
                now = origin_data[ind]
                if len(now) >= 13 and now[0:13] == '*End Assembly':
                    break
                    
                # case instance
                if len(now) >= 11 and now[0:11] == '*Instance, ':
                    # 解析实例参数
                    data_now = now.split(',')
                    instance_name = None
                    part_name = None
                    
                    for param in data_now:
                        param = param.strip()
                        if param.startswith('name='):
                            instance_name = param.split('=')[1].strip()
                        elif param.startswith('part='):
                            part_name = param.split('=')[1].strip()
                    
                    ind += 1
                    # 检查是否有变换矩阵
                    transform = np.eye(4)  # 默认单位矩阵
                    if ind < len(origin_data) and origin_data[ind][0] != '*':
                        # 解析第一行：平移变换
                        transform_data = list(map(float, origin_data[ind].split(',')))
                        if len(transform_data) >= 3:
                            # 平移变换
                            transform[0, 3] = transform_data[0]
                            transform[1, 3] = transform_data[1] 
                            transform[2, 3] = transform_data[2]
                        ind += 1
                        
                        # 检查是否有第二行：旋转变换
                        if ind < len(origin_data) and origin_data[ind][0] != '*':
                            rotation_data = list(map(float, origin_data[ind].split(',')))
                            if len(rotation_data) >= 7:
                                # 旋转轴起点
                                axis_start = np.array([rotation_data[0], rotation_data[1], rotation_data[2]])
                                # 旋转轴终点
                                axis_end = np.array([rotation_data[3], rotation_data[4], rotation_data[5]])
                                # 旋转角度（度）
                                angle_deg = rotation_data[6]
                                
                                # 计算旋转轴方向向量
                                axis_vector = axis_end - axis_start
                                axis_vector = axis_vector / np.linalg.norm(axis_vector)  # 归一化
                                
                                # 将角度转换为弧度
                                angle_rad = np.radians(angle_deg)
                                
                                # 使用Rodrigues旋转公式构建旋转矩阵
                                cos_theta = np.cos(angle_rad)
                                sin_theta = np.sin(angle_rad)
                                one_minus_cos = 1 - cos_theta
                                
                                ux, uy, uz = axis_vector
                                
                                # 构建旋转矩阵
                                rotation_matrix = np.array([
                                    [cos_theta + ux*ux*one_minus_cos, 
                                    ux*uy*one_minus_cos - uz*sin_theta, 
                                    ux*uz*one_minus_cos + uy*sin_theta],
                                    [uy*ux*one_minus_cos + uz*sin_theta, 
                                    cos_theta + uy*uy*one_minus_cos, 
                                    uy*uz*one_minus_cos - ux*sin_theta],
                                    [uz*ux*one_minus_cos - uy*sin_theta, 
                                    uz*uy*one_minus_cos + ux*sin_theta, 
                                    cos_theta + uz*uz*one_minus_cos]
                                ])
                                
                                # 将旋转矩阵嵌入到4x4变换矩阵中
                                transform[:3, :3] = rotation_matrix
                                
                                # 如果旋转轴不过原点，需要调整平移部分
                                # T = T_translate * T_rotate_about_axis
                                # 先将点移动到旋转轴起点，然后旋转，再移回
                                translation_adjustment = axis_start - rotation_matrix @ axis_start
                                transform[0, 3] += translation_adjustment[0]
                                transform[1, 3] += translation_adjustment[1]
                                transform[2, 3] += translation_adjustment[2]
                                
                            ind += 1

                    self.instances[instance_name] = (part_name, transform)
                    continue
        
                # case node set
                if len(now) >= 12 and now[0:12] == '*Nset, nset=':
                    data_now = now.split('=')[1].split(',')
                    for ii in range(len(data_now)):
                        data_now[ii] = data_now[ii].strip()
                    name = data_now[0].strip()
                    ind += 1
                    if not 'generate' in data_now:
                        ind0 = ind
                        now = origin_data[ind]
                        while now[0] != '*':
                            ind += 1
                            now = origin_data[ind]
                            ind1 = ind
                        datalist = [[
                            int(i) for i in row.replace('\n', '').replace(
                                ',', ' ').strip().split()
                        ] for row in origin_data[ind0:ind1]]
                        datalist = [
                            element for sublist in datalist for element in sublist
                        ]
                        self.sets_nodes[name] = set(
                            (torch.tensor(datalist, dtype=torch.int) - 1).tolist())
                    else:
                        now = list(map(int, origin_data[ind].split(',')))
                        self.sets_nodes[name] = set(
                            (torch.arange(now[0], now[1]+1, now[2]) - 1).tolist())
                        ind += 1
                    continue
        
                # case element set  
                if len(now) >= 14 and now[0:14] == '*Elset, elset=':
                    data_now = now.split('=')[1].split(',')
                    for ii in range(len(data_now)):
                        data_now[ii] = data_now[ii].strip()
                    name = data_now[0].strip()
                    instance_name = None
                    
                    # 检查是否指定了实例
                    for param in data_now:
                        if param.strip().startswith('instance='):
                            instance_name = param.strip().split('=')[1]
                            break
                    
                    ind += 1
                    if not 'generate' in data_now:
                        ind0 = ind
                        now = origin_data[ind]
                        while now[0] != '*':
                            ind += 1
                            now = origin_data[ind]
                            ind1 = ind
                        datalist = [[
                            int(i) for i in row.replace('\n', '').replace(
                                ',', ' ').strip().split()
                        ] for row in origin_data[ind0:ind1]]
                        datalist = [
                            element for sublist in datalist for element in sublist
                        ]
                        if instance_name:
                            # 如果指定了实例，使用实例名作为前缀
                            full_name = f"{instance_name}.{name}"
                        else:
                            full_name = name
                        self.sets_elems[full_name] = set(
                            (torch.tensor(datalist, dtype=torch.int) - 1).tolist())
                    else:
                        now = list(map(int, origin_data[ind].split(',')))
                        if instance_name:
                            full_name = f"{instance_name}.{name}"
                        else:
                            full_name = name
                        self.sets_elems[full_name] = set(
                            (torch.arange(now[0], now[1]+1, now[2]) - 1).tolist())
                        ind += 1
                    continue
        
                # case node
                if len(now) >= 5 and now[0:5] == '*Node':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        float(i) for i in row.replace('\n', '').strip().split(',')
                    ] for row in origin_data[ind0:ind1]]
                    self.nodes = torch.tensor(datalist)
                    self.nodes[:, 0] -= 1
                    continue
        
                ind += 1

    class Materials():
        # materials [density, type:(0:linear, 1:neohooken), para:]

        def __init__(self) -> None:
            self.type: int
            self.mat_para: list[float]
            self.density: float = 0.0

        def read(self, origin_data: list[str], ind: int):
            while ind < len(origin_data):
                now = origin_data[ind]

                if len(now) >= 13 and now[0:13] == '*Hyperelastic':
                    self.type = 1
                    ind += 1
                    now = origin_data[ind]
                    self.mat_para = list(map(float, now.split(',')))
                    self.mat_para[0] = self.mat_para[0] * 2
                    self.mat_para[1] = 2 / (self.mat_para[1])

                if len(now) >= 8 and now[0:8] == '*Elastic':
                    self.type = 0
                    ind += 1
                    now = origin_data[ind]
                    self.mat_para = list(map(float, now.split(',')))

                if len(now) >= 8 and now[0:8] == '*Density':
                    ind += 1
                    now = origin_data[ind]
                    self.density = float(now.split(',')[0])

                if len(now) >= 9 and now[0:9] == '*Material':
                    break
                if len(now) >= 2 and now[0:2] == '**':
                    break
                ind += 1


    def __init__(self) -> None:
        """
        Initializes the FEA_INP class.

        This method initializes the FEA_INP class and sets up the necessary attributes.

        Args:
            None

        Returns:
            None
        """

        self.part: dict['str', FEA_INP.Parts] = {}
        self.material: dict['str', FEA_INP.Materials] = {}
        self.assemble: FEA_INP.Assembly = FEA_INP.Assembly()
        self.disp_result: list[dict['str', torch.Tensor]] = []

    def Read_INP(self, path):
        """
        Reads an INP file.

        This method reads an INP file and extracts the necessary information such as assembly, parts, and materials.

        Args:
            path (str): The path to the INP file.

        Returns:
            None
        """
        threads = []
        self.part = {}
        self.material = {}

        f = open(path)
        origin_data = f.readlines()
        f.close()
        for findex in range(len(origin_data)):
            now = origin_data[findex]
            if len(now) >= 16 and now[0:16] == '*Assembly, name=':
                name = now[16:].replace('\n', '').strip()
                self.assemble = FEA_INP.Assembly()
                self.assemble.read(origin_data=origin_data, ind=findex + 1)

            if len(now) >= 12 and now[0:12] == '*Part, name=':
                name = now[12:].replace('\n', '').strip()
                self.part[name] = FEA_INP.Parts()
                self.part[name].read(origin_data=origin_data, ind=findex + 1)

            if len(now) >= 16 and now[0:16] == '*Material, name=':
                name = now[16:].replace('\n', '').strip()
                self.material[name] = FEA_INP.Materials()
                self.material[name].read(
                    origin_data=origin_data,
                    ind=findex + 1
                )

        for p_key in self.part.keys():
            p = self.part[p_key]
            for sec in p.sections:
                index = torch.tensor(list(p.sets_elems[sec[0]]))
                mat = self.material[sec[1]]
                p.elems_material[index, 0] = index.type_as(p.elems_material)
                p.elems_material[index, 1] = mat.density
                p.elems_material[index, 2] = mat.type
                p.elems_material[index, 3] = mat.mat_para[0]
                p.elems_material[index, 4] = mat.mat_para[1]
