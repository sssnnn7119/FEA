
import numpy as np
from threading import Thread
import torch

class parts():
    def __init__(self, name) -> None:
        self.__name = name
        self.elems: dict['str', np.ndarray]
        self.nodes: torch.Tensor
        self.sections: list
        self.sets_nodes: dict['str', set]
        self.sets_elems: dict['str', set]
        self.surfaces_tri: dict['str', np.ndarray]
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
        self.surfaces_tri = {}
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
            if len(now) >= 5 + len(self.__name) and now[
                    0:5 + len(self.__name)] == '*End ' + self.__name:
                break

            ind += 1

        self.sections = section
        self.elems_material = -torch.ones(
            [self.num_elems_2D + self.num_elems_3D, 5])


class materials():
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


class FEA_INP():


    def __init__(self) -> None:
        """
        Initializes the FEA_INP class.

        This method initializes the FEA_INP class and sets up the necessary attributes.

        Args:
            None

        Returns:
            None
        """

        self.part: dict['str', parts] = {}
        self.material: dict['str', materials] = {}
        self.assemble: parts
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
                self.assemble = parts('Assembly')
                self.assemble.read(origin_data=origin_data, ind=findex + 1)

            if len(now) >= 12 and now[0:12] == '*Part, name=':
                name = now[12:].replace('\n', '').strip()
                self.part[name] = parts(name='Part')
                self.part[name].read(origin_data=origin_data, ind=findex + 1)

            if len(now) >= 16 and now[0:16] == '*Material, name=':
                name = now[16:].replace('\n', '').strip()
                self.material[name] = materials()
                threads.append(
                    Thread(target=self.material[name].read,
                           kwargs={
                               'origin_data': origin_data,
                               'ind': findex + 1
                           }))
                threads[-1].start()

        for i in range(len(threads)):
            threads[i].join()

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

    def Read_Result(self, file_path, file_list):
        """
        Reads result files.

        This method reads result files and extracts the displacement results.

        Args:
            file_path (str): The path to the result files.
            file_list (list): A list of result file names.

        Returns:
            None
        """
        self.disp_result = []
        for file in file_list:
            self.disp_result.append({})
            data = np.loadtxt(open(file_path + file, 'rb'),
                              delimiter=',',
                              skiprows=1,
                              dtype=str)
            name0 = data[:, 3]
            label0 = data[:, 4].astype(float)
            U0 = data[:, 11:].astype(float)

            name = np.unique(name0)
            for i in name:
                index = i == name0
                num = np.sum(index.astype(int))
                self.disp_result[-1][i[1:-1]] = torch.tensor(
                    np.append(label0[index].reshape(num, 1),
                              U0[index, :],
                              axis=1).tolist())

    def Get_Volumn(self, part_ind):
        if type(part_ind) == int:
            part_ind = list(self.part.keys())[part_ind]

        volumn = torch.zeros(self.part[part_ind].num_elems_3D)

        def volumn_Tet(self, elem):
            pt0 = self.part[part_ind].nodes[elem[:, 1]][:, 1:]
            pt1 = self.part[part_ind].nodes[elem[:, 2]][:, 1:]
            pt2 = self.part[part_ind].nodes[elem[:, 3]][:, 1:]
            pt3 = self.part[part_ind].nodes[elem[:, 4]][:, 1:]
            vec1 = pt1 - pt0
            vec2 = pt2 - pt0
            vec3 = pt3 - pt0
            volumn = torch.abs(
                torch.sum(torch.cross(vec1, vec2) * vec3, axis=1)) / 6
            return volumn

        if 'C3D10H' in list(self.part[part_ind].elems.keys()):
            elem = self.part[part_ind].elems['C3D10H']
            volumn[elem[:, 0]] = volumn_Tet(self, elem)

        if 'C3D10' in list(self.part[part_ind].elems.keys()):
            elem = self.part[part_ind].elems['C3D10']
            volumn[elem[:, 0]] = volumn_Tet(self, elem)

        if 'C3D4H' in list(self.part[part_ind].elems.keys()):
            elem = self.part[part_ind].elems['C3D4H']
            volumn[elem[:, 0]] = volumn_Tet(self, elem)

        if 'C3D4' in list(self.part[part_ind].elems.keys()):
            elem = self.part[part_ind].elems['C3D4']
            volumn[elem[:, 0]] = volumn_Tet(self, elem)

        return volumn

    def Get_Volumn_Closed_Shell(self, shell_elems, nodes):
        node1 = nodes[shell_elems[:, 0], :]
        node2 = nodes[shell_elems[:, 1], :]
        node3 = nodes[shell_elems[:, 2], :]
        Volumn = 1 / 6 * (torch.cross(node1, node2, dim=1) * node3).sum()
        return Volumn

    def Find_Surface(self, surf_set: list[str], part_ind=0):
        """
        surf_set: list of surface set name
        part_ind: index of part
        """
        if type(part_ind) == int:
            part_ind = list(self.part.keys())[part_ind]

        nodes = self.part[part_ind].nodes
        elems_map = self.part[part_ind].elems
        shell_elems = []
        normal_vec = []
        elem_index = []
        center_point = []

        def add_surf(a, b, c):
            node1 = nodes[elems[i, a], 1:]
            node2 = nodes[elems[i, b], 1:]
            node3 = nodes[elems[i, c], 1:]
            elem_index.append(elems[i, 0])
            normal_vec.append(torch.cross(node2 - node1, node3 - node1) / 2)
            center_point.append((node1 + node2 + node3) / 3)
            shell_elems.append(elems[i, [a, b, c]])

        for surf in surf_set:
            node_set = self.part[part_ind].sets_nodes[surf]
            elems_set = self.part[part_ind].sets_elems[surf]

            for key in list(elems_map.keys()):
                elems_map_now = elems_map[key]
                elems = np.array([
                    elems_map_now[i] for i in range(len(elems_map_now))
                    if int(elems_map_now[i, 0]) in elems_set
                ])

                for i in range(len(elems)):

                    # find the surface nodes
                    if i == 243:
                        print('ok')

                    # nodes of element that lie on the surface mark 1, else 0
                    index_surface_nodes = np.array(
                        [True if j in node_set else False for j in elems[i]])
                    # modify the normal vector
                    if elems.shape[1] == 11:
                        if index_surface_nodes[2] and index_surface_nodes[3] and index_surface_nodes[4] and index_surface_nodes[6] and index_surface_nodes[9] and index_surface_nodes[10]: 
                            add_surf(2, 6, 9)
                            add_surf(3, 10, 6)
                            add_surf(4, 9, 10)
                            add_surf(6, 10, 9)
                        if index_surface_nodes[1] and index_surface_nodes[3] and index_surface_nodes[4]and index_surface_nodes[7] and index_surface_nodes[8] and index_surface_nodes[10]:
                            add_surf(1, 8, 7)
                            add_surf(3, 7, 10)
                            add_surf(4, 10, 8)
                            add_surf(8, 10, 7)
                        if index_surface_nodes[1] and index_surface_nodes[2] and index_surface_nodes[4]and index_surface_nodes[5] and index_surface_nodes[8] and index_surface_nodes[9]:
                            add_surf(1, 5, 8)
                            add_surf(2, 9, 5)
                            add_surf(4, 8, 9)
                            add_surf(9, 8, 5)
                        if index_surface_nodes[1] and index_surface_nodes[2] and index_surface_nodes[3]and index_surface_nodes[5] and index_surface_nodes[6] and index_surface_nodes[7]:
                            add_surf(1, 7, 5)
                            add_surf(2, 5, 6)
                            add_surf(3, 6, 7)
                            add_surf(5, 7, 6)
                    if elems.shape[1] == 5:
                        if index_surface_nodes[2] and index_surface_nodes[3] and index_surface_nodes[4]:
                            add_surf(2, 3, 4)
                        if index_surface_nodes[1] and index_surface_nodes[3] and index_surface_nodes[4]:
                            add_surf(1, 4, 3)
                        if index_surface_nodes[1] and index_surface_nodes[2] and index_surface_nodes[4]:
                            add_surf(1, 2, 4)
                        if index_surface_nodes[1] and index_surface_nodes[3] and index_surface_nodes[2]:
                            add_surf(1, 3, 2)
        return elem_index, np.array(shell_elems), torch.stack(
            normal_vec, dim=0), torch.stack(center_point, dim=0)

if __name__ == '__main__':
    torch.set_default_device(torch.device('cuda'))
    torch.set_default_tensor_type(torch.DoubleTensor)

    fem = FEA_INP()
    fem.Read_INP(
        'Z:/RESULT/T20240123135421_/Cache/TopOptRun.inp'
    )
    fem.Read_INP(
        'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/TRO20230207Morph/20230411Revise/Result2/Bend0/FEA/c3d4.inp'
    )
    fem.Read_Result(
        'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/TRO20230207Morph/20230411Revise/Result2/Bend0/FEA/',
        ['c3d4.csv'])

    # fem.Get_Shape_Function(0)
    # fem.FEA_Opt()
    fem.FEA_Main()

    print('ok')
