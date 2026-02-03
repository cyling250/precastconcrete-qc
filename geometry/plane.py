import numpy as np


class BasePlane(object):
    def __init__(self, params: np.ndarray):
        # 平面模型参数（平面上任意一点 cx, cy, cx） + （平面法向量 nx, ny, nz）
        # 如果形状是(4,)则是ABCD，如果形状是(6,)则是cx,cy,cx,nx,ny,nz
        # 在生成时，使用UUID作为平面的唯一标识符
        import uuid

        self._id = uuid.uuid4()
        self.parameters: np.ndarray | None = None
        if type(params) is not np.ndarray:
            raise AttributeError("平面模型参数必须是numpy数组")
        else:
            if params.shape[0] == 4:
                # 转换为cx,cy,cz,nx,ny,nz格式
                # params是标准形式Ax+By+Cz+D=0
                n = params[0:3]  # 法向量(A,B,C)
                d = params[3]  # 常数项D

                # 找到平面上的一个点c
                # 选择x=0,y=0，计算z=(-D)/C（如果C不为0）
                if abs(n[2]) > 1e-6:
                    c = np.array([0, 0, -d / n[2]])
                # 选择x=0,z=0，计算y=(-D)/B（如果B不为0）
                elif abs(n[1]) > 1e-6:
                    c = np.array([0, -d / n[1], 0])
                # 选择y=0,z=0，计算x=(-D)/A（如果A不为0）
                else:
                    c = np.array([-d / n[0], 0, 0])

                self.parameters: np.ndarray | None = np.hstack((c, n))
            elif params.shape[0] == 6:
                self.parameters: np.ndarray | None = params
            else:
                raise ValueError("平面模型参数必须是4或6维数组")
        # 归一化平面法向量
        self.parameters[3:6] = self.parameters[3:6] / np.linalg.norm(
            self.parameters[3:6]
        )

    def __eq__(self, other) -> bool:
        """判断两个平面是否是同一个平面"""
        return self._id == other._id

    def __hash__(self):
        """返回平面模型的哈希值"""
        return hash(self._id)

    def get_standard(self):
        """获取平面的标准形式参数Ax+By+Cz+D=0

        Returns:
            np.ndarray: 标准形式参数，形状为 (4,)
        """
        c = self.parameters[0:3]
        n = self.parameters[3:6]
        d = -np.dot(c, n)
        return np.hstack((n, d))

    @staticmethod
    def read_from_npy(npy_path: str):
        """从npy文件读取平面模型参数

        Args:
            npy_path (str): npy文件路径
        """
        params = np.load(npy_path)
        return BasePlane(params)
