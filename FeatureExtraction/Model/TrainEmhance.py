import torch
from torch_geometric.data import Data
from collections import defaultdict
from itertools import combinations

class trainEmb:

    def combData(self, dataset, seed, threshold):
        torch.manual_seed(seed)
        yToInd = defaultdict(list)
        for idx, data in enumerate(dataset):
            y = data.y.item()
            yToInd[y].append(idx)

        counts = {y: len(indices) for y, indices in yToInd.items()}
        if not counts:
            raise ValueError('数据集为空')
        maxCount = max(counts.values())

        newDataset = []
        for y, indices in yToInd.items():
            print(f'平衡类别{y}中......')
            cnt = len(indices)
            if cnt == maxCount:
                newDataset.extend(dataset[i] for i in indices)
            else:
                need = int((maxCount - cnt) * threshold)

                totalComb = pow(2, cnt) - cnt - 1
                if need > totalComb:
                    raise ValueError(
                        f'类别{y}样本数{cnt}下，需要生成{need}个新样本，但只能生成{totalComb}个'
                    )
                k = 2
                while need > 0 and k <= cnt:
                    print(f'{k}图相互组合中')
                    combs = list(combinations(indices, k))
                    maxComb = len(combs)
                    toGenerate = min(need, maxComb)
                    for i in range(toGenerate):
                        combo = combs[i]
                        combData = [dataset[j] for j in combo]
                        newData = self.combine(combData)
                        newDataset.append(newData)
                    need -= toGenerate
                    k += 1
                newDataset.extend([dataset[i] for i in indices])
        return newDataset

    def combine(self, graphs):
        if not graphs:
            raise ValueError('空列表的组合')
        x = torch.cat([g.x for g in graphs])
        edgeIndexs = []
        offset = 0
        for g in graphs:
            edgeIndexs.append(g.edge_index + offset)
            offset += g.x.size(0)
        edge_index = torch.cat(edgeIndexs, dim=1)

        y = graphs[0].y

        edge_attr = None
        edge_type = None
        if all(hasattr(g, 'edge_attr') for g in graphs):
            edge_attr = torch.cat([g.edge_attr for g in graphs], dim=0)
        if all(hasattr(g, 'edge_type') for g in graphs):
            edge_type = torch.cat([g.edge_type for g in graphs], dim=0)
        newData = Data(x=x, edge_index=edge_index, y=y)
        if edge_type is not None:
            newData.edge_type = edge_type
        else:
            raise ValueError('边类型为空')
        if edge_attr is not None:
            newData.edge_attr = edge_attr
        else:
            raise ValueError('边特征为空')
        return newData