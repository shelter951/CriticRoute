import torch
import numpy as np
from collections import defaultdict

MAX_DIST = 30
MAX_STEP = 10


def calc_position_distance(a, b):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return dist


def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    xy_dist = max(np.sqrt(dx ** 2 + dy ** 2), 1e-8)
    xyz_dist = max(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2), 1e-8)

    # the simulator's api is weired (x-y axis is transposed)
    heading = np.arcsin(dx / xy_dist)  # [-pi/2, pi/2]
    if b[1] < a[1]:
        heading = np.pi - heading
    heading -= base_heading

    elevation = np.arcsin(dz / xyz_dist)  # [-pi/2, pi/2]
    elevation -= base_elevation

    return heading, elevation, xyz_dist


def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts


class FloydGraph(object):
    def __init__(self):
        self._dis = defaultdict(lambda: defaultdict(lambda: 95959595))
        self._point = defaultdict(lambda: defaultdict(lambda: ""))
        self._visited = set()

    def distance(self, x, y):
        if x == y:
            return 0
        else:
            return self._dis[x][y]

    def add_edge(self, x, y, dis):
        if dis < self._dis[x][y]:
            self._dis[x][y] = dis
            self._dis[y][x] = dis
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k):
        for x in self._dis:
            for y in self._dis:
                if x != y:
                    if self._dis[x][k] + self._dis[k][y] < self._dis[x][y]:
                        self._dis[x][y] = self._dis[x][k] + self._dis[k][y]
                        self._dis[y][x] = self._dis[x][y]
                        self._point[x][y] = k
                        self._point[y][x] = k
        self._visited.add(k)

    def visited(self, k):
        return (k in self._visited)

    def path(self, x, y):
        """
        :param x: start
        :param y: end
        :return: the path from x to y [v1, v2, ..., v_n, y]
        """
        if x == y:
            return []
        if self._point[x][y] == "":  # Direct edge
            return [y]
        else:
            k = self._point[x][y]
            # print(x, y, k)
            # for x1 in (x, k, y):
            #     for x2 in (x, k, y):
            #         print(x1, x2, "%.4f" % self._dis[x1][x2])
            return self.path(x, k) + self.path(k, y)


class GraphMap(object):
    def __init__(self, start_vp):
        self.start_vp = start_vp  # start viewpoint

        self.node_positions = {}  # viewpoint to position (x, y, z)
        self.graph = FloydGraph()  # shortest path graph
        self.node_embeds = {}  # {viewpoint: feature (sum feature, count)}
        self.node_stop_scores = {}  # {viewpoint: prob}
        self.node_nav_scores = {}  # {viewpoint: {t: prob}}
        self.node_step_ids = {}
        self.pooling_mode = 'mean'

    def update_graph(self, ob):
        self.node_positions[ob['viewpoint']] = ob['position']
        for cc in ob['candidate']:
            self.node_positions[cc['viewpointId']] = cc['position']
            dist = calc_position_distance(ob['position'], cc['position'])
            self.graph.add_edge(ob['viewpoint'], cc['viewpointId'], dist)
        self.graph.update(ob['viewpoint'])

    def update_node_embed(self, vp, embed, rewrite=False):
        if rewrite:
            self.node_embeds[vp] = [embed, 1]
        else:
            if vp in self.node_embeds:
                if self.pooling_mode == "max":
                    pooling_features, _ = torch.max(torch.stack([self.node_embeds[vp][0], embed.clone()]), dim=0)
                    self.node_embeds[vp][0] = pooling_features
                elif self.pooling_mode == "mean":
                    self.node_embeds[vp][0] += embed.clone()
                else:
                    raise NotImplementedError('`pooling_mode` Only support ["mean", "max"]')
                self.node_embeds[vp][1] += 1
            else:
                self.node_embeds[vp] = [embed, 1]


    def get_node_embed(self, vp):
        """
        获取节点的embedding。如果节点还没有embedding，返回零向量。
        这样可以避免在nav_gmap_variable中访问未初始化节点的KeyError。
        
        ⚠️ 关键修复：确保总是返回有效的embedding，即使节点还没有初始化。
        这解决了update_graph添加新节点到node_positions，但还没有初始化embedding的问题。
        """
        if vp not in self.node_embeds:
            # 如果节点还没有embedding，尝试从已有embedding推断维度
            if len(self.node_embeds) > 0:
                # 使用第一个已有embedding的维度
                sample_embed = next(iter(self.node_embeds.values()))[0]
                # 返回零向量（与原embedding相同的设备和dtype）
                zero_embed = torch.zeros_like(sample_embed)
                # 同时初始化这个节点的embedding，避免下次再查询时重复计算
                self.node_embeds[vp] = [zero_embed, 1]
                return zero_embed
            else:
                # 如果没有任何embedding，这是一个严重错误
                # 但为了健壮性，我们尝试从node_positions推断（虽然这不应该发生）
                raise ValueError(
                    f"No embeddings available to infer dimension for viewpoint {vp}. "
                    f"This usually means the graph was not properly initialized. "
                    f"Total nodes in node_positions: {len(self.node_positions)}, "
                    f"Total nodes with embeddings: {len(self.node_embeds)}"
                )
        
        if self.pooling_mode == "max":
            return self.node_embeds[vp][0]
        elif self.pooling_mode == "mean":
            return self.node_embeds[vp][0] / self.node_embeds[vp][1]
        else:
            raise NotImplementedError('`pooling_mode` Only support ["mean", "max"]')

    def get_pos_fts(self, cur_vp, gmap_vpids, cur_heading, cur_elevation, angle_feat_size=4):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.node_positions[cur_vp], self.node_positions[vp],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.graph.distance(cur_vp, vp) / MAX_DIST, \
                     len(self.graph.path(cur_vp, vp)) / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)

    def save_to_json(self):
        nodes = {}
        for vp, pos in self.node_positions.items():
            nodes[vp] = {
                'location': pos,  # (x, y, z)
                'visited': self.graph.visited(vp),
            }
            if nodes[vp]['visited']:
                nodes[vp]['stop_prob'] = self.node_stop_scores[vp]['stop']
                nodes[vp]['og_objid'] = self.node_stop_scores[vp]['og']
            else:
                nodes[vp]['nav_prob'] = self.node_nav_scores[vp]

        edges = []
        for k, v in self.graph._dis.items():
            for kk in v.keys():
                edges.append((k, kk))

        return {'nodes': nodes, 'edges': edges}


