from commonroad.common.file_reader import *
from commonroad.scenario.laneletcomplement import *
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork, LineMarking

import numpy as np

class LaneletCurveNetworkReader:
    def __init__(self, textfile):
        f = open(textfile, 'r')
        self.lines = f.readlines()
        self.line_index = 0
        if 'ROADWAY' in self.lines[self.line_index]:
            self.line_index += 1

        lanelets = list()
        nsegs = int(self.advance())

        for i_seg in range(nsegs):
            segid = int(self.advance())
            nlanes = int(self.advance())
            for i_lane in range(nlanes):
                assert i_lane+1 == int(self.advance())
                lane_id = (segid)*100 + i_lane+1
                width = float(self.advance())
                #width = 3.6
                tokens = self.advance().split(' ')
                speed_limit = float(tokens[1])

                tokens = self.advance().split(' ')
                line_marking_left_vertices = LineMarking.DASHED if tokens[0] == "broken" else LineMarking.SOLID

                tokens = self.advance().split(' ')
                line_marking_right_vertices = LineMarking.DASHED if tokens[0] == "broken" else LineMarking.SOLID

                adjacent_left, adjacent_left_same_direction = None, None
                #print("ID", lane_id)
                #print(line_marking_left_vertices, "  ", i_lane, "  ", nlanes-1)
                #print(line_marking_right_vertices, "  ", i_lane, "  ", nlanes)
                if line_marking_left_vertices == LineMarking.DASHED and i_lane < nlanes-1:
                    adjacent_left = lane_id + 1
                    adjacent_left_same_direction = True

                adjacent_right, adjacent_right_same_direction = None, None
                if line_marking_right_vertices == LineMarking.DASHED and i_lane > 0:
                    adjacent_right = lane_id  - 1
                    adjacent_right_same_direction = True

                n_conns = int(self.advance())
                predecessor = []
                successor = []
                predecessor_connections = {}
                successor_connections = {}
                for i_conn in range(n_conns):
                    tokens = self.advance()
                    tokens = tokens.replace('(', '')
                    tokens = tokens.replace(')', '')
                    tokens = tokens.split(' ')
                    if tokens[0] == "D":
                        mylane = CurveIndex(int(tokens[1])-1, float(tokens[2]))
                        target = CurveIndex(int(tokens[3])-1, float(tokens[4]))
                        targetid = int(tokens[5])*100 + int(tokens[6])
                        successor.append(targetid)
                        successor_connections[targetid] = (mylane, target)
                    else:
                        mylane = CurveIndex(int(tokens[1])-1, float(tokens[2]))
                        target = CurveIndex(int(tokens[3])-1, float(tokens[4]))
                        targetid = int(tokens[5])*100 + int(tokens[6])
                        predecessor.append(targetid)
                        predecessor_connections[targetid] = (mylane, target)
                npts = int(self.advance())

                left_vertices = []
                right_vertices = []
                center_vertices = []

                curve = []
                for i_pt in range(npts):
                    line = self.advance()
                    cleanedline = line.replace('(', '')
                    cleanedline = cleanedline.replace(')', '')
                    tokens = cleanedline.split(' ')
                    x = float(tokens[0])
                    y = float(tokens[1])
                    th = float(tokens[2])
                    s = float(tokens[3])
                    k = float(tokens[4])
                    kd = float(tokens[5])
                    if textfile == "ngsim_80.txt" and lane_id == 401 and i_pt > 112:
                        if i_pt < 126:
                            continue
                        else:
                            s -= 3.6542
                    curve.append(CurvePt(VecSE2(x,y,th), s, k, kd))
                    center_vertices.append(np.array([x, y]))


                    left_vertices.append(np.array([x + width*0.5*np.cos(th + 0.5*np.pi), y + width*0.5*np.sin(th + 0.5*np.pi)]))

                    right_vertices.append(np.array([x - width*0.5*np.cos(th + 0.5*np.pi), y - width*0.5*np.sin(th + 0.5*np.pi)]))

                #print(lane_id, " left ", adjacent_left,  " right ", adjacent_right)
                lanelet = LaneLetCurve(
                       left_vertices=np.array(left_vertices), center_vertices=np.array(center_vertices), right_vertices=np.array(right_vertices), center_curve=curve,
                       lanelet_id=lane_id,
                       predecessor=predecessor, predecessor_connections=predecessor_connections, successor=successor, successor_connections=successor_connections,
                       adjacent_left=adjacent_left, adjacent_left_same_direction=adjacent_left_same_direction,
                       adjacent_right=adjacent_right, adjacent_right_same_direction=adjacent_right_same_direction,
                       speed_limit=speed_limit,
                       line_marking_left_vertices=line_marking_left_vertices,
                       line_marking_right_vertices=line_marking_right_vertices)
                lanelets.append(lanelet)
        self.lanelet_network = LaneletNetwork.create_from_lanelet_list(lanelets)

        for lanelet in self.lanelet_network.lanelets:
            lane_id = lanelet.lanelet_id
            successor = lanelet.successor
            for s_id in successor:
                s_lanelet = self.lanelet_network.find_lanelet_by_id(s_id)
                con = lanelet.successor_connections[s_id]
                if con[0].i == len(lanelet.center_curve)-2:
                    delta_x = lanelet.center_curve[-1].pos.x-s_lanelet.center_curve[con[1].i].pos.x
                    delta_y = lanelet.center_curve[-1].pos.y-s_lanelet.center_curve[con[1].i].pos.y
                    dist = np.hypot(delta_x, delta_y)
                    new_x = s_lanelet.center_curve[con[1].i].pos.x
                    new_y = s_lanelet.center_curve[con[1].i].pos.y
                    new_th = s_lanelet.center_curve[con[1].i].pos.th
                    new_pt = CurvePt(VecSE2(new_x,
                                            new_y,
                                            new_th), lanelet.center_curve[-1].s+dist,
                                      s_lanelet.center_curve[con[1].i].k,
                                      s_lanelet.center_curve[con[1].i].kd)
                    lanelet.center_curve.append(new_pt)
                    lanelet._center_vertices = np.concatenate((lanelet.center_vertices, np.array([[new_x,new_y]])), axis=0)
                    lanelet._left_vertices = np.concatenate((lanelet.left_vertices, [s_lanelet.left_vertices[con[1].i]]), axis=0)
                    lanelet._right_vertices = np.concatenate((lanelet.right_vertices, [s_lanelet.right_vertices[con[1].i]]), axis=0)


        f.close()

    def advance(self):
        line = self.lines[self.line_index].strip()
        self.line_index += 1
        return line
