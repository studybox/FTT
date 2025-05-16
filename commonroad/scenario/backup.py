import math
import numpy as np
# add frenet state to the original state




def proj_on_line(a, b):
    # dot(a,b) / dot(b,b) ⋅ b
    s = (a.x*b.x + a.y*b.y) / (b.x*b.x + b.y*b.y)
    return VecE2(s*b.x, s*b.y)
     
def get_lerp_time_unclamped(A, B, Q):
    a = Q - A
    b = B - A
    c = proj_on_line(a, b)
    if b.x != 0.0:
        t = c.x / b.x
    elif b.y != 0.0:
        t = c.y / b.y 
    else:
        t = 0.0
    return t 
    
def get_lerp_time(A, B, Q):
    return np.clip(get_lerp_time_unclamped(A, B, Q), 0.0, 1.0)

def lerp_angle(a, b, t):
    a + np.arctan(np.sin(b-a), np.cos(b-a))*t
    return a
    
def lerp_pos(a, b, t):
    x = a.x + (b.x-a.x)*t
    y = a.y + (b.y-a.y)*t
    th = lerp_angle(a.th, b.th, t)
    return VecSE2(x,y,th)

def lerp_curve(a, b, t):
    return CurvePt(lerp_pos(a.pos, b.pos, t), a.s + (b.s - a.s)*t, a.k + (b.k - a.k)*t, a.kd + (b.kd - a.kd)*t)    
    
def inertial2body(point, reference):
    s, c = np.sin(reference.th), np.cos(reference.th)
    deltax = point.x - reference.x 
    deltay = point.y - reference.y
    return VecE2(c*deltax + s*deltay, c*deltay - s*deltax)    
    
def get_curve_projection(posG, footpoint, curveind):
    F = inertial2body(posG, footpoint)
    return CurveProjection(curveind, F.y, F.th)
    
def proj_on_curve(posG, curve):
    ind = index_closest_to_point(curve, posG)
    
    curveind = None
    footpoint = None
    
    if ind > 0 and ind < len(curve)-1:
        t_lo = get_lerp_time(curve[ind-1].pos, curve[ind].pos, posG)
        t_hi = get_lerp_time(curve[ind].pos, curve[ind+1].pos, posG)
        
        p_lo = lerp_curve(curve[ind-1].pos, curve[ind].pos, t_lo)
        p_hi = lerp_curve(curve[ind].pos, curve[ind+1].pos, t_hi)
        
        d_lo = np.hypot(p_lo - posG)
        d_hi = np.hypot(p_hi - posG)
        
        if d_lo < d_hi:
            footpoint = p_lo
            curveind = CurveIndex(ind-1, t_lo)
        else:
            footpoint = p_hi
            curveind = CurveIndex(ind, t_hi)
    elif ind == 0:
        t = get_lerp_time(curve[0], curve[1], posG)
        footpoint = lerp_curve( curve[0].pos, curve[1].pos, t)
        curveind = CurveIndex(ind, t)
    else: # ind == len(curve)-1
        t = get_lerp_time( curve[-2], curve[-1], posG )
        footpoint = lerp_curve( curve[-2].pos, curve[-1].pos, t)
        curveind = CurveIndex(ind-1, t)
    return get_curve_projection(posG, footpoint, curveind)

def index_closest_to_point(curve, target):
    
    a = 0
    b = len(curve)-1
    c = (a+b)//2
    
    sqdist_a = np.hypot(curve[a].pos[0] - target[0], curve[a].pos[1] - target[1])
    sqdist_b = np.hypot(curve[b].pos[0] - target[0], curve[b].pos[1] - target[1])
    sqdist_c = np.hypot(curve[c].pos[0] - target[0], curve[c].pos[1] - target[1])
    
    while True:
        if b == a:
            return a
        elif b == a + 1:
            return b if sqdist_b < sqdist_a else a
        elif a + 1 == c and b - 1 == c:
            if sqdist_a < sqdist_b and sqdist_a < sqdist_c:
                return a
            elif sqdist_b < sqdist_a and sqdist_b < sqdist_c:
                return b
            else:
                return c
        left = (a+c)//2
        sqdist_l = np.hypot(curve[left].pos[0] - target[0], curve[left].pos[1] - target[1])
        
        right = (c+b)//2
        sqdist_r = np.hypot(curve[right].pos[0] - target[0], curve[right].pos[1] - target[1])
        
        if sqdist_l < sqdist_r:
            b = c
            sqdist_b = sqdist_c
            c = left 
            sqdist_c = sqdist_l
        else:
            a = c
            sqdist_a = sqdist_c
            c = right 
            sqdist_c = sqdist_r
            
               
        

def proj(posG, lanelet, lanelet_network, move_along_curves=True):
    curveproj = proj_on_curve(posG, lanelet.center_vertices)
    retid = lanelet.lanelet_id
    # TODO Check this 
    #if curveproj.ind.is_start() and len(lanelet.predecessor)>0 :
    #elif curveproj.ind.is_end() and len(lanelet.successor)>0 :
    return RoadProjection(curveproj, retid)
    
    
def projF(posG, lanelet_network):
    # TODO
    best_dist2 = float('inf')
    best_proj = RoadProjection(CurveProjection(CurveIndex(-1, -1.0),
                                               float('nan'), float('nan')), -1)
    for lanelet in lanelet_network.lanelets:
        roadproj = proj(posG, lanelet, lanelet_network, move_along_curves=False)
        targetlane = lanelet_network[roadproj.lanelet_id]
        footpoint = 
        dist2 =  normsquared(VecE2(posG - footpoint.pos))
        if dist2 < best_dist2:
            best_dist2 = dist2 
            best_proj = roadproj
    return best_proj
