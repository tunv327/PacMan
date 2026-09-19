import random
import heapq
import math
# 0 = empty black rectangle, 1 = dot, 2 = big dot, 3 = vertical line,
# 4 = horizontal line, 5 = top right, 6 = top left, 7 = bot left, 8 = bot right
# 9 = gate
boards = [
[6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
[3, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
[3, 3, 1, 6, 4, 4, 5, 1, 6, 4, 4, 4, 5, 1, 3, 3, 1, 6, 4, 4, 4, 5, 1, 6, 4, 4, 5, 1, 3, 3],
[3, 3, 2, 3, 0, 0, 3, 1, 3, 0, 0, 0, 3, 1, 3, 3, 1, 3, 0, 0, 0, 3, 1, 3, 0, 0, 3, 2, 3, 3],#5
[3, 3, 1, 7, 4, 4, 8, 1, 7, 4, 4, 4, 8, 1, 7, 8, 1, 7, 4, 4, 4, 8, 1, 7, 4, 4, 8, 1, 3, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
[3, 3, 1, 6, 4, 4, 5, 1, 6, 5, 1, 6, 4, 4, 4, 4, 4, 4, 5, 1, 6, 5, 1, 6, 4, 4, 5, 1, 3, 3],
[3, 3, 1, 7, 4, 4, 8, 1, 3, 3, 1, 7, 4, 4, 5, 6, 4, 4, 8, 1, 3, 3, 1, 7, 4, 4, 8, 1, 3, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3],#10
[3, 7, 4, 4, 4, 4, 5, 1, 3, 7, 4, 4, 5, 0, 3, 3, 0, 6, 4, 4, 8, 3, 1, 6, 4, 4, 4, 4, 8, 3],
[3, 0, 0, 0, 0, 0, 3, 1, 3, 6, 4, 4, 8, 0, 7, 8, 0, 7, 4, 4, 5, 3, 1, 3, 0, 0, 0, 0, 0, 3],
[3, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 3],
[8, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 6, 4, 4, 9, 9, 4, 4, 5, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 7],
[4, 4, 4, 4, 4, 4, 8, 1, 7, 8, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 7, 8, 1, 7, 4, 4, 4, 4, 4, 4],#15
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
[4, 4, 4, 4, 4, 4, 5, 1, 6, 5, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 6, 5, 1, 6, 4, 4, 4, 4, 4, 4],
[5, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 7, 4, 4, 4, 4, 4, 4, 8, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 6],
[3, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 3],
[3, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 6, 4, 4, 4, 4, 4, 4, 5, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 3],#20
[3, 6, 4, 4, 4, 4, 8, 1, 7, 8, 0, 7, 4, 4, 5, 6, 4, 4, 8, 0, 7, 8, 1, 7, 4, 4, 4, 4, 5, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
[3, 3, 1, 6, 4, 4, 5, 1, 6, 4, 4, 4, 5, 1, 3, 3, 1, 6, 4, 4, 4, 5, 1, 6, 4, 4, 5, 1, 3, 3],
[3, 3, 1, 7, 4, 5, 3, 1, 7, 4, 4, 4, 8, 1, 7, 8, 1, 7, 4, 4, 4, 8, 1, 3, 6, 4, 8, 1, 3, 3],
[3, 3, 2, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 2, 3, 3],#25
[3, 7, 4, 5, 1, 3, 3, 1, 6, 5, 1, 6, 4, 4, 4, 4, 4, 4, 5, 1, 6, 5, 1, 3, 3, 1, 6, 4, 8, 3],
[3, 6, 4, 8, 1, 7, 8, 1, 3, 3, 1, 7, 4, 4, 5, 6, 4, 4, 8, 1, 3, 3, 1, 7, 8, 1, 7, 4, 5, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3],
[3, 3, 1, 6, 4, 4, 4, 4, 8, 7, 4, 4, 5, 1, 3, 3, 1, 6, 4, 4, 8, 7, 4, 4, 4, 4, 5, 1, 3, 3],
[3, 3, 1, 7, 4, 4, 4, 4, 4, 4, 4, 4, 8, 1, 7, 8, 1, 7, 4, 4, 4, 4, 4, 4, 4, 4, 8, 1, 3, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
[3, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 3],
[7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8]
         ]
def generate_single_dot_map(base_level):
    """Tạo bản đồ có đúng 1 dot tại 1 trong 4 vị trí ngẫu nhiên"""
    new_level = [row[:] for row in base_level]  # copy
    # Xóa toàn bộ dot cũ
    for i in range(len(new_level)):
        for j in range(len(new_level[i])):
            if new_level[i][j] in (1, 2):  # dot hoặc powerup
                new_level[i][j] = 0

    # Các vị trí có thể spawn dot
    positions = [
        (40, random.randint(60, 200)),   # bên trái
        (550, random.randint(60, 160)),  # bên phải
        (random.randint(200, 300), 490),   # phía trên
        (random.randint(60, 535), 600)   # phía dưới
    ]

    x, y = random.choice(positions)

    # Chuyển tọa độ pixel thành chỉ số hàng/cột
    row = y // ((690 - 50) // 32)
    col = x // (600 // 30)

    # Gán dot thường (1) hoặc power dot (2)
    new_level[row][col] = 1
    return new_level 
def find_dot_position(level):
    num1 = (690 - 50) // 32  # chiều cao mỗi ô
    num2 = 600 // 30         # chiều rộng mỗi ô

    for i, row in enumerate(level):       # i = hàng, row = list các cột
        for j, val in enumerate(row):     # j = cột, val = giá trị ô
            if val == 1 or val == 2:      # nếu là dot thường hoặc power dot
                dot_x = j * num2 + num2 // 2
                dot_y = i * num1 + num1 // 2
                return dot_x, dot_y
    return None, None  # nếu không tìm thấy
def heuristic(a, b):
    """Heuristic = khoảng cách Manhattan"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def is_valid(cell, level):
    """Kiểm tra ô có thể đi được (tránh tường, cửa box, nhà ma)"""
    x, y = cell
    if 0 <= y < len(level) and 0 <= x < len(level[0]):
        tile = level[y][x]
        # tile >=3 là tường, 9 là cửa box
        if tile >= 3 or tile == 9:
            return False
        # loại bỏ vùng “ghost box” (trung tâm bản đồ)
        if 8 < x < 21 and 12 < y < 20:
            return False
        return True
    return False

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos, level):
    x, y = pos
    moves = [(1,0),(-1,0),(0,1),(0,-1)]
    neighbors = []
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= ny < len(level) and 0 <= nx < len(level[0]):
            if level[ny][nx] in (0,1,2,9):
                neighbors.append((nx, ny))
    return neighbors

def ghost_penalty(x, y, ghosts, powerup=False):
    """
    Khi powerup = False → tránh ghost (phạt cao nếu gần).
    Khi powerup = True  → bỏ qua ghost (coi như dot, không phạt).
    """
    if powerup:
        return 0.0  # coi ma như dot, không phạt

    penalty = 0.0
    for g in ghosts:
        gx = int(g.x_pos // (600 // 30))
        gy = int(g.y_pos // ((690 - 50) // 32))

        # bỏ qua ghost đã chết hoặc trong box
        if getattr(g, "dead", False) or getattr(g, "in_box", False):
            continue

        dist = math.hypot(x - gx, y - gy)
        if dist < 1.5:
            penalty += 50.0
        elif dist < 3.0:
            penalty += 12.0
        elif dist < 5.0:
            penalty += 4.0
    return penalty

def astar_avoid_ghost(level, start, goal, ghosts, powerup):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for neighbor in get_neighbors(current, level):
            base_cost = 1
            gx, gy = neighbor
            ghost_cost = ghost_penalty(gx, gy, ghosts, powerup)
            tentative_g = g_score[current] + base_cost + ghost_cost
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

def get_direction_from_path(current, next_step):
    cx, cy = current
    nx, ny = next_step
    if nx > cx:
        return 0   # phải
    elif nx < cx:
        return 1   # trái
    elif ny < cy:
        return 2   # lên
    elif ny > cy:
        return 3   # xuống
    return 0
