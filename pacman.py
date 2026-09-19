#build pacMan
from data_train import log_training_auto
import numpy as np
import torch
from board import boards, generate_single_dot_map ,find_dot_position ,astar_avoid_ghost ,get_direction_from_path 
import pygame
import math
import copy
from pacman_ai import DQNAgent, get_state 
import random 
pygame.init()
#thêm cho pacman tự di chuyển 
state_dim = 38 # PacMan + 4 ghosts + powerup 
action_dim = 4 # phải, trái, lên, xuống 
agent = DQNAgent(state_dim, action_dim)
agent.load("pacman_dqn.pth")



WIDTH=600#chiều rộng
HEIGHT=690#chiều cao
screen=pygame.display.set_mode((WIDTH,HEIGHT))
timer=pygame.time.Clock()
fps=60
font =pygame.font.Font('freesansbold.ttf',20)
level = copy.deepcopy(boards)
level_initial=copy.deepcopy(level)
dot_x, dot_y = find_dot_position(level)
color='blue'
PI=math.pi
player_images=[]
for i in range(1,5):
    player_images.append(pygame.transform.scale(pygame.image.load(f'assets/{i}.png'),(30,30)))
red_img=pygame.transform.scale(pygame.image.load(f'assetGhots/red.png'),(30,30))
pink_img=pygame.transform.scale(pygame.image.load(f'assetGhots/pink.png'),(30,30))
blue_img=pygame.transform.scale(pygame.image.load(f'assetGhots/blue.png'),(30,30))
orange_img=pygame.transform.scale(pygame.image.load(f'assetGhots/orange.png'),(30,30))
powerup_img=pygame.transform.scale(pygame.image.load(f'assetGhots/powerup.png'),(30,30))
dead_img=pygame.transform.scale(pygame.image.load(f'assetGhots/dead.png'),(30,30))
player_x=279#vị trí ban đầu của pacman
player_y=354#vị trí ban đầu của pacman
# Vị trí của Ghots:
#vị trí của red
red_x=260
red_y=280
red_direction=0
red_dead=False
red_box=False
#vị trí của pink
pink_x=338
pink_y=273
pink_direction=2
pink_dead=False
pink_box=False
#vị trí của blue
blue_x=233
blue_y=320
blue_direction=2
blue_dead=False
blue_box=False
#vị trí của orange
orange_x=338
orange_y=320
orange_direction=2
orange_dead=False
orange_box=False

direction=0
counter=0
flicker = False#biến nhấp nháy
turns_allowed = [False,False,False,False]#right,left,up,down
direction_command=0
player_speed=2
ghot_speed=[2,2,2,2]
score= float("0")
powerup=False
power_counter=0
eaten_ghost=[False,False,False,False]
targets=[(player_x,player_y),(player_x,player_y),(player_x,player_y),(player_x,player_y)]
moving =False
startup_coutnter=0
lives = 3
game_over=False
game_won=False
step=0
reward =0
prev_score=score
prev_live=lives
prev_playerx = player_x
prev_playery = player_y
stuck_counter =0
prev_action =None
min_dist = float("inf")
nearest_dot = None
best_reward = float("-inf")
best_score = -float("inf")
episode =  0
current_goal = None

class Ghosts:
    def __init__(self,x_coord,y_coord,target,speed,img,direct,dead,box,id):
        self.x_pos=x_coord
        self.y_pos=y_coord
        self.center_x=self.x_pos+15
        self.center_y=self.y_pos+15
        self.target=target
        self.speed=speed
        self.img=img
        self.direction=direct
        self.dead=dead
        self.in_box=box
        self.id=id
        self.turns,self.in_box=self.check_position()
        self.rect=self.draw()
        
    def draw(self):
        if (not powerup and not self.dead)or(eaten_ghost[self.id]and powerup and not self.dead):
            screen.blit(self.img,(self.x_pos,self.y_pos))
        elif powerup and not self.dead and not eaten_ghost[self.id]:
            screen.blit(powerup_img,(self.x_pos,self.y_pos))
        else:
            screen.blit(dead_img,(self.x_pos,self.y_pos))
        ghost_rect=pygame.rect.Rect((self.center_x-15,self.center_y-15),(30,30))
        return ghost_rect
    
    def check_position(self):
        turns = [False, False, False, False]  # [Right, Left, Up, Down]
        num1 = (HEIGHT - 50) // 32  # chiều cao mỗi ô
        num2 = WIDTH // 30          # chiều rộng mỗi ô
        num3 = 10                   # khoảng kiểm tra “sát tường”

    # --- Cập nhật tâm hiện tại ---
        self.center_x = self.x_pos + 15
        self.center_y = self.y_pos + 15
        row = int(self.center_y // num1)
        col = int(self.center_x // num2)

    # --- Cập nhật trạng thái trong lồng (trước khi check can_move) ---
        if 233 < self.x_pos < 358 and 250 < self.y_pos < 340:
            self.in_box = True           
        else:
            self.in_box = False

    # --- Hàm phụ kiểm tra xem ô có thể đi qua không ---
        def can_move(row, col):
            if 0 <= row < len(level) and 0 <= col < len(level[row]):
                tile = level[row][col]
            # <3 = đường hợp lệ
            # ==9 = cửa lồng (cho phép nếu ghost đang trong lồng hoặc đã chết)
                if tile < 3:
                    return True
                if tile == 9 and (self.in_box or self.dead):
                    return True
            return False

    # --- Kiểm tra 4 hướng cơ bản ---
        if can_move(row, (self.center_x + num3) // num2):  # RIGHT
            turns[0] = True
        if can_move(row, (self.center_x - num3) // num2):  # LEFT
            turns[1] = True
        if can_move((self.center_y - num3) // num1, col):  # UP
            turns[2] = True
        if can_move((self.center_y + num3) // num1, col):  # DOWN
            turns[3] = True

    # --- Cho phép rẽ mượt ---
        if self.direction in (2, 3):  # đi dọc
            if 8 <= self.center_x % num2 <= 15:
                if can_move((self.center_y + num3) // num1, col):
                    turns[3] = True
                if can_move((self.center_y - num3) // num1, col):
                    turns[2] = True
            if 8 <= self.center_y % num1 <= 15:
                if can_move(row, (self.center_x - num2) // num2):
                    turns[1] = True
                if can_move(row, (self.center_x + num2) // num2):
                    turns[0] = True

        if self.direction in (0, 1):  # đi ngang
            if 10 <= self.center_x % num2 <= 15:
                if can_move((self.center_y + num1) // num1, col):
                    turns[3] = True
                if can_move((self.center_y - num1) // num1, col):
                    turns[2] = True
            if 8 <= self.center_y % num1 <= 15:
                if can_move(row, (self.center_x - num3) // num2):
                    turns[1] = True
                if can_move(row, (self.center_x + num3) // num2):
                    turns[0] = True

    # --- Warp tunnel ---
        if self.center_x // 30 >= 29:
            turns[0] = True
            turns[1] = True

        self.turns = turns
        return self.turns, self.in_box

    def move_generic(self):
        if self.direction ==0:
            if self.target[0]>self.x_pos and self.turns[0]:
                self.x_pos+=self.speed
            elif not self.turns[0]:
                if self.target[1]>self.y_pos and self.turns[3]:
                    self.direction=3
                    self.y_pos+=self.speed
                elif self.target[1]<self.y_pos and self.turns[2]:
                    self.direction=2
                    self.y_pos-=self.speed
                elif self.target[0]<self.x_pos and self.turns[1]:
                    self.direction=1
                    self.x_pos-=self.speed
                elif self.turns[3]:
                    self.direction=3
                    self.y_pos+=self.speed
                elif self.turns[2]:
                    self.direction=2
                    self.y_pos-=self.speed
                elif self.turns[1]:
                    self.direction=1
                    self.x_pos-=self.speed
            elif self.turns[0]:
                if self.target[1]>self.y_pos and self.turns[3]:
                    self.direction=3
                    self.y_pos+=self.speed
                if self.target[1]<self.y_pos and self.turns[2]:
                    self.direction=2
                    self.y_pos-=self.speed
                else:
                    self.x_pos+=self.speed        
        elif self.direction ==1:
            if self.target[1]>self.y_pos and self.turns[3]:
                self.direction=3
            elif self.target[0]<self.x_pos and self.turns[1]:
                self.x_pos-=self.speed
            elif not self.turns[1]:
                if self.target[1]>self.y_pos and self.turns[3]:
                    self.direction=3
                    self.y_pos+=self.speed
                elif self.target[1]<self.y_pos and self.turns[2]:
                    self.direction=2
                    self.y_pos-=self.speed
                elif self.target[0]>self.x_pos and self.turns[0]:
                    self.direction=0
                    self.x_pos+=self.speed
                elif self.turns[3]:
                    self.direction=3
                    self.y_pos+=self.speed
                elif self.turns[2]:
                    self.direction=2
                    self.y_pos-=self.speed
                elif self.turns[0]:
                    self.direction=0
                    self.x_pos+=self.speed
            elif self.turns[1]:
                if self.target[1]>self.y_pos and self.turns[3]:
                    self.direction=3
                    self.y_pos+=self.speed
                if self.target[1]<self.y_pos and self.turns[2]:
                    self.direction=2
                    self.y_pos-=self.speed
                else:
                    self.x_pos-=self.speed                    
        elif self.direction ==2:
            if self.target[0]<self.x_pos and self.turns[1]:
                self.direction=1
                self.x_pos-=self.speed
            elif self.target[1]<self.y_pos and self.turns[2]:
                self.direction=2
                self.y_pos-=self.speed
            elif not self.turns[2]:
                if self.target[0]>self.x_pos and self.turns[0]:
                    self.direction=0
                    self.x_pos+=self.speed
                elif self.target[0]<self.x_pos and self.turns[1]:
                    self.direction=1
                    self.x_pos-=self.speed
                elif self.target[1]>self.y_pos and self.turns[3]:
                    self.direction=3
                    self.y_pos+=self.speed
                elif self.turns[1]:
                    self.direction=1
                    self.x_pos-=self.speed
                elif self.turns[3]:
                    self.direction=3
                    self.y_pos+=self.speed               
                elif self.turns[0]:
                    self.direction=0
                    self.x_pos+=self.speed
            elif self.turns[2]:
                if self.target[0]>self.x_pos and self.turns[0]:
                    self.direction=0
                    self.x_pos+=self.speed
                elif self.target[0]<self.x_pos and self.turns[1]:
                    self.direction=1
                    self.x_pos-=self.speed
                else:
                    self.y_pos-=self.speed        
        elif self.direction ==3:
            if self.target[1]>self.y_pos and self.turns[3]:
                self.y_pos+=self.speed
            elif not self.turns[3]:
                if self.target[0]>self.x_pos and self.turns[0]:
                    self.direction=0
                    self.x_pos+=self.speed
                elif self.target[0]<self.x_pos and self.turns[1]:
                    self.direction=1
                    self.x_pos-=self.speed 
                elif self.target[1]<self.y_pos and self.turns[2]:
                    self.direction=2
                    self.y_pos-=self.speed   
                elif self.turns[2]:
                    self.direction=2
                    self.y_pos-=self.speed
                elif self.turns[1]:
                    self.direction=1
                    self.x_pos-=self.speed
                elif self.turns[0]:
                    self.direction=0
                    self.x_pos+=self.speed            
            elif self.turns[3]:
                if self.target[0]>self.x_pos and self.turns[0]:
                    self.direction=0
                    self.x_pos+=self.speed
                elif self.target[0]<self.x_pos and self.turns[1]:
                    self.direction=1
                    self.x_pos-=self.speed
                else:
                    self.y_pos+=self.speed
        if self.x_pos <-15:
            self.x_pos=WIDTH-15
        elif self.x_pos >WIDTH-15:
            self.x_pos=-15
        return self.x_pos,self.y_pos,self.direction


        
                   
def draw_board():
    num1 =((HEIGHT-50)//32)#chiều cao của mỗi ô =20
    num2 =(WIDTH//30)#chiều rộng của mỗi ô =20
    for i in range(len(level)):
       for j in range(len(level[i])):
           if level[i][j]==1:
               pygame.draw.circle(screen,'white',(j*num2+(0.5*num2),i*num1+(0.5*num1)),2)
           if level[i][j]==2 and not flicker:
               pygame.draw.circle(screen,'white',(j*num2+(0.5*num2),i*num1+(0.5*num1)),5)
           if level[i][j]==3:    
               pygame.draw.line(screen,color,(j*num2+(0.5*num2),i*num1),
                                (j*num2+(.5*num2),i*num1+num1),2) 
           if level[i][j]==4:    
               pygame.draw.line(screen,color,(j*num2,i*num1+(0.5*num1)),
                                (j*num2+num2,i*num1+(0.5*num1)),2) 
           if level[i][j]==5:
               pygame.draw.arc(screen,color,[(j*num2-(num2*0.4))-2,(i*num1+(0.5*num1)),num2,num1],0,PI/2,2)
           if level[i][j]==6:
               pygame.draw.arc(screen,color,[(j*num2+(num2*0.5)),(i*num1+(0.5*num1)),num2,num1],PI/2,PI,2)
           if level[i][j]==7:
               pygame.draw.arc(screen,color,[(j*num2+(num2*0.5)),(i*num1-(0.4*num1)),num2,num1],PI,3*PI/2,2)
           if level[i][j]==8:
               pygame.draw.arc(screen,color,[(j*num2-(num2*0.4)-2),(i*num1-(0.4*num1)),num2,num1],3*PI/2,2*PI,2)
               
           if level[i][j]==9:    
               pygame.draw.line(screen,'white',(j*num2,i*num1+(0.5*num1)),
                                (j*num2+num2,i*num1+(0.5*num1)),2)                   

def draw_player():
    #0_right,1_left,2_up,3_down
    if direction == 0:
        screen.blit(player_images[counter//5], (player_x,player_y))
    elif direction == 1:
        screen.blit(pygame.transform.flip(player_images[counter//5],True,False), (player_x,player_y))
    elif direction == 2:
        screen.blit(pygame.transform.rotate(player_images[counter//5],90), (player_x,player_y))
    elif direction == 3:
        screen.blit(pygame.transform.rotate(player_images[counter//5],270), (player_x,player_y))

def draw_misc():
    score_text=font.render(f'Score: {score}',True,'white')
    screen.blit(score_text,(10,HEIGHT-20))
    if powerup:
       pygame.draw.circle(screen,color,(550,HEIGHT-20),10) 
    for i in range(lives):
        screen.blit(pygame.transform.scale(player_images[0],(20,20)),(400+i*30,HEIGHT-30))
    if game_over:
        pygame.draw.rect(screen,'white',[192,233,218,157],0,10)
        pygame.draw.rect(screen,'dark gray',[202,243,198,137],0,10)
        gameover_text=font.render('Game over!',True,'red')           
        screen.blit(gameover_text,(242,300))
    if game_won:
        pygame.draw.rect(screen,'white',[192,233,218,157],0,10)
        pygame.draw.rect(screen,'dark gray',[202,243,198,137],0,10)
        gameover_text=font.render('Victory!',True,'red')           
        screen.blit(gameover_text,(262,300))
    
def check_position(centerx, centery):
    turns = [False, False, False, False]
    num1 = (HEIGHT - 50) // 32  # chiều cao mỗi ô
    num2 = WIDTH // 30          # chiều rộng mỗi ô
    num3 = 10                   # khoảng kiểm tra “sát tường”

    def can_move(row, col):
        if 0 <= row < len(level) and 0 <= col < len(level[row]):
            return level[row][col] < 3
        return False

    row = centery // num1
    col = centerx // num2

    # Luôn cho phép đi qua đường hầm
    if col >= 29:
        turns[0] = True
        turns[1] = True
        return turns

    # Kiểm tra 4 hướng cơ bản
    if can_move(row, (centerx + num3) // num2):  # RIGHT
        turns[0] = True
    if can_move(row, (centerx - num3) // num2):  # LEFT
        turns[1] = True
    if can_move((centery - num3) // num1, col):  # UP
        turns[2] = True
    if can_move((centery + num3) // num1, col):  # DOWN
        turns[3] = True

    # Cho phép rẽ mượt hơn khi gần tâm ô
    # Ưu tiên cho phép rẽ khi gần giữa ô (giúp điều khiển mượt)
    margin_x = centerx % num2
    margin_y = centery % num1

    # Nếu đang đi dọc, cho phép rẽ trái/phải khi gần giữa ô
    if direction in (2, 3):
        if 8 <= margin_x <= 20:
            if can_move(row, (centerx + num2) // num2):
                turns[0] = True
            if can_move(row, (centerx - num2) // num2):
                turns[1] = True
    # Nếu đang đi ngang, cho phép rẽ lên/xuống khi gần giữa ô
    if direction in (0, 1):
        if 8 <= margin_y <= 20:
            if can_move((centery + num1) // num1, col):
                turns[3] = True
            if can_move((centery - num1) // num1, col):
                turns[2] = True

    return turns

def check_collisions(scoree,power,power_count,eaten_ghosts):
    num1 = (HEIGHT - 50) // 32  # chiều cao mỗi ô
    num2 = WIDTH // 30          # chiều rộng mỗi ô
    if 0< player_x <580:
        if level[center_y//num1][center_x//num2]==1:
            level[center_y//num1][center_x//num2]=0
            scoree+=10
        if level[center_y//num1][center_x//num2]==2:
            level[center_y//num1][center_x//num2]=0
            scoree+=50
            power=True
            power_count=0
            eaten_ghosts=[False,False,False,False]        
    return scoree,power,power_count,eaten_ghosts

def move_player(x, y):
    num1 = (HEIGHT - 50) // 32  # chiều cao mỗi ô
    num2 = WIDTH // 30          # chiều rộng mỗi ô

    #0_right,1_left,2_up,3_down
    if direction == 0 and turns_allowed[0]:
        x += player_speed
        # Snap Y về giữa ô
        y = ((y + 15) // num1) * num1 - 15 + num1 // 2
    elif direction == 1 and turns_allowed[1]:
        x -= player_speed
        y = ((y + 15) // num1) * num1 - 15 + num1 // 2
    if direction == 2 and turns_allowed[2]:
        y -= player_speed
        # Snap X về giữa ô
        x = ((x + 15) // num2) * num2 - 15 + num2 // 2
    elif direction == 3 and turns_allowed[3]:
        y += player_speed
        x = ((x + 15) // num2) * num2 - 15 + num2 // 2
    return x, y   

def get_targets(red, blue, orange, pink, player_x, player_y):
    runaway_x = 585 if player_x < 300 else 15
    runaway_y = 615 if player_y < 300 else 35
    return_target = (290, 293)

    ghosts = [red, blue, orange, pink]
    targets = []

    if powerup:
        runaway_targets = [runaway_x, runaway_y]
        for i, ghost in enumerate(ghosts):
            if not ghost.dead and not eaten_ghost[i]:
                targets.append((runaway_x, runaway_y))
            else:
                targets.append(return_target)
    else:
        # RED
        if not red.dead:
            if red.in_box:
                red_target = (280, 238)
            elif 0 < player_x < 280 and 0 < player_y < 239:
                red_target = (player_x, player_y)
            else:
                red_target = (160, 115)
        else:
            red_target = return_target
        targets.append(red_target)

        # BLUE
        if not blue.dead:
            if blue.in_box:
                blue_target = (280, 238)
            elif 300 < player_x < 600 and 0 < player_y < 239:
                blue_target = (player_x, player_y)
            else:
                blue_target = (435, 115)
        else:
            blue_target = return_target
        targets.append(blue_target)

        # ORANGE
        if not orange.dead:
            if orange.in_box:
                orange_target = (280, 238)
            elif 0 < player_x < 280 and 260 < player_y < 600:
                orange_target = (player_x, player_y)
            else:
                orange_target = (100, 417)
        else:
            orange_target = return_target
        targets.append(orange_target)

        # PINK
        if not pink.dead:
            if pink.in_box:
                pink_target = (280, 238)
            elif 300 < player_x < 600 and 260 < player_y < 600:
                pink_target = (player_x, player_y)
            else:
                pink_target = (400, 480)
        else:
            pink_target = return_target
        targets.append(pink_target)

    return targets

 # --- Q-learning agent điều khiển Pacman ---

def find_nearest_dot(level, player_x, player_y):
    """Tìm dot gần nhất còn lại trên bản đồ"""
    num1 = (690 - 50) // 32
    num2 = 600 // 30
    min_dist = float('inf')
    nearest_dot = None
    for i in range(len(level)):
        for j in range(len(level[i])):
            if level[i][j] in (1, 2):  # dot hoặc powerdot
                dot_x = j * num2 + num2 // 2
                dot_y = i * num1 + num1 // 2
                dist = ((player_x - dot_x)**2 + (player_y - dot_y)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_dot = (j, i)
    return nearest_dot
       
run =True
while run:
    timer.tick(fps)
    if counter <19 :
        counter+=1
        if counter>3:
            flicker = False
    else:
        counter=0
        flicker = True
    if powerup and power_counter <600:
        power_counter+=1
    elif powerup and power_counter >=600:
        powerup=False
        power_counter=0
        eaten_ghost=[False,False,False,False]
    if startup_coutnter <60 and not game_over and not game_won:
        moving =False
        startup_coutnter+=1
    else:
        moving =True
    
    if powerup:
        for i in range(4):
            if [red_dead,blue_dead,orange_dead,pink_dead][i]:
                ghot_speed[i]=4
            else:
                ghot_speed[i]=1
    else:
        for i in range(4):
            ghot_speed[i]=2
        
    screen.fill('black')
    draw_board()
    center_x=player_x+15
    center_y=player_y+15
    game_won=True
    for i in range(len(level)):
        if 1 in level[i] or 2 in level[i]:
            game_won=False
    player_circle=pygame.draw.circle(screen,'white',(center_x,center_y), 15,1)
    draw_player()
    draw_misc()
 
    red=Ghosts(red_x,red_y,targets[0],ghot_speed[0],red_img,red_direction,red_dead,red_box,0)
    blue=Ghosts(blue_x,blue_y,targets[1],ghot_speed[1],blue_img,blue_direction,blue_dead,blue_box,1)
    orange=Ghosts(orange_x,orange_y,targets[2],ghot_speed[2],orange_img,orange_direction,orange_dead,orange_box,2)
    pink=Ghosts(pink_x,pink_y,targets[3],ghot_speed[3],pink_img,pink_direction,pink_dead,pink_box,3)
    targets = get_targets(red, blue, orange, pink, player_x, player_y)   
    
    ghosts = [red, blue, orange, pink]
    turns_allowed = check_position(center_x,center_y)
    if moving:
        player_x,player_y=move_player(player_x,player_y)
        red_x,red_y,red_direction=red.move_generic()
        blue_x,blue_y,blue_direction=blue.move_generic()
        orange_x,orange_y,orange_direction=orange.move_generic()
        pink_x,pink_y,pink_direction=pink.move_generic()
    score,powerup,power_counter,eaten_ghost=check_collisions(score,powerup,power_counter,eaten_ghost)
    
    if not powerup:
        if (player_circle.colliderect(red.rect)and not red.dead or
            player_circle.colliderect(blue.rect)and not blue.dead or
            player_circle.colliderect(pink.rect)and not pink.dead or
            player_circle.colliderect(orange.rect)and not orange.dead):
            if lives > 1:
                powerup=False
                lives -=1
                startup_coutnter=0
                score-=50
                player_x=279
                player_y=354
                direction=0
                direction_command=0
                red_x=260
                red_y=280
                red_direction=0 
                red_dead=False
                blue_x=233
                blue_y=320
                blue_direction=2
                blue_dead=False
                pink_x=338
                pink_y=273
                pink_direction=2
                pink_dead=False
                orange_x=338
                orange_y=320
                orange_direction=2
                orange.dead=False
                eaten_ghost=[False,False,False,False]
                targets=[(player_x,player_y),(player_x,player_y),(player_x,player_y),(player_x,player_y)]
            else:
                game_over=True
                moving =False
                startup_coutnter=0
    else:
        if player_circle.colliderect(red.rect) and not red_dead and not eaten_ghost[0]:
            red_dead=True
            eaten_ghost[0]=True
            score+=50 
        if player_circle.colliderect(blue.rect) and not blue_dead and not eaten_ghost[1]:
            blue_dead=True
            eaten_ghost[1]=True
            score+=50
        if player_circle.colliderect(orange.rect) and not orange_dead and not eaten_ghost[2]:
            orange_dead=True
            eaten_ghost[2]=True
            score+=50
        if player_circle.colliderect(pink.rect) and not pink_dead and not eaten_ghost[3]:
            pink_dead=True
            eaten_ghost[3]=True
            score+=50  
    cell_w = 600 // 30
    cell_h = (690 - 50) // 32
    start = (center_x // cell_w, center_y // cell_h)
    goal = (dot_x // cell_w, dot_y // cell_h)
    if current_goal is None or level[current_goal[1]][current_goal[0]] not in (1, 2):
        current_goal = find_nearest_dot(level, center_x, center_y)
    if current_goal is not None:
        path = astar_avoid_ghost(level, start, current_goal, ghosts, powerup)
        if len(path) > 1:
            next_step = path[1]
            direction = get_direction_from_path(start, next_step)

    # Cập nhật reward đơn giản
    if score > prev_score:
        reward += (score - prev_score) * 2  # tăng trọng số ăn pellet
        prev_score = score
    # Thưởng cho sống sót lâu
    reward += 20

    # Thưởng lớn khi thắng
    if game_won:
        reward += 200

    # Phạt mạnh khi thua
    if game_over:
        reward -= 100

    #    --- Tính reward dựa trên khoảng cách --- #
    if nearest_dot is not None:
        current_dot_dist = ((player_x - dot_x)**2 + (player_y - dot_y)**2)**0.5
        if 'prev_dot_dist' not in locals():
            prev_dot_dist = current_dot_dist

        delta = prev_dot_dist - current_dot_dist
        reward += delta * 10  # thưởng tỉ lệ khoảng cách tiến gần
        prev_dot_dist = current_dot_dist


    # Phạt nếu quá gần ghost
    for g in [red, blue, orange, pink]:
        dist = ((player_x - g.x_pos)**2 + (player_y - g.y_pos)**2)**0.5
        if not powerup:
            if dist < 40:
                reward -= (40 - dist) * 0.0001  # phạt mạnh khi lại gần
            else:
                reward += dist * 20       # thưởng nhẹ khi giữ khoảng cách
        # Phạt khi đứng yên
    if (player_x == prev_playerx) and (player_y == prev_playery):
        reward -= 10
        stuck_counter += 1
    else:
        stuck_counter = 0
    if stuck_counter >= 15:
        agent.epsilon = min(1.0, agent.epsilon + 0.001)# tăng lại random để khám phá
        stuck_counter = 0
        if len(path) > 1:
            next_step = path[1]
            next_dir = get_direction_from_path(start, next_step)
            direction = next_dir           
            stuck_counter = 0
            

    state = get_state(player_x, player_y, ghosts, level, powerup)
    action = agent.select_action(state)
    turns_allowed = check_position(center_x, center_y)
    action = random.choice(turns_allowed)
    #direction_command = action
    #direction = action
    min_dist = float('inf')
    nearest_dot = None
    for i in range(len(level)):
        for j in range(len(level[i])):
            if level[i][j] in (1, 2):  # dot hoặc power dot
                dot_x = j * (600 // 30) + (600 // 60)
                dot_y = i * ((690 - 50) // 32) + ((690 - 50) // 64)
                dist = ((player_x - dot_x)**2 + (player_y - dot_y)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_dot = (dot_x, dot_y)
    if nearest_dot is not None:
        current_dot_dist = np.sqrt((player_x - nearest_dot[0])**2 + (player_y - nearest_dot[1])**2)
        if 'prev_dot_dist' not in locals():
            prev_dot_dist = current_dot_dist
        if current_dot_dist < prev_dot_dist:
            reward += 100
        else:
            reward -= 5
    prev_dot_dist = current_dot_dist
    # Di chuyển
    player_x, player_y = move_player(player_x, player_y)
    
    # Phạt khi quay đầu ngay
    if prev_action is not None:
        if (prev_action == 0 and action == 1) or (prev_action == 1 and action == 0) or \
        (prev_action == 2 and action == 3) or (prev_action == 3 and action == 2):
            reward -= 5  # phạt khi quay đầu ngay
    prev_action = action
    
    
     # --- LẤY TRẠNG THÁI MỚI ---
    next_state = get_state(player_x, player_y, ghosts, level, powerup)
    done = game_over or game_won
     # --- LƯU TRẢI NGHIỆM & TRAIN ---
    agent.memory.push(state, action, reward, next_state, done)
    agent.train()
    agent.decay_epsilon()
    state = next_state
    prev_score = score
    if (player_x != prev_playerx) or (player_y != prev_playery):
        step += 1
    prev_playerx = player_x
    prev_playery = player_y
    if step % 10 == 0:
        agent.update_target()
        print(f"Step={step}, score={score}, Epsilon={agent.epsilon:.3f}")

       
    
    # --- Khi thắng/thua ---
    if game_won:
        log_training_auto(score, step, agent.epsilon,reward/10000)
        game_won = False
        pygame.time.delay(500)
        reward = 0
        prev_action = None
        player_x, player_y = 279, 354
        direction = direction_command = 0
        powerup=False
        # Reset ghost
        red_x, red_y, red_direction, red_dead = 260, 280, 0, False
        blue_x, blue_y, blue_direction, blue_dead = 233, 320, 2, False
        pink_x, pink_y, pink_direction, pink_dead = 338, 273, 2, False
        orange_x, orange_y, orange_direction, orange_dead = 338, 320, 2, False
        eaten_ghost = [False, False, False, False]
        lives = 3
        episode += 1
        agent.save("pacman_dqn.pth")
        print(f"[ Model saved] Score={score:.2f}, Step={step}, Epsilon={agent.epsilon:.3f}")
        print(f"Huấn luyện xong lần: {episode}")
        print("--------------------------------------------------")    
        step = 0
        score = 0.0
        level = copy.deepcopy(level_initial)
        
    if game_over: 
        log_training_auto(score, step, agent.epsilon,reward/10000)
        pygame.time.delay(300)
        reward = 0
        prev_action = None
        game_over = False
        powerup=False      
        red_x, red_y, red_direction, red_dead = 260, 280, 0, False
        blue_x, blue_y, blue_direction, blue_dead = 233, 320, 2, False
        pink_x, pink_y, pink_direction, pink_dead = 338, 273, 2, False
        orange_x, orange_y, orange_direction, orange_dead = 338, 320, 2, False
        eaten_ghost = [False, False, False, False]
        player_x, player_y = 279, 354
        direction = direction_command = 0
        lives = 3
        episode += 1
        agent.save("pacman_dqn.pth")
        print(f"[ Model saved] Score={score:.2f}, Step={step}, Epsilon={agent.epsilon:.3f}")
        print(f"Huấn luyện xong lần: {episode}")
        print("--------------------------------------------------")    
        step = 0
        score = 0.0
        level = copy.deepcopy(level_initial)

     
        
        
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            run=False
        if event.type==pygame.KEYDOWN:
            '''if event.key == pygame.K_RIGHT:
                direction_command=0
            if event.key == pygame.K_LEFT:
                direction_command=1
            if event.key == pygame.K_UP:
                direction_command=2 
            if event.key == pygame.K_DOWN:
                direction_command=3  '''
            if event.key ==pygame.K_SPACE  and (game_over or game_won):
                powerup=False
                startup_coutnter=0
                player_x=279
                player_y=354
                direction=0
                direction_command=0
                red_x=260
                red_y=280
                red_direction=0 
                red_dead=False
                blue_x=233
                blue_y=320
                blue_direction=2
                blue_dead=False
                pink_x=338
                pink_y=273
                pink_direction=2
                pink_dead=False
                orange_x=338
                orange_y=320
                orange_direction=2
                orange.dead=False
                eaten_ghost=[False,False,False,False] 
                lives=3
                score=0
                level=copy.deepcopy(boards)
                game_over=False
                game_won=False
        '''if event.type==pygame.KEYUP:
            if event.key == pygame.K_RIGHT and direction_command == 0:
                direction_command=direction
            if event.key == pygame.K_LEFT and direction_command == 1:
                direction_command=direction
            if event.key == pygame.K_UP and direction_command == 2:
                direction_command=direction
            if event.key == pygame.K_DOWN and direction_command == 3:
                direction_command=direction               
    if direction_command == 0 and turns_allowed[0]:
        direction = 0     
    elif direction_command == 1 and turns_allowed[1]:
        direction = 1     
    elif direction_command == 2 and turns_allowed[2]:
        direction = 2     
    elif direction_command == 3 and turns_allowed[3]:
        direction = 3   '''       
    if player_x > WIDTH:
         player_x = 15
    elif player_x < 15:
         player_x = WIDTH-30  
    if red.in_box and red.dead: 
        red_dead=False
        eaten_ghost[0]=False
    if blue.in_box and blue.dead: 
        blue_dead=False
        eaten_ghost[1]=False
    if orange.in_box and orange.dead: 
        orange_dead=False
        eaten_ghost[2]=False
    if pink.in_box and pink.dead: 
        pink_dead=False
        eaten_ghost[3]=False  
                         
    pygame.display.flip()
pygame.quit()