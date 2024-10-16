import math
class SweepWPGenerator:
    def __init__(self, init_pos=(0,0), length=0, angle=90, gap=0, iteration=1):
        # (x,y)
        self.init_pos, self.length, self.angle, self.gap, self.iteration = init_pos, length, angle, gap, iteration
        self.wp = []

    def generate(self,turn_orientation=1):

        x,y = self.init_pos
        self.wp = [(x,y)]
        
        for _ in range(self.iteration):
            # get the last point in wp
            x1,y1=self.wp[-1]
            # add sweeping movement
            angle_in_rad = self.angle*(math.pi/180)
            x_add=self.length*math.cos(angle_in_rad)
            y_add=self.length*math.sin(angle_in_rad)
            point1 = (x1+x_add, y1+y_add)
            self.wp.append(point1)
            # add gap 
            angle_in_rad -= turn_orientation*(90*(math.pi/180))
            x_add=self.gap*math.cos(angle_in_rad)
            y_add=self.gap*math.sin(angle_in_rad)
            point2 = (point1[0]+x_add,point1[1]+y_add)
            self.wp.append(point2)
            # go back sweeping movement
            angle_in_rad -= turn_orientation*(90*(math.pi/180))
            x_add=self.length*math.cos(angle_in_rad)
            y_add=self.length*math.sin(angle_in_rad)
            point3=(point2[0]+x_add,point2[1]+y_add)
            self.wp.append(point3)
            # add gap
            angle_in_rad += turn_orientation*(90*(math.pi/180))
            x_add=self.gap*math.cos(angle_in_rad)
            y_add=self.gap*math.sin(angle_in_rad)
            point4=(point3[0]+x_add, point3[1]+y_add)
            self.wp.append(point4)

        return self.wp
# wp=SweepWPGenerator(length=10,gap=5,iteration=5)
# waypoint=wp.generate()
# print(waypoint)