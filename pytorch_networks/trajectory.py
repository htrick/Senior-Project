import math

class Trajectory:
    # Create a new Trajectory object once, with the number of output points, the width, and the height
    def __init__(self, numberOfOutputs, imageWidth, imageHeight):
        self.width = imageWidth
        self.height = imageHeight
        self.numOutputs = numberOfOutputs
        self.bottomCenter = ((self.width - 1) // 2, self.height - 1)
        top_left = (int(0.425 * self.width), int(0.64 * self.height))
        top_right = (int(0.575 * self.width), int(0.64 * self.height))
        bottom_left = (int(0.37 * self.width), int(self.height - 1))
        bottom_right = (int(0.63 * self.width), int(self.height - 1))
        self.outline = (top_left, top_right, bottom_left, bottom_right)

    # Get the angle of the trajectory in degress (right is negative)
    def get_angle(self, end_point):
        return self.__calculate_angle(end_point)

    # Get the coordinates of the trajectory point in (x, y) format
    def get_target_point(self, points_list):
        return self.__calculate_direction(points_list)
    
    # Get the 4 points for the outline of the width of the robot for 3 feet in front
    # Format: (top_left, top_right, bottom_left, bottom_right)
    def get_outline(self):
        return self.outline
    


    # The functions below should not be called from outside of this class

    def __calculate_angle(self, endPoint):
        if self.bottomCenter [1] - endPoint[1] == 0:
            angle = math.pi * (-1 if endPoint[0] > self.bottomCenter[0] else 1)
        else:
            angle = math.atan((self.bottomCenter[0] - endPoint[0]) / (self.bottomCenter[1] - endPoint[1]))
        return round(angle * 180 / math.pi) # angle in degrees, right is negative

    # Find the furthest point from the bottom center
    def __calculate_direction(self, points_list):
        highest_value = min(points_list)
        # if the robot is blocked
        if self.__is_blocked(points_list):
            x = 0 if points_list.index(highest_value) < (self.numOutputs / 2) else self.width - 1
            y = self.height - 1
            return (x, y)
        
        # if the highest point is in the upper half of the image
        if highest_value < 0.5:
            x = int(points_list.index(highest_value) * self.width / self.numOutputs)
            y = int(highest_value * self.height)
            return (x, y)

        furthestPoint = (0, 0) # otherwise, find the furthest point to the right or left
        furthestDistance = 0
        for i in range(len(points_list)):
            x = int(i * self.width / self.numOutputs)
            y = int(points_list[i] * self.height)
            distance = math.sqrt(((self.bottomCenter[0] - x) ** 2) + ((self.bottomCenter[1] - y) ** 2))
            if distance > furthestDistance:
                furthestDistance = distance
                furthestPoint = (x, y)
        return furthestPoint
    
    # Check if any points are in the area of the outline polygon
    def __is_blocked(self, points_list):
        for i in range(len(points_list)):
            x_percent = i / self.numOutputs
            y_percent = points_list[i]
            if y_percent > 0.64 and x_percent > ((y_percent - 3.4218) / -6.55) and x_percent < ((y_percent + 3.1236) / 6.55):
                return True
        return False
