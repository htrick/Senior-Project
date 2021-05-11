import math

class Trajectory:
    def __init__(self, points_list, imageWidth, imageHeight):
        self.points_list = points_list
        self.width = imageWidth
        self.height = imageHeight
        self.numOutputs = len(points_list)
        bottomCenter = ((self.width - 1) // 2, self.height - 1)
        self.end_point = self.__calculate_direction(bottomCenter)
        self.angle = self.__calculate_angle(bottomCenter, self.end_point)

    def get_angle(self):
        return self.angle

    def get_target_point(self):
        return self.end_point
    
    def __calculate_angle(self, bottomCenter, endPoint):
        if bottomCenter [1] - endPoint[1] == 0:
            angle = math.pi * (-1 if endPoint[0] > bottomCenter[0] else 1)
        else:
            angle = math.atan((bottomCenter[0] - endPoint[0]) / (bottomCenter[1] - endPoint[1]))
        return round(angle * 180 / math.pi) # angle in degrees, right is negetive

    # Find the furthest point from the bottom center
    def __calculate_direction(self, bottomCenter):
        highest_value = min(self.points_list)
        if highest_value < 0.5: # if the highest point is in the upper half of the image
            x = int(self.points_list.index(highest_value) * self.width / self.numOutputs)
            y = int(highest_value * self.height)
            return (x, y)

        furthestPoint = (0, 0) # otherwise, find the furthest point to the right or left
        furthestDistance = 0
        for i in range(len(self.points_list)):
            x = int(i * self.width / self.numOutputs)
            y = int(self.points_list[i] * self.height)
            distance = math.sqrt(((bottomCenter[0] - x) ** 2) + ((bottomCenter[1] - y) ** 2))
            if distance > furthestDistance:
                furthestDistance = distance
                furthestPoint = (x, y)
        return furthestPoint
