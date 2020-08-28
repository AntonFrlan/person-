class Centroid():
    def __init__(self, dot, width, height, maxDis=40):
        self.dot = dot
        self.width = width
        self.height = height
        self.oldDot = None
        self.maxDis = maxDis
        self.dis = 0
        self.found = True
        self.speed = 0

    def check(self, xa, ya, xb, yb):
        if self.dot is not None and not self.found:
            j = self.center([xa, ya, xb, yb])
            if j[0] < self.dot[0][0] < j[2] and j[1] < self.dot[0][1] < j[3]:
                return True
        return False

    def update(self, dot, xa, ya, xb, yb):
        if not self.found:
            self.width = int((xb - xa) / 2)
            self.height = int((yb - ya) / 2)
            if dot is not None:
                self.dot = dot
            else:
                self.dot[0][0] = (xa + xb) / 2
                self.dot[0][1] = (ya + yb) / 2
            self.found = True

    def disReset(self):
        self.dis = 0

    def old(self):
        if self.oldDot is not None:
            self.speed = self.dot - self.oldDot
        self.oldDot = self.dot

    def changeFound(self):
        self.found = not self.found

    def nestanite(self, w, h):
        self.dis += 1
        x = self.dot[0][0]
        y = self.dot[0][1]
        if self.dis >= self.maxDis or x < 0 or y < 0 or x > w or y > h:
            return True
        return False

    def errorLK(self):
        if 60 < abs(self.dot[0][0] - self.oldDot[0][0]) or 60 < abs(self.dot[0][1] - self.oldDot[0][1]):
            self.dot = self.oldDot + self.speed
            return True
        return False

    def center(self, j):
      #  j[2] += self.width
      #  j[0] -= self.width
       # j[3] += self.height
        #j[1] -= self.height
        return j
