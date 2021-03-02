#/usr/bin/env python
import unittest



Q = [
#Faces
(1, +1,  0,  0),
(2, -1,  0,  0),
(3,  0, +1,  0),  #UP 
(4,  0, -1,  0),  #DOWN
(5,  0,  0, +1),
(6,  0,  0, -1),

#Edges
(7,  +1, +1,  0),
(8,  -1, -1,  0),
(9,  +1,  0, +1),
(10, -1,  0, -1),
(11,  0, +1, +1),
(12,  0, -1, -1),
(13, +1, -1,  0),
(14, -1, +1,  0),
(15, +1,  0, -1),
(16, -1,  0, +1),
(17,  0, +1, -1),
(18,  0, -1, +1),

#Corners
(19, +1, +1, +1),
(20, -1, -1, -1),
(21, +1, +1, -1),
(22, -1, -1, +1),
(23, +1, -1, +1),
(24, -1, +1, -1),
(25, -1, +1, +1),
(26, +1, -1, -1)
]


def sign(n):
    if n == 0:
        return " 0"
    elif n < 0:
        return str(n)
    else:
        return "+" + str(n)

def nullorsign(n):
    if n == 0:
        return "  "
    else:
        return sign(n)


def direction((l, x, y, z)):
    return sign(x) + " " + sign(y) + " " + sign(z)


def text_direction((l, x, y, z)):
    text = []
    if x < 0: text.append("LEFT")
    elif x > 0: text.append("RIGHT")
    
    if y < 0: text.append("DOWN")
    elif y > 0: text.append("UP")
    
    if z < 0: text.append("FORWARD")
    elif z > 0: text.append("BACKWARD")
    
    return "_".join(text)




def increment(index, n):
    if n == 0:
        return index
    elif n < 0:
        return "(%s - 1)" % index
    elif n > 0:
        return "(%s + 1)" % index

def index(line):
    return "(%s * yg * zg) + (%s * zg) + %s" % (increment("i", line[1]), increment("j", line[2]), increment("k", line[3]))




def function(line):
    here = """
//%s DIRECTION    Q%s
// %s
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::iQ%s(tNi i, tNi j, tNi k)
{
    return %s;
}
"""
    return (here % (text_direction(line), line[0], direction(line), line[0], index(line)))


def opposite(n):
    if n%2:
        return n+1
    else :
        return n-1

def count_dirns((q, x, y, z)):
    dirn = 0
    if x != 0: dirn += 1
    if y != 0: dirn += 1
    if z != 0: dirn += 1
    return dirn

def get_corners():
    return [q for (q,x,y,z) in Q if count_dirns(q,x,y,z) == 3]

def get_edges():
    return [q for (q,x,y,z) in Q if count_dirns(q,x,y,z) == 2]

def get_faces():
    return [q for (q,x,y,z) in Q if count_dirns(q,x,y,z) == 1]



def get_rights():
    return [q for (q,x,y,z) in Q if x == +1]

def get_lefts():
    return [q for (q,x,y,z) in Q if x == -1]


def get_ups():
    return [q for (q,x,y,z) in Q if y == +1]

def get_downs():
    return [q for (q,x,y,z) in Q if y == -1]


def get_forwards():
    return [q for (q,x,y,z) in Q if z == +1]

def get_backwards():
    return [q for (q,x,y,z) in Q if z == -1]




for line in Q:
    print("tNi inline dirnQ%s(tNi i, tNi j, tNi k);" % line[0])

for line in Q:
    print(function(line))



def bounce_back(t):
    print("Q[dirnQ%s(i,j,k)].q[Q%s] = Q[dirnQ000(i,j,k)].q[Q%s];" % (opposite(t), opposite(t), t))


print("Right")
for t in get_rights():
    bounce_back(t)

print("Left")
for t in get_lefts():
    bounce_back(t)


print("Up")
for t in get_ups():
    bounce_back(t)

print("Down")
for t in get_downs():
    bounce_back(t)


print("Forwards")
for t in get_forwards():
    bounce_back(t)

print("Backwards")
for t in get_backwards():
    bounce_back(t)























class TestStringMethods(unittest.TestCase):

    def test_opposite(self):
        self.assertEqual(opposite(1), 2)
        self.assertEqual(opposite(16), 15)

    def test_count_dirns(self):
        self.assertEqual(count_dirns((99, 1, -1, 1)), 3)
        
    def test_sign(self):
        self.assertEqual(sign(0), " 0")
        self.assertEqual(sign(-1), "-1")
        self.assertEqual(sign(1), "+1")

    def test_nullorsign(self):
        self.assertEqual(nullorsign(0), "  ")
        self.assertEqual(nullorsign(-1), "-1")
        self.assertEqual(nullorsign(1), "+1")

    def test_direction(self):
        self.assertEqual(direction((99, 1, 1, 1)), "+1 +1 +1")
        self.assertEqual(direction((99, 1, -1, 0)), "+1 -1  0")

    def test_increment(self):
        self.assertEqual(increment("i", 0), "i")
        self.assertEqual(increment("j", 1), "(j + 1)")
        self.assertEqual(increment("k", -1), "(k - 1)")

    def test_index(self):
        line = (0, 1, 0, -1)
        self.assertEqual(index(line), "((i + 1) * yg * zg) + (j * zg) + (k - 1)")

    def test_text_direction(self):
        self.assertEqual(text_direction((99, 0, 1, 0)), "UP")        
        self.assertEqual(text_direction((99, 1, 0, -1)), "RIGHT_FORWARD")        
        self.assertEqual(text_direction((99, 1, 1, 1)), "RIGHT_UP_BACKWARD")


    def test_tops(self):
        self.assertListEqual(get_ups(), [3, 7, 11, 14, 17, 19, 21, 24, 25])

            


    def test_function(self):
        here = """
//UP DIRECTION    Q3
//  0 +1  0
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::iQ3(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + ((j + 1) * zg) + k;
}
"""
        line = (3, 0, 1, 0)
        self.assertEqual(function(line), here)




if __name__ == '__main__':
    unittest.main()