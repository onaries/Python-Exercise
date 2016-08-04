from OpenGL.GL import *
from OpenGL.GLU import *
import pygame, pygame.image
from pygame.locals import *
import ARAR as ar
import pickle
import time
from scipy import *
from numpy import *
import sys
sys.path.append("../objloader")
import objloader
import cPickle

width,height=1000,747
#width,height = 2048,1152

def set_projection_from_camera(K):

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    fx = K[0,0]
    fy = K[1,1]
    fovy = 2*arctan(0.5*height/fy)*180/pi
    aspect = (width*fy)/(height*fx)

    # define the near and far clipping planes
    near = 0.1
    far = 100

    # set perspective
    gluPerspective(fovy,aspect,near,far)
    glViewport(0,0,width,height)

def set_modelview_from_camera(Rt):
    ''' Set the model view matrix from camera pose. '''

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # rotate teapot 90 deg around x-axis so that z-axis is up
    Rx = array([[1,0,0],[0,0,-1],[0,1,0]])

    # set rotation to best approximation
    R = Rt[:,:3]
    U,S,V = linalg.svd(R)
    R = dot(U,V)
    R[0,:] = -R[0,:] # change sign of x-axis

    # set translation
    t = Rt[:,3]
    t[0] = -t[0]/2

    # setup 4*4 model view matrix
    M = eye(4)
    M[:3,:3] = dot(R,Rx)
    M[:3,3] = t

    # transpose and flatten to get column order
    M = M.T
    m = M.flatten()

    # replace model view with the new matrix
    glLoadMatrixf(m)


def setup():
    """ Setup window and pygame environment """
    pygame.init()
    pygame.display.set_mode((width, height), OPENGL|DOUBLEBUF)
    pygame.display.set_caption('OpenGL AR demo')

def rotate(degrees, K, Rt, width, height, img):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    ar.draw_background(img)
    ar.set_projection_from_camera(K, width, height)
    ar.set_modelview_from_camera(Rt)
    ar.draw_teapot(0.1, degrees)
    draw_teapots()
    pygame.display.flip()

def draw_teapots():
    for i in [0.2, 0, -0.2]:
        for j in [0.2, 0, -0.2]:
            glPushMatrix()
            if not (i==0 and j==0):
                ar.draw_teapot(0.05, pos=[i,0,j])
            glPopMatrix()


def draw_furniture(obj, size, angle=[0, 0, 0], pos=[0, 0, 0]):
    """ Draw a red teapot at the origin """
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_COLOR_MATERIAL)
    glShadeModel(GL_SMOOTH)
    glClear(GL_DEPTH_BUFFER_BIT)

    glMaterialfv(GL_FRONT, GL_AMBIENT, [0,0,0,0.0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0.0, 0.0, 0.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.7, 0.6, 0.6, 0.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 0.25*128.0)


    # Translation
    glTranslatef(pos[0], pos[1], pos[2])

    # Rotation
    glRotatef(angle[0], 1, 0, 0)
    glRotatef(angle[1], 0, 1, 0)
    glRotatef(angle[2], 0, 0, 1)

    # obj = objloader.OBJ("Sofa_3_3ds.obj", swapyz=True)

    glScalef(size, size, size)
    glCallList(obj.gl_list)

    """ All this late night reading made sense of it...
    glDisable should be called after glEnable and uses if that feature...
    although state switching is expensive...
    the image was not being redrawn as GL_LIGHTING turns the image
    texture black as it is parallel to the plane of the backgorund texture
    the background, http://stackoverflow.com/questions/
    802079/should-i-call-glenable-and-gldisable-every-time-
    i-draw-something"""
    glDisable(GL_LIGHTING)
    glDisable(GL_LIGHT0)
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_COLOR_MATERIAL)
    glDisable(GL_DEPTH_TEST)


if __name__=="__main__":
    # load camera data
    img = './data/book_perspective.jpg'
    #img = "../../data/mag_front.jpg"
    #img  = "../../data/mag_perspective.jpg"
    #img = "../../data/mag_perspective_1.jpg"
    #pkl = img[:-4]+".pkl"
    with open("./data/ar_camera.pkl", "r") as f:
    #with open(pkl, "r") as f:
        K = pickle.load(f)
        Rt = pickle.load(f)

    setup()
    glEnable(GL_NORMALIZE)
    ar.draw_background(img)
    ar.set_projection_from_camera(K, width, height)
    ar.set_modelview_from_camera(Rt)

    """ Load once into memory as this is time-taking """
    obj = objloader.OBJ("toyplane3.obj")
    # obj = objloader.OBJ("Sofa_3_3ds.obj")
    #obj_2 = objloader.OBJ("toyplane.obj")
#     obj = None
#     with open("../objloader/toyplane.pkl", 'rb') as input:
#         obj = cPickle.load(input)
#     if not obj:
#         sys.exit()

    #(scale, degrees, pos) = (0.001486436280241436, [0, -5, 0], [0.09500000000000001, 0, 0.10500000000000002])
    degrees = [0,0,0]
    pos = [0,0,0]
    scale = 0.005

    ar.draw_furniture(obj, scale)
    pygame.display.flip()

    clock = pygame.time.Clock()
    ticker = 30
    rotate_unit = 0
    translate_unit = [0,0,0]
    scale_unit = 1
    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == (QUIT,KEYDOWN):
                break
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    #degrees[0] += 10
                    translate_unit[0] = 0.005
                if event.key == K_DOWN:
                    #degrees[0] -= 10
                    translate_unit[0] = -0.005
                if event.key == K_RIGHT:
                    #degrees[1] += 10
                    translate_unit[2] = -0.005
                if event.key == K_LEFT:
                    #degrees[1] -= 10
                    translate_unit[2] = 0.005
                if event.key == K_MINUS:
                    scale_unit = 1/1.1
                    #print scale_unit
                if event.key == K_PERIOD:
                    scale_unit = 1.1
                    #print scale_unit
                if event.key == K_q:
                    rotate_unit = 5
                if event.key == K_s:
                    rotate_unit  = -5
            if event.type == KEYUP:
                if event.key == K_q or event.key == K_s:
                    rotate_unit = 0
                if event.key in (K_UP, K_DOWN):
                    translate_unit[0] = 0
                    #print translate_unit
                if event.key in (K_LEFT, K_RIGHT):
                    translate_unit[2] = 0
                    #print translate_unit
                if event.key in (K_PERIOD, K_MINUS):
                    scale_unit = 1


        for i,x in enumerate(pos):
            pos[i] += translate_unit[i]
        degrees[1] += rotate_unit
        scale *= scale_unit
        print (scale, degrees, pos)
        #degrees[0] += 10
        #degrees[0] %= 360
        """ Simply flipping creates flicker
        but redrawing followed by flipping removes the flicker """
        #rotate(degrees, K, Rt, width, height, img) # includes flip()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        ar.draw_background(img)
        ar.set_projection_from_camera(K, width, height)
        ar.set_modelview_from_camera(Rt)
        #ar.draw_teapot(scale, degrees, pos)
        draw_furniture(obj, scale, degrees, pos)
        #ar.draw_furniture(obj_2, scale, degrees, pos)
        pygame.display.flip()
        clock.tick(ticker)
        #time.sleep(0.1)
