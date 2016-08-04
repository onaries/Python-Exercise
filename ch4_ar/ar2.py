from numpy import *
from pylab import *
import PIL

from OpenGL import GL
from OpenGL import GLU
from OpenGL import GLUT
import pygame, pygame.image
from pygame.locals import *
import pickle


width,height = 640,360

def set_projection_from_camera(K):

    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glLoadIdentity()

    fx = K[0,0]
    fy = K[1,1]
    fovy = 2*arctan(0.5*height/fy)*180/pi
    aspect = (float)(width*fy)/(height*fx)

    print aspect

    # define the near and far clipping planes
    near = 0.1
    far = 100

    # set perspective
    GLU.gluPerspective(fovy,aspect,near,far)
    GL.glViewport(0,0,width,height)

def set_modelview_from_camera(Rt):
    ''' Set the model view matrix from camera pose. '''

    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glLoadIdentity()

    # rotate teapot 90 deg around x-axis so that z-axis is up
    Rx = array([[1,0,0],[0,0,1],[0,1,0]])

    # set rotation to best approximation
    R = Rt[:,:3]
    U,S,V = linalg.svd(R)
    R = dot(U,V)
    R[0,:] = -R[0,:] # change sign of x-axis

    # set translation
    t = Rt[:,3]
    t[0] = -t[0]
    print t

    # print t

    # setup 4*4 model view matrix
    M = eye(4)
    # print R
    M[:3,:3] = dot(R,Rx)
    # print M[:3,:3]
    M[:3,3] = t

    # transpose and flatten to get column order
    M = M.T

    print M.T
    m = M.flatten()

    # replace model view with the new matrix
    GL.glLoadMatrixf(m)

def draw_background(imname):
    ''' Draw background image using a quad. '''

    # load background image (should be .bmp) to OpenGL texture
    bg_image = pygame.image.load(imname).convert()
    bg_data = pygame.image.tostring(bg_image, "RGBA", 1)

    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glLoadIdentity()

    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    # bind the texture
    GL.glEnable(GL.GL_TEXTURE_2D)
    GL.glBindTexture(GL.GL_TEXTURE_2D, GL.glGenTextures(1))
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, bg_data)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)

    # create quad to fill the whole window
    GL.glBegin(GL.GL_QUADS)
    GL.glTexCoord2f(0.0, 0.0); GL.glVertex3f(-1.0, -1.0, -1.0)
    GL.glTexCoord2f(1.0, 0.0); GL.glVertex3f(1.0, -1.0, -1.0)
    GL.glTexCoord2f(1.0, 1.0); GL.glVertex3f(1.0, 1.0, -1.0)
    GL.glTexCoord2f(0.0, 1.0); GL.glVertex3f(-1.0, 1.0, -1.0)
    GL.glEnd()

    # clear the texture
    GL.glDeleteTextures(1)

def draw_teapot(size):
    ''' Draw a red teapot at the origin. '''

    GL.glEnable(GL.GL_LIGHTING)
    GL.glEnable(GL.GL_LIGHT0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

    # draw red teapot
    GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT, [0,0,0,0])
    GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, [1.0,1.0,1.0,0.0])
    GL.glMaterialfv(GL.GL_FRONT, GL.GL_SPECULAR, [0.7,0.6,0.6,0.0])
    GL.glMaterialf(GL.GL_FRONT, GL.GL_SHININESS, 0.25*128.0)
    GLUT.glutSolidTeapot(size)


    # GLUT.glutSolidCube(size)
    # GLUT.glutSolidCylinder(size)

def draw_cube(size):
    ''' Draw a red teapot at the origin. '''

    GL.glEnable(GL.GL_LIGHTING)
    GL.glEnable(GL.GL_LIGHT0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

    # draw red teapot
    GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT, [0,0,0,0])
    GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, [0.0,0.0,0.0,0.0])
    GL.glMaterialfv(GL.GL_FRONT, GL.GL_SPECULAR, [0.7,0.6,0.6,0.0])
    GL.glMaterialf(GL.GL_FRONT, GL.GL_SHININESS, 0.25*128.0)
    GLUT.glutSolidCube(size)

def setup():
    ''' Setup window and pygame environment. '''
    pygame.init()
    pygame.display.set_mode((width, height), OPENGL | DOUBLEBUF)
    pygame.display.set_caption('OpenGL AR demo')

def load_and_draw_model(filename):
    ''' Loads a model from an .obj file using objloader.py.
      Assumes there is a .mtl material file with the same name. '''
    GL.glEnable(GL.GL_LIGHTING)
    GL.glEnable(GL.GL_LIGHT0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

    # set model color
    GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT, [0,0,0,0])
    GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, [0.5,0.5,1.0,0.0])
    GL.glMaterialf(GL.GL_FRONT, GL.GL_SHININESS, 0.25*128.0)

    # load from a file
    import objloader
    obj = objloader.OBJ(filename,swapyz=True)
    GL.glCallList(obj.gl_list)


with open('./data/ar_camera11.pkl','r') as f:
    K = pickle.load(f)
    Rt = pickle.load(f)

if __name__ == "__main__":

    setup()
    # GL.glEnable(GL.GL_NORMALIZE)
    draw_background('./data/s20160720_113436.jpg')

    set_projection_from_camera(K)
    set_modelview_from_camera(Rt)

    # print K
    # print Rt

    draw_teapot(0.05)
    # draw_cube(0.1)
    # load_and_draw_model('Sofa_3_3ds.obj')
    pygame.display.flip()
    MyClock = pygame.time.Clock()
    ticker = 100
    while True:
        event = pygame.event.poll()
        if event.type in (QUIT,KEYDOWN):
            break

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        # draw_background('./data/s20160720_113436.jpg')

        # set_projection_from_camera(K)
        # set_modelview_from_camera(Rt)
        # draw_teapot(0.1)
        #draw_cube(0.1)
        # load_and_draw_model('Sofa_3_3ds.obj')
        # pygame.display.flip()
        MyClock.tick(ticker)
