import pyrap.tables
from pyrap.tables import table
from Pyxis.ModSupport import *
import mqt
import cal
import imager
import stefcal
import lsm
import ms
import std
import numpy as np
import pylab as plt
import pickle
import pyfits
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats            


#Setting Pyxis global variables
v.OUTDIR = '.'
v.BASELINE = ""
v.CALTECH = ""
v.POSNEG = ""
#v.MS = 'KAT7_1445_1x16_12h.ms'
v.DESTDIR_Template = "${OUTDIR>/}plots-${MS:BASE}"
v.OUTFILE_Template = "${DESTDIR>/}${MS:BASE}${_<CALTECH}${_<BASELINE}"
imager.DIRTY_IMAGE_Template = "${OUTFILE}.dirty.fits"
imager.RESTORED_IMAGE_Template = "${OUTFILE}.restored.fits"
imager.RESIDUAL_IMAGE_Template = "${OUTFILE}.residual.fits"
imager.MASK_IMAGE_Template = "${OUTFILE}.mask.fits"
imager.MODEL_IMAGE_Template = "${OUTFILE}.model.fits"
lsm.PYBDSM_OUTPUT_Template = "${OUTFILE}${_<POSNEG}.lsm.html"
v.PICKLENAME = ""
v.PICKLEFILE_Template = "${DESTDIR>/}Pickle/${MS:BASE}${_<PICKLENAME}.p"
v.GHOSTMAP_Template = "${DESTDIR>/}${MS:BASE}_GhostMap${_<BASELINE}.txt"
#v.LOG_Template = "${OUTDIR>/}log-${MS:BASE}"


class Ellipse():
    #This calculates the parameters of a given ellipse
    def __init__(self,ms):
        self.baseline_parm_n = np.array([])
        self.baseline_parm = np.array([])
        self.baseline_l = np.array([])
        self.b_number = np.array([])
        self.phi_m = np.array([])
        self.c_m = np.array([])
        self.theta = np.array([])
        self.theta_m = np.array([])
        self.sin_delta = 0 
        self.ms = ms
        print "self.ms.nb = ",self.ms.nb


    def generate_xy(self,x_c,y_c,delta,a,b,num_points,c,reverse):
        t = np.linspace(0,c*np.pi,num=num_points,endpoint = True)
        x = x_c + a*np.cos(t)*np.cos(delta)-b*np.sin(t)*np.sin(delta)
        y = y_c + a*np.cos(t)*np.sin(delta)+b*np.sin(t)*np.cos(delta)
        if reverse:
           #x = -1*x
           y = -1*y  
        return x,y

    def plot_ellipse(self,x,y):
        plt.plot(x,y)
        plt.show()

    def determine_ABCDEF(self,x,y):
        x_4 = np.sum(x**4)
        x_3 = np.sum(x**3)
        x_3_y = np.sum(x**3*y)
        x_2_y = np.sum(x**2*y)
        x_2_y_2 = np.sum(x**2*y**2)
        x_2 = np.sum(x**2) 
        x_y = np.sum(x*y)
        x_y_2 = np.sum(x*y**2)
        x_2_y_2 = np.sum(x**2*y**2)
        x_y_3 = np.sum(x*y**3)
        y_2 = np.sum(y**2)
        y_3 = np.sum(y**3)
        y_4 = np.sum(y**4)
        

        A = np.array([(x_4,x_3,x_3_y,x_2_y,x_2_y_2),(x_3,x_2,x_2_y,x_y,x_y_2),(x_3_y,x_2_y,x_2_y_2,x_y_2,x_y_3),(x_2_y,x_y,x_y_2,y_2,y_3),(x_2_y_2,x_y_2,x_y_3,y_3,y_4)])   
 
        F = np.array([-1*x_2,-1*np.sum(x),-1*x_y,-1*np.sum(y),-1*y_2])

        P = np.linalg.solve(A,F)

        B_00 = -1*(P[0]*x_2+P[1]*np.sum(x)+P[2]*x_y+P[3]*np.sum(y)+P[4]*y_2)

        #print "B_00 = ",B_00        

        P = np.append(P,B_00)

        return P

    def determine_center_orientation(self,P):
        A = np.array([(-2*P[0],-1*P[2]),(-1*P[2],-2*P[4])])
        F = np.array([P[1],P[3]])
        c = np.linalg.solve(A,F)
        tan_2_omega = (P[2])/(P[0]-P[4])
        omega = np.arctan(tan_2_omega)/2
        return c,omega

    def determine_ab(self,x,y,c,omega):
        u = (x - c[0])*np.cos(omega)+(y-c[1])*np.sin(omega)
        v = -1*(x - c[0])*np.sin(omega)+(y-c[1])*np.cos(omega)

        D_1 = np.sum(u**2)
        D_2 = np.sum(v**2)
        C_11 = np.sum(u**4)
        C_12 = np.sum(u**2*v**2)
        C_22 = np.sum(v**4)

        A = np.array([(C_11,C_12),(C_12,C_22)])
        F = np.array([D_1,D_2])
        ax = np.linalg.solve(A,F)
        ax[0] = 1/np.sqrt(ax[0])
        ax[1] = 1/np.sqrt(ax[1])
        return ax

    def determine_transformation(self,x,y):
        P = self.determine_ABCDEF(x,y)
        c,omega = self.determine_center_orientation(P)
        ax = self.determine_ab(x,y,c,omega)

        t_val = np.array([c[0],c[1],omega*(180/np.pi),ax[0],ax[1]])

        return t_val
        
    def calculate_baseline_trans(self):
        nb_temp = self.ms.na*(self.ms.na-1)/2+self.ms.na
        self.baseline_parm = np.zeros((nb_temp-self.ms.na,5),dtype=float)
        self.baseline_parm_n = np.zeros((nb_temp-self.ms.na,5),dtype=float)
        self.baseline_l = np.zeros((nb_temp-self.ms.na,),dtype=float)
        self.b_number = np.zeros((self.ms.na,self.ms.na),dtype=float)
        self.theta = np.zeros((nb_temp-self.ms.na,),dtype=float)

        b_counter = 0
        for j in xrange(self.ms.na):
            for k in xrange(j+1,self.ms.na):
                u_l = self.ms.uvw[(self.ms.A1==j)&(self.ms.A2==k),0]#/self.ms.wave
                v_l = self.ms.uvw[(self.ms.A1==j)&(self.ms.A2==k),1]#/self.ms.wave
                if u_l <> np.array([]) and v_l <> np.array([]):
                   self.baseline_parm[b_counter,:] = self.determine_transformation(u_l,v_l)
                   self.baseline_parm_n[b_counter,:] = self.determine_transformation(-1*u_l,-1*v_l)
                   self.baseline_l[b_counter] = np.sqrt((self.ms.pos[k,0]-self.ms.pos[j,0])**2 + (self.ms.pos[k,1]-self.ms.pos[j,1])**2)
                   delta_z = self.ms.pos[j,2]-self.ms.pos[k,2]
                   print "**************************************"
                   print "baseline = ",j
                   print "baseline = ",k
                   print "delta_z = ",delta_z
                   print "self.baseline_parm[b_counter,1] = ",self.baseline_parm[b_counter,1]/np.cos(self.ms.dec0)                     
                   print "**************************************"
                   u_new = (u_l[0])/self.baseline_parm[b_counter,3]
                   v_new = (v_l[0]-self.baseline_parm[b_counter,1])/(self.baseline_parm[b_counter,3]*np.sin(self.ms.dec0)) 
                   #v_new = (v_l[0]-self.baseline_parm[b_counter,1])/self.baseline_parm[b_counter,4] 
                   self.theta[b_counter] = np.arccos(u_new)*np.sign(v_new)
                self.b_number[j,k] = b_counter
                b_counter = b_counter + 1
                            
    def test_angle(self,b1,b2):
        print "#############################"
        print "b1 = ",b1
        print "b2 = ",b2
        delta_x = self.ms.pos[b1,0]-self.ms.pos[b2,0]
        delta_y = self.ms.pos[b1,1]-self.ms.pos[b2,1]
        print "X = ",delta_x
        print "Y = ",delta_y
        print "Size = ",np.sqrt(delta_x**2+delta_y**2)
        delta_x_n = delta_x/np.sqrt(delta_x**2 + delta_y**2)
        delta_y_n = delta_y/np.sqrt(delta_x**2 + delta_y**2)
        
        plt.plot([0,delta_x_n],[0,delta_y_n],"b")
        plt.hold("on")
        
        alpha = np.arctan2(delta_y_n,delta_x_n)

        print "alpha = ",alpha*(180/np.pi)

        u = self.ms.uvw[(self.ms.A1==b1)&(self.ms.A2==b2),0]#/self.ms.wave
        v = self.ms.uvw[(self.ms.A1==b1)&(self.ms.A2==b2),1]#/self.ms.wave
             
        u_0 = u[0]
        v_0 = v[0]

        y_c = self.c_m[b1,b2]
        a = self.phi_m[b1,b2]
        theta_v = self.theta_m[b1,b2]
        
        v_0 = v_0 - y_c*np.cos(self.ms.dec0)

        u_0 = u_0/a
        v_0 = v_0/(self.sin_delta*a)

        print "np.sqrt(u_0**2 + v_0**2) = ",np.sqrt(u_0**2+ v_0**2)

        plt.plot([0,u_0],[0,v_0],"r")

        theta = -1*np.arctan2(v_0,u_0)
        print "theta = ",theta*(180/np.pi)
        print "theta_v = ",theta_v*(180/np.pi)
        H = 90 - (theta+alpha)*(180/np.pi)
        #H = 90 - (theta+alpha)*(180/np.pi)
        print "H = ",H
        plt.xlim([-1,1])
        plt.ylim([-1,1]) 
        print "##########################################"  
        plt.show()

    def plot_test(self,b1,b2,c="b"):
        u = self.ms.uvw[(self.ms.A1==b1)&(self.ms.A2==b2),0]#/self.ms.wave
        v = self.ms.uvw[(self.ms.A1==b1)&(self.ms.A2==b2),1]#/self.ms.wave
        a = self.phi_m[b1,b2]
        a_n = self.phi_m[b2,b1]
        y_c = self.c_m[b1,b2]
        theta_v = self.theta_m[b1,b2]
        theta_v_n = self.theta_m[b2,b1]
        y_c_n = self.c_m[b2,b1]
        b = self.baseline_parm[self.b_number[b1,b2],4]  
        b_n = -1*b 
        u_0,v_0 = self.generate_xy(0,0,0,1,1,200,1,True)
        #plt.plot(u_0,v_0)
        #plt.plot(u_0[0],v_0[0],"bs")
        #plt.plot(u_0[-1],v_0[-1],"bv")
        #plt.show()
      
      
        #ADDITION
        v_t = v - y_c
        u_t = u
        #SCALING
        u_t = u_t/a
        v_t = v_t/(self.sin_delta*a)
        #ROTATION (Clockwise)
        u_new = u_t*np.cos(theta_v) + v_t*np.sin(theta_v)
        v_new = -1*u_t*np.sin(theta_v) + v_t*np.cos(theta_v)
      
        #ROTATION
        #u_new = u_0*np.cos(theta_v) - v_0*np.sin(theta_v)
        #v_new = u_0*np.sin(theta_v) + v_0*np.cos(theta_v)

        #u_new_n = u_0*np.cos(theta_v) - v_0*np.sin(theta_v)
        #v_new_n = u_0*np.sin(theta_v) + v_0*np.cos(theta_v)


        #SCALING
        #u_new = a*u_new 
        #v_new = b*v_new + y_c
    
        #u_new_n = a_n*u_new_n 
        #v_new_n = b_n*v_new_n + y_c_n
      
 
        #plt.plot(u,v,"b")
        #plt.hold("on")
        #plt.plot(u[0],v[0],"bs")
        #plt.plot(u[-1],v[-1],"bv")
        plt.plot(u_new,v_new,c) 
        plt.plot(u_new[0],v_new[0],c+"s")
        plt.plot(u_new[-1],v_new[-1],c+"v")
        plt.plot([0,u_new[0]],[0,v_new[0]],c)
        plt.plot([0,u_new[100]],[0,v_new[100]],c)
        plt.plot([0,u_new[200]],[0,v_new[200]],c)
        plt.plot([0,u_new[300]],[0,v_new[300]],c)
        plt.plot([0,u_new[400]],[0,v_new[400]],c)
        plt.plot([0,u_new[500]],[0,v_new[500]],c)
        plt.plot([0,u_new[600]],[0,v_new[600]],c)
        plt.plot([0,u_new[700]],[0,v_new[700]],c)
        #plt.show() 
        #plt.plot(-1*u,-1*v,"g")
        plt.hold("on")
        #plt.plot(-1*u[0],-1*v[0],"gs")
        #plt.plot(-1*u[-1],-1*v[-1],"gv")
        #plt.plot(u_new_n,v_new_n,"c") 
        #plt.plot(u_new_n[0],v_new_n[0],"cs")
        #plt.plot(u_new_n[-1],v_new_n[-1],"cv")
        #plt.show() 
        #u_new_n = a_n*u_0 
        #v_new_n = a_n*v_0 + y_c_n

        #u = (u)/a
        #v = (v-y_c)/b 

        #plt.plot([0,u[0]],[0,v[0]],c)
        #plt.hold('on')
        #plt.plot([0,u[200]],[0,v[200]],c)
        #plt.plot(u,v,c)
        #plt.hold('on')
        #plt.plot(u[0],v[0],c+"s")
        #plt.plot(u[-1],v[-1],c+"v")

        #dot_p = u[0]*u[-1] + v[0]*v[-1]
        #s_b = np.sqrt(u[0]**2 + v[0]**2)
        #s_e = np.sqrt(u[-1]**2 + v[-1]**2)

        #answ = dot_p/(s_b*s_e)

        #print "dot_p = ",dot_p
        #print "s_b = ",s_b
        #print "s_e = ",s_e
        #print "answ = ",answ

        #theta = np.arccos(answ)
      
        #print "theta1 = ", theta*(180/np.pi)
        #print "theta2 = ", np.arccos(u[0])*(180/np.pi)
        #if (answ > 0):
        #   theta = theta
        #else:
        #   theta = theta + np.pi/2

        #print "theta = ", theta*(180/np.pi)
        #plt.plot(u_new,v_new,"r") 
        #plt.plot(u_new_n,v_new_n,"r") 

        #plt.show()

    def create_phi_c(self):
        self.phi_m = np.zeros((self.ms.na,self.ms.na),dtype=float)
        self.l = np.zeros((self.ms.na,self.ms.na),dtype=float)
        self.c_m = np.zeros((self.ms.na,self.ms.na),dtype=float)
        self.theta_m = np.zeros((self.ms.na,self.ms.na),dtype=float)

        #self.sin_delta = np.mean(self.baseline_parm[:,4]/self.baseline_parm[:,3])
        self.sin_delta = np.sin(self.ms.dec0)
        print "self.sin_delta = ",self.sin_delta
        print "sin(delta) = ",np.absolute(np.sin(self.ms.dec0))      
        for j in xrange(self.ms.na):
            for k in xrange(j+1,self.ms.na):
                y_c = self.baseline_parm[self.b_number[j,k],1]
                b = self.baseline_parm[self.b_number[j,k],3]
                l = self.baseline_l[self.b_number[j,k]]
                theta_v = self.theta[self.b_number[j,k]]
                self.phi_m[k,j] = -1*b
                self.phi_m[j,k] = b
                self.c_m[k,j] = -1*y_c/(np.cos(self.ms.dec0))          
                self.c_m[j,k] = y_c/(np.cos(self.ms.dec0))          
                self.theta_m[j,k] = theta_v
                self.theta_m[k,j] = theta_v
                self.l[k,j] = l
                self.l[j,k] = l
        
        if exists("${DESTDIR>/}Pickle/"):
           x.sh("rm -fr ${DESTDIR>/}Pickle/")

        x.sh("mkdir ${DESTDIR>/}Pickle/")

        v.PICKLENAME = "baseline_parm"
        file_name = self.ms.p_wrapper.pyxis_to_string(v.PICKLEFILE)
        pickle.dump(self.baseline_parm, open(file_name, 'wb'))
        v.PICKLENAME = "b_counter"
        file_name = self.ms.p_wrapper.pyxis_to_string(v.PICKLEFILE)
        pickle.dump(self.b_number, open(file_name, 'wb'))
        

        print "self.phi_m = ",self.phi_m
        v.PICKLENAME = "phi_m"
        file_name = self.ms.p_wrapper.pyxis_to_string(v.PICKLEFILE)
        pickle.dump(self.phi_m, open(file_name, 'wb'))
        print "self.b_m = ",self.c_m
        v.PICKLENAME = "b_m"
        file_name = self.ms.p_wrapper.pyxis_to_string(v.PICKLEFILE)
        pickle.dump(self.c_m, open(file_name, 'wb'))
        print "self.theta_m = ",self.theta_m*(180/np.pi)
        v.PICKLENAME = "theta_m"
        file_name = self.ms.p_wrapper.pyxis_to_string(v.PICKLEFILE)
        print "02-01 = ",(self.theta_m[0,2]-self.theta_m[0,1])*(180/np.pi)
        print "14-45 = ",(self.theta_m[1,4]-self.theta_m[4,5])*(180/np.pi)
        print "35-46 = ",(self.theta_m[3,5]-self.theta_m[4,6])*(180/np.pi)
        pickle.dump(self.theta_m, open(file_name, 'wb'))
        v.PICKLENAME = "sin_delta"
        file_name = self.ms.p_wrapper.pyxis_to_string(v.PICKLEFILE)
        pickle.dump(self.sin_delta, open(file_name, 'wb'))
        v.PICKLENAME = "antnames"
        file_name = self.ms.p_wrapper.pyxis_to_string(v.PICKLEFILE)
        pickle.dump(self.ms.names, open(file_name, 'wb'))
        v.PICKLENAME = "wave"
        file_name = self.ms.p_wrapper.pyxis_to_string(v.PICKLEFILE)
        pickle.dump(self.ms.wave, open(file_name, 'wb'))
        v.PICKLENAME = "declination"
        file_name = self.ms.p_wrapper.pyxis_to_string(v.PICKLEFILE)
        pickle.dump(self.ms.dec0, open(file_name, 'wb'))
           
        #pickle.dump(self.c_m, open(self.MS[2:-4]+'_b_m.p', 'wb'))
        #pickle.dump(self.theta_m, open(self.MS[2:-4]+'_theta_m.p', 'wb'))
        #pickle.dump(self.sin_delta, open(self.MS[2:-4]+'_sin_delta.p','wb'))

class Pyxis_helper():

   def __init__(self):
       pass
   
   def flip_fits(self):
       #image = self.pyxis_to_string("${imager.RESTORED_IMAGE}")
       #ff = pyfits.open(image,mode="update")
       #ff[0].data[0,0,:,:] = ff[0].data[0,0,::-1,:]
       #ff.close()
       image = self.pyxis_to_string("${imager.DIRTY_IMAGE}")
       ff = pyfits.open(image,mode="update")
       ff[0].data[0,0,:,:] = ff[0].data[0,0,::-1,:]
       ff.close()

   def negative_fits(self):
       image = self.pyxis_to_string("${imager.RESTORED_IMAGE}")
       ff = pyfits.open(image,mode="update")
       ff[0].data[0,0,:,:] = ff[0].data[0,0,:,:]*-1
       ff.close()
       image = self.pyxis_to_string("${imager.DIRTY_IMAGE}")
       ff = pyfits.open(image,mode="update")
       ff[0].data[0,0,:,:] = ff[0].data[0,0,:,:]*-1
       ff.close()

   def pybdsm_search_pq(self,baseline):
       self.setting_BASELINE(baseline)
       v.POSNEG = "pos"
       lsm.pybdsm_search()
       self.negative_fits()
       v.POSNEG = "neg"
       lsm.pybdsm_search()
       self.negative_fits()

   def make_image_with_mask(self,options={},column="CORRECTED_DATA"):
       empty_image = self.pyxis_to_string("${DESTDIR>/}${MS:BASE}_empty.fits")
       imager.make_empty_image(image=empty_image)
       ff = pyfits.open(empty_image,mode='update')
       ff[0].data[:] = 0
       ff.close()
       mask_im = self.pyxis_to_string("${DESTDIR>/}${MS:BASE}_mask_temp.fits")
       x.sh("tigger-restore $empty_image ${DESTDIR>/}${GHOSTMAP:BASE}.lsm.html $mask_im -b 15.0 -f")
       peak_flux = pyfits.open(mask_im)[0].data[...].max()
       sigmas = [0.682689492137086,0.954499736103642,0.997300203936740,0.999936657516334,0.999999426696856]
       sigma_val = sigmas[3] # 3Sigma 
       thresh = peak_flux*(1-sigma_val) 
       imager.make_threshold_mask(input=mask_im,threshold=thresh) # make mask
       imager.make_image(column=column,dirty=options,restore=options,restore_lsm=False,mask=True) # Clean with a mask   

   
   #Setting standard pyxis imager settings
   def image_settings(self,npix=4096,cellsize="20arcsec",mode = "channel",stokes="I",weight="natural",filter="16arcsec,16arcsec,0deg",wprojplanes=0,niter=1000,gain=0.1,threshold=0,clean_alg="hogbom"):
       imager.npix = npix
       imager.cellsize=cellsize
       imager.stokes=stokes
       imager.weight=weight
       imager.filter = filter
       imager.wprojplanes = wprojplanes
       imager.niter = niter
       imager.gain = gain
       imager.threshold = threshold
       imager.mode = mode
       imager.CLEAN_ALGORITHM = clean_alg

   #Setting global variable v.BASELINE
   def setting_BASELINE(self,antenna = "all"):
       antenna_temp_str = ""
       if antenna <> "all":
          for k in xrange(len(antenna)):
              antenna_temp_str = antenna_temp_str + str(antenna[k])+"_"
          v.BASELINE = antenna_temp_str[:-1]
       else:
          v.BASELINE = ""

   #Creating advanced pyxis imager settings
   def image_advanced_settings(self,antenna="all",img_nchan=1,img_chanstart=0,img_chanstep=1):
       #Creates baseline name and select string for image
       self.setting_BASELINE(antenna)
       if antenna <> "all":
          antenna_str = "("
          #antenna_str = " && ("
          for k in range(len(antenna)):
              for i in range(k+1,len(antenna)):
                  antenna_str = antenna_str + "(ANTENNA1 = "+str(antenna[k])+" && ANTENNA2 = "+str(antenna[i])+") || "
                  #print "antenna_str = ",antenna_str
          antenna_str = antenna_str[0:len(antenna_str)-4]
          antenna_str = antenna_str+")"
       else:
         antenna_str = ""

       #strp=""" " """
       #strsel=((strp+"sumsqr(UVW[:2])<16.e6"+antenna_str+strp)).replace(" ","")
       strsel = antenna_str

       options = {}
       options["select"] = strsel
       options["img_nchan"] = img_nchan
       options["img_chanstart"] = img_chanstart
       options["img_chanstep"] = img_chanstep
       #options["operation"] = operation
       #options2 = {}
       #options2["operation"] = operation
       return options

   #Converting pyxis notation to string
   def pyxis_to_string(self,variable):
       variable = interpolate_locals("variable")
       return variable

class Mset(object):
  def __init__(self, ms_name):

    self.p_wrapper = Pyxis_helper()
    self.wave = 0
    self.na = 0
    self.nb = 0
    self.ns = 0
    self.pos =np.array([])
    self.uvw =np.array([])
    self.corr_data = np.array([])
    self.data = np.array([])
    self.A1 = np.array([])
    self.A2 = np.array([])
    self.ra0 = 0
    self.dec0 = 0
    v.MS = ms_name

  def extract(self):
      ms_v = self.p_wrapper.pyxis_to_string("${MS>/}")
      tl=table(ms_v,readonly=False)
      self.A1=tl.getcol("ANTENNA1")
      self.A2=tl.getcol("ANTENNA2")
      self.uvw=tl.getcol("UVW")
      self.corr_data=tl.getcol("CORRECTED_DATA")
      self.data = tl.getcol("DATA")
      ta=table(ms_v+"ANTENNA")
      tab=table(ms_v+"SPECTRAL_WINDOW",readonly=False)
      self.wave=3e8/tab.getcol("REF_FREQUENCY")[0]
      print "self.wave = ",self.wave 
      self.pos = ta.getcol("POSITION")
      self.na=len(ta.getcol("NAME"))
      self.names = ta.getcol("NAME")
      temp = self.uvw[(self.A1==0)&(self.A2==0),0]
      if len(temp) == 0:
         self.auto = False
         self.nb=self.na*(self.na-1)/2
      else: 
         self.nb=self.na*(self.na-1)/2+self.na
      temp = self.uvw[(self.A1==0)&(self.A2==1),0] 
      self.ns=len(temp)
      tf = table(ms_v+"/FIELD")
      phase_centre = (tf.getcol("PHASE_DIR"))[0,0,:]
      self.ra0, self.dec0 = phase_centre[0], phase_centre[1] #Field centre in radians
      print "self.dec0 = ",self.dec0*(180/np.pi)
      tl.close()
      ta.close()
      tab.close()
      tf.close()

  def write(self,column):
      ms_v = self.p_wrapper.pyxis_to_string("${MS>/}")
      tl=table(ms_v, readonly=False)
      if column == "CORRECTED_DATA":
         tl.putcol(column ,self.corr_data)
      else:
         tl.putcol(column ,self.data)
      tl.close()

class Sky_model(object):

  def __init__(self, ms , point_sources):

        self.ms = ms
        self.point_sources = point_sources
        self.std_point = 0.0001
        self.total_flux = self.point_sources[0,0]+self.point_sources[1,0]

  def visibility(self,column):
        u=self.ms.uvw[:,0]
        #print u.shape
        v=self.ms.uvw[:,1]
        if column=="CORRECTED_DATA":
           vis=np.zeros((self.ms.corr_data.shape[0],),dtype=self.ms.data.dtype)
           #print self.ms.data.shape[0]
        else:
           vis=np.zeros((self.ms.data.shape[0],),dtype=self.ms.data.dtype)

        vis = self.point_sources[0,0]*np.exp((-2*np.pi*1j*(u*self.point_sources[0,1]+v*self.point_sources[0,2]))/self.ms.wave)
        vis = vis + self.point_sources[1,0]*np.exp((-2*np.pi*1j*(u*self.point_sources[1,1]+v*self.point_sources[1,2]))/self.ms.wave)
        if column=="CORRECTED_DATA":
           self.ms.corr_data[:,0,3]=vis
           self.ms.corr_data[:,0,0]=vis
        else:
           self.ms.data[:,0,3]=vis
           self.ms.data[:,0,0]=vis

        self.ms.write(column)

  # converting from l and m coordinate system to ra and dec
  def lm2radec(self,l,m):#l and m in radians
      rad2deg = lambda val: val * 180./np.pi
      #ra0,dec0 = extract(MS) # phase centre in radians
      rho = np.sqrt(l**2+m**2)
      if rho==0:
         ra = self.ms.ra0
         dec = self.ms.dec0
      else:
         cc = np.arcsin(rho)
         ra = self.ms.ra0 - np.arctan2(l*np.sin(cc), rho*np.cos(self.ms.dec0)*np.cos(cc)-m*np.sin(self.ms.dec0)*np.sin(cc))
         dec = np.arcsin(np.cos(cc)*np.sin(self.ms.dec0) + m*np.sin(cc)*np.cos(self.ms.dec0)/rho)
      return rad2deg(ra), rad2deg(dec)

  # converting ra and dec to l and m coordiantes
  def radec2lm(self,ra_d,dec_d):# ra and dec in degrees
      rad2deg = lambda val: val * 180./np.pi
      deg2rad = lambda val: val * np.pi/180
      #ra0,dec0 = extract(MS) # phase center in radians
      ra_r, dec_r = deg2rad(ra_d), deg2rad(dec_d) # coordinates of the sources in radians
      l = np.cos(dec_r)* np.sin(ra_r - self.ms.ra0)
      m = np.sin(dec_r)*np.cos(self.ms.dec0) - np.cos(dec_r)*np.sin(self.ms.dec0)*np.cos(ra_r-self.ms.ra0)
      return rad2deg(l),rad2deg(m)

  # creating meqtrees skymodel    
  def meqskymodel(self,point_sources,antenna=""): 
      str_out = "#format: name ra_d dec_d i\n"
      for i in range(len(point_sources)):
          print "i = ",i
          amp, l ,m = point_sources[i,0], point_sources[i,1], point_sources[i,2] 
          #m = np.absolute(m) if m < 0 else -1*m # changing the signs since meqtrees has its own coordinate system
          ra_d, dec_d = self.lm2radec(l,m)
          print ra_d, dec_d
          l_t,m_t = self.radec2lm(ra_d,dec_d)
          print l_t, m_t
          name = "A"+ str(i)
          str_out += "%s %.10g %.10g %.4g\n"%(name, ra_d, dec_d,amp)
      self.ms.p_wrapper.setting_BASELINE(antenna) 
      file_name = self.ms.p_wrapper.pyxis_to_string(v.GHOSTMAP)
      simmodel = open(file_name,"w")
      simmodel.write(str_out)
      simmodel.close()
      x.sh("tigger-convert $GHOSTMAP -t ASCII --format \"name ra_d dec_d i\" -f ${DESTDIR>/}${GHOSTMAP:BASE}.lsm.html")

class Calibration(object):
   def __init__(self, ms, antenna, cal_tech,total_flux):

        self.antenna = antenna
        self.ms = ms
        self.a_list = self.get_antenna(self.antenna,self.ms.names)
        #self.a_names = self.ms.names[self.a_list]
        print "a_list = ",self.a_list
        v.CALTECH = cal_tech
        #self.cal_tech = cal_tech
        self.R = np.array([])
        self.p = np.array([])
        self.q = np.array([])
        self.G = np.array([])
        self.g = np.array([])
        self.u_m = np.array([])
        self.v_m = np.array([])
        self.total_flux = total_flux

   def get_antenna(self,ant,ant_names):
       if isinstance(ant[0],int) :
          return np.array(ant)
       if ant == "all":
          return np.arange(len(ant_names))
       new_ant = np.zeros((len(ant),))
       for k in xrange(len(ant)):
           for j in xrange(len(ant_names)):
               if (ant_names[j] == ant[k]):
                 new_ant[k] = j
       return new_ant

   def calculate_delete_list(self):
       if self.antenna == "all":
          return np.array([])
       d_list = list(xrange(self.ms.na))
       for k in range(len(self.a_list)):
           d_list.remove(self.a_list[k])
       return d_list

   def read_R(self,column):
       self.R = np.zeros((self.ms.na,self.ms.na,self.ms.ns),dtype=complex)
       self.u_m = np.zeros((self.ms.na,self.ms.na,self.ms.ns),dtype=complex)
       self.v_m = np.zeros((self.ms.na,self.ms.na,self.ms.ns),dtype=complex)
       
       self.p = np.ones((self.ms.na,self.ms.na),dtype = int)
       self.p = np.cumsum(self.p,axis=0)-1
       self.q = self.p.transpose()

       for j in xrange(self.ms.na):
           for k in xrange(j+1,self.ms.na):
               #print "j = ",j
               #print "k = ",k
               if column == "CORRECTED_DATA":
                  r_jk = self.ms.corr_data[(self.ms.A1==j) & (self.ms.A2==k),0,0]
               else:
                  r_jk = self.ms.data[(self.ms.A1==j) & (self.ms.A2==k),0,0]
               
               #print "self.ms.A1 = ",self.ms.A1
               
               #print "antenna_b = ",(self.ms.A1==j) & (self.ms.A2==k)

               #print "r_jk = ",r_jk
               
               u_jk = self.ms.uvw[(self.ms.A1==j)&(self.ms.A2==k),0]
               v_jk = self.ms.uvw[(self.ms.A1==j)&(self.ms.A2==k),1]
               
               self.R[j,k,:] = r_jk
               self.R[k,j,:] = r_jk.conj()
               self.u_m[j,k,:] = u_jk 
               self.u_m[k,j,:] = -1*u_jk
               self.v_m[j,k,:] = v_jk
               self.v_m[k,j,:] = -1*v_jk                

           if self.ms.auto:
              self.R[j,j,:] = self.ms.corr_data[(self.A1==j)&(self.A2==j),0,0]
           else: 
              self.R[j,j,:] = self.total_flux  
       
       if self.antenna <> "all":
          d_list = self.calculate_delete_list()
          #print "d_list = ",d_list
          self.R = np.delete(self.R,d_list,axis = 0)
          self.R = np.delete(self.R,d_list,axis = 1)
          self.u_m = np.delete(self.u_m,d_list,axis = 0)
          self.u_m = np.delete(self.u_m,d_list,axis = 1)
          self.v_m = np.delete(self.v_m,d_list,axis = 0)
          self.v_m = np.delete(self.v_m,d_list,axis = 1)
          self.p = np.delete(self.p,d_list,axis = 0)
          self.p = np.delete(self.p,d_list,axis = 1)
          self.q = np.delete(self.q,d_list,axis = 0)
          self.q = np.delete(self.q,d_list,axis = 1)
       #print "self.p = ",self.p
       #print "self.q = ",self.q 
       print "R[:,:,200] = ",self.R[:,:,200] 
              
   def cal_G_eig(self):
        #U_n=np.zeros(self.R.shape,dtype=complex)
        #S_n=np.zeros(self.R.shape, dtype=complex)
        #V_n=np.zeros(self.R.shape, dtype=complex)
        D =np.zeros(self.R.shape, dtype=complex)
        Q=np.zeros(self.R.shape, dtype=complex)
        self.g=np.zeros((self.R.shape[0],self.ms.ns) , dtype=complex)
        self.G=np.zeros(self.R.shape ,dtype=complex)
        temp =np.ones((self.R.shape[0],self.R.shape[0]) ,dtype=complex)
        #print "self.R.shape = ",self.R.shape
        #print "self.R = ",self.R
        for t in range(self.ms.ns):
           #print "t=",t
           d,Q[:,:,t] = np.linalg.eigh(self.R[:,:,t])
           D[:,:,t] = np.diag(d)
           Q_H = Q[:,:,t].conj().transpose()
           #R_2 = np.dot(Q[:,:,t],np.dot(D[:,:,t],Q_H))
           abs_d=np.absolute(d)
           index=abs_d.argmax()
           if (d[index] > 0):
             self.g[:,t]=Q[:,index,t]*np.sqrt(d[index])
           else:
             self.g[:,t]=Q[:,index,t]*np.sqrt(np.absolute(d[index]))*1j

           self.G[:,:,t] = np.dot(np.diag(self.g[:,t]),temp)
           self.G[:,:,t]= np.dot (self.G[:,:,t] ,np.diag(self.g[:,t].conj()))

        self.G = self.G
        #print "G = ",self.G[:,:,200]

   def write_to_MS(self,column,type_w):
       if column == "CORECTED_DATA":
          t_data = self.ms.corr_data
       else:
          t_data = self.ms.data

       if self.ms.auto:
          s = 0
       else:
          s = 1

       for j in xrange(self.R.shape[0]):
           for k in xrange(j+s,self.R.shape[0]):
               if type_w == "R":
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.R[j,k,:]
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.R[j,k,:]
               elif type_w == "G":
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.G[j,k,:]
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.G[j,k,:]
               elif type_w == "GT":
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.G[j,k,:]**(-1)
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.G[j,k,:]**(-1)
               elif type_w == "GTR":
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.G[j,k,:]**(-1)*self.R[j,k,:]
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.G[j,k,:]**(-1)*self.R[j,k,:]
               elif type_w == "R-1":
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.R[j,k,:]-1
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.R[j,k,:]-1
               elif type_w == "G-1":
                  #t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.G[j,k,:]-1.03-0.07*np.exp(-2*np.pi*1j*(self.u_m[j,k,:]/self.ms.wave)*(1*(np.pi)/180))
                  #t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.G[j,k,:]-1.03-0.07*np.exp(-2*np.pi*1j*(self.u_m[j,k,:]/self.ms.wave)*(1*(np.pi)/180))
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.G[j,k,:]-1
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.G[j,k,:]-1
               elif type_w == "GT-1":
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.G[j,k,:]**(-1)-1
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.G[j,k,:]**(-1)-1
               elif type_w == "GTR-R":
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.G[j,k,:]**(-1)*self.R[j,k,:]-self.R[j,k,:]
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.G[j,k,:]**(-1)*self.R[j,k,:]-self.R[j,k,:]
                 
       if column == "CORRECTED_DATA":
          self.ms.corr_data = t_data
       else:
          self.ms.data = t_data

       self.ms.write(column)
       
class T_ghost():
    def __init__(self,
                point_sources = np.array([]),
                antenna = "",
                MS=""):
        self.p_wrapper = Pyxis_helper()
        self.antenna = antenna
        self.A_1 = point_sources[0,0]
        self.A_2 = point_sources[1,0]
        self.l_0 = point_sources[1,1]
        self.m_0 = point_sources[1,2]
        v.MS = MS

        if v.MS == "EW_EXAMPLE":
           self.ant_names = [0,1,2]
           self.a_list = self.get_antenna(self.antenna,self.ant_names)
           self.b_m = np.zeros((3,3))          
           self.theta_m = np.zeros((3,3))          
           self.phi_m = np.array([(0,3,5),(-3,0,2),(-5,-2,0)]) 
           self.sin_delta = None
           self.wave = 3e8/1.45e9
           self.dec = np.pi/2.0       
             
        else:
           v.PICKLENAME = "antnames"
           file_name = self.p_wrapper.pyxis_to_string(v.PICKLEFILE)
           self.ant_names = pickle.load(open(file_name,"rb"))

           #print "ant_names = ",self.ant_names
           self.a_list = self.get_antenna(self.antenna,self.ant_names)
           print "a_list = ",self.a_list
       
           v.PICKLENAME = "phi_m"
           file_name = self.p_wrapper.pyxis_to_string(v.PICKLEFILE)
           self.phi_m = pickle.load(open(file_name,"rb"))
           #self.phi_m =  pickle.load(open(MS[2:-4]+"_phi_m.p","rb"))
        
           v.PICKLENAME = "b_m"
           file_name = self.p_wrapper.pyxis_to_string(v.PICKLEFILE)
           self.b_m = pickle.load(open(file_name,"rb"))
           #self.b_m = pickle.load(open(MS[2:-4]+"_b_m.p","rb"))

           v.PICKLENAME = "theta_m"
           file_name = self.p_wrapper.pyxis_to_string(v.PICKLEFILE)
           self.theta_m = pickle.load(open(file_name,"rb"))
           #self.theta_m = pickle.load(open(MS[2:-4]+"_theta_m.p","rb"))
        
           v.PICKLENAME = "sin_delta"
           file_name = self.p_wrapper.pyxis_to_string(v.PICKLEFILE)
           self.sin_delta = pickle.load(open(file_name,"rb"))
           #self.sin_delta = pickle.load(open(MS[2:-4]+"_sin_delta.p","rb"))
           self.sin_delta = None

           v.PICKLENAME = "wave"
           file_name = self.p_wrapper.pyxis_to_string(v.PICKLEFILE)
           self.wave = pickle.load(open(file_name,"rb"))

           v.PICKLENAME = "declination"
           file_name = self.p_wrapper.pyxis_to_string(v.PICKLEFILE)
           self.dec = pickle.load(open(file_name,"rb"))
           print "phi_m = ",self.phi_m
           print "b_m = ",self.b_m
           print "theta_m =",self.theta_m 
           print "dec = ",self.dec*(180/np.pi)
           print "phi_m/lambda = ",self.phi_m/self.wave
           print "b_m = ",(self.b_m*np.cos(self.dec))/self.wave
           print "1/wave =",3e8/self.wave
           print "wave =",self.wave
           
           self.n = len(self.a_list)
           self.norm_f = (self.n**2-self.n)/2
           print "n =",self.n

    def get_antenna(self,ant,ant_names):
        if isinstance(ant[0],int) :
           return np.array(ant)
        if ant == "all":
           return np.arange(len(ant_names))
        new_ant = np.zeros((len(ant),))
        for k in xrange(len(ant)):
            for j in xrange(len(ant_names)):
                if (ant_names[j] == ant[k]):
                   new_ant[k] = j
        return new_ant
        
    def calculate_delete_list(self):
        if self.antenna == "all":
           return np.array([])
        d_list = list(xrange(len(self.ant_names)))
        for k in range(len(self.a_list)):
            d_list.remove(self.a_list[k])
        return d_list

    def plot_visibilities_pq(self,baseline,u=None,v=None,resolution=0,image_s=0,s=0,wave=None,dec=None,approx=False):
        if wave == None:
           wave = self.wave
        if dec == None:
           dec = self.dec
        u_temp = u
        v_temp = v
        u,v,V_G_pq,V_R_pq,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline,resolution=resolution,image_s=image_s,s=s,wave=wave,dec=dec)
       
        uu,vv = np.meshgrid(u,v)
        
        fig,axs = plt.subplots(2,2)
 
        cs = axs[1,0].contourf(uu,vv,V_G_pq.real)
        axs[1,0].set_title("Real---$g_{%s,%s}(X_{%s,%s}^{-1}(u,v))$"%(str(baseline[0]),str(baseline[1]),str(baseline[0]),str(baseline[1])))
        axs[1,0].set_xlabel("$u$ [1/rad]")
        axs[1,0].set_ylabel("$v$ [1/rad]")
        fig.colorbar(cs,ax=axs[1,0],use_gridspec=True,shrink=0.9) #ax.set_title("extend = %s" % extend)
        if u_temp <> None:
           axs[1,0].plot(u_temp,v_temp,'k')
        cs = axs[1,1].contourf(uu,vv,V_G_pq.imag)
        axs[1,1].set_title("Imaginary---$g_{%s,%s}(X_{%s,%s}^{-1}(u,v))$"%(str(baseline[0]),str(baseline[1]),str(baseline[0]),str(baseline[1])))
        axs[1,1].set_xlabel("$u$ [1/rad]")
        axs[1,1].set_ylabel("$v$ [1/rad]")
        fig.colorbar(cs,ax=axs[1,1],use_gridspec=True,shrink=0.9) #ax.set_title("extend = %s" % extend)
        if u_temp <> None:
           axs[1,1].plot(u_temp,v_temp,'k')
        cs = axs[0,0].contourf(uu,vv,V_R_pq.real)
        axs[0,0].set_title("Real---$r_{%s,%s}(X_{%s,%s}^{-1}(u,v))$"%(str(baseline[0]),str(baseline[1]),str(baseline[0]),str(baseline[1])))
        axs[0,0].set_xlabel("$u$ [1/rad]")
        axs[0,0].set_ylabel("$v$ [1/rad]")
        fig.colorbar(cs,ax=axs[0,0],use_gridspec=True,shrink=0.9) #ax.set_title("extend = %s" % extend)
        if u_temp <> None:
           axs[0,0].plot(u_temp,v_temp,'k')
        cs = axs[0,1].contourf(uu,vv,V_R_pq.imag)
        axs[0,1].set_title("Imaginary---$r_{%s,%s}(X_{%s,%s}^{-1}(u,v))$"%(str(baseline[0]),str(baseline[1]),str(baseline[0]),str(baseline[1])))
        axs[0,1].set_xlabel("$u$ [1/rad]")
        axs[0,1].set_ylabel("$v$ [1/rad]")
        fig.colorbar(cs,ax=axs[0,1],use_gridspec=True,shrink=0.9) #ax.set_title("extend = %s" % extend)
        if u_temp <> None:
           axs[0,1].plot(u_temp,v_temp,'k')
        plt.tight_layout()
        plt.show()

        if u_temp <> None:
           u,v,V_G_pq,V_R_pq,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline,u=u_temp,v=v_temp,resolution=resolution,image_s=image_s,s=s,wave=wave,dec=dec)
           u,v,V_G_pq_app,V_R_pq,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline,u=u_temp,v=v_temp,resolution=resolution,image_s=image_s,s=s,wave=wave,dec=dec,approx=True)
           #baseline_new = [0,0]
           #baseline_new[0] = baseline[1]
           #baseline_new[1] = baseline[0]   
           #u,v,V_G_qp,V_R_qp,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline_new,u=u_temp,v=v_temp,resolution=resolution,image_s=image_s,s=s)
           #V_R_pq = (V_R_pq + V_R_qp)/2
           
           #V_G_pq = (V_G_pq + V_G_qp)/2
           fig,axs = plt.subplots(2,2)
           print "V_G_pq = ",V_G_pq.real
           axs[1,0].set_title("Real---$g_{%s,%s}(X_{%s,%s}^{-1}(u,v))$"%(str(baseline[0]),str(baseline[1]),str(baseline[0]),str(baseline[1])))
           axs[1,0].set_xlabel("Timeslot [n]")
           #axs[1,0].set_ylim([0.8,1.2])
           axs[1,0].plot(V_G_pq.real,'k',label="True")
           if approx:
              axs[1,0].plot(V_G_pq_app.real,'r',label="Approx")
              axs[1,0].plot(V_G_pq.real-V_G_pq_app.real+np.mean(V_G_pq_app.real),'b',label="Diff+m")
              axs[1,0].legend(loc=8,ncol=3,prop={"size":9})
          
           

           axs[1,1].set_title("Imaginary---$g_{%s,%s}(X_{%s,%s}^{-1}(u,v))$"%(str(baseline[0]),str(baseline[1]),str(baseline[0]),str(baseline[1])))
           axs[1,1].set_xlabel("Timeslot [n]")
           #axs[1,1].set_ylim([-0.3,0.2])
           axs[1,1].plot(V_G_pq.imag,'k',label="True")
           if approx:
              axs[1,1].plot(V_G_pq_app.imag,'r',label="Approx")
              axs[1,1].plot(V_G_pq.imag-V_G_pq_app.imag,'b',label="Diff")
              axs[1,1].legend(loc=8,ncol=3,prop={"size":9})
           

           axs[0,0].set_title("Real---$r_{%s,%s}(X_{%s,%s}^{-1}(u,v))$"%(str(baseline[0]),str(baseline[1]),str(baseline[0]),str(baseline[1])))
           axs[0,0].set_xlabel("Timeslot [n]")
           axs[0,0].set_ylim([0.8,1.2])
           axs[0,0].plot(V_R_pq.real,'k')
           

           axs[0,1].set_title("Imaginary---$r_{%s,%s}(X_{%s,%s}^{-1}(u,v))$"%(str(baseline[0]),str(baseline[1]),str(baseline[0]),str(baseline[1])))
           axs[0,1].set_xlabel("Timeslot [n]")
           axs[0,1].plot(V_R_pq.imag,'k')
           axs[0,1].set_ylim([-0.3,0.2])
           plt.tight_layout()
           plt.show()
           
    # resolution --- arcsecond, image_s --- degrees
    def visibilities_pq_2D(self,baseline,u=None,v=None,resolution=0,image_s=0,s=0,wave=None,dec=None,approx=False):
        if wave == None:
           wave = self.wave
        if dec == None:
           dec = self.dec

        sin_delta = np.sin(dec)
        #SELECTING ONLY SPECIFIC INTERFEROMETERS
        #####################################################
        b_list = self.get_antenna(baseline,self.ant_names)
        #print "b_list = ",b_list
        d_list = self.calculate_delete_list()
        #print "d_list = ",d_list

        phi = self.phi_m[b_list[0],b_list[1]]
        delta_b = (self.b_m[b_list[0],b_list[1]]/wave)*np.cos(dec)
        theta = self.theta_m[b_list[0],b_list[1]]


        p = np.ones(self.phi_m.shape,dtype = int)
        p = np.cumsum(p,axis=0)-1
        q = p.transpose()

        if d_list == np.array([]):
            p_new = p
            q_new = q
            phi_new = self.phi_m
        else:
            p_new = np.delete(p,d_list,axis = 0)
            p_new = np.delete(p_new,d_list,axis = 1)
            q_new = np.delete(q,d_list,axis = 0)
            q_new = np.delete(q_new,d_list,axis = 1)

            phi_new = np.delete(self.phi_m,d_list,axis = 0)
            phi_new = np.delete(phi_new,d_list,axis = 1)

            b_new = np.delete(self.b_m,d_list,axis = 0)
            b_new = np.delete(b_new,d_list,axis = 1)

            b_new = (b_new/wave)*np.cos(dec)

            theta_new = np.delete(self.theta_m,d_list,axis = 0)
            theta_new = np.delete(theta_new,d_list,axis = 1)
        #####################################################

        #print "theta_new = ",theta_new
        #print "b_new = ",b_new
        #print "phi_new = ",phi_new 
        #print "delta_sin = ",self.sin_delta 

        #print "phi = ",phi
        #print "delta_b = ",delta_b
        #print "theta = ",theta*(180/np.pi)

        if u <> None:
           u_dim1 = len(u)
           u_dim2 = 1
           uu = u
           vv = v
           l_cor = None
           m_cor = None
        else:
           # FFT SCALING
           ######################################################
           delta_u = 1/(2*s*image_s*(np.pi/180))
           delta_v = delta_u
           delta_l = resolution*(1.0/3600.0)*(np.pi/180.0)
           delta_m = delta_l
           N = int(np.ceil(1/(delta_l*delta_u)))+1

           if (N % 2) == 0:
              N = N + 1

           delta_l_new = 1/((N-1)*delta_u)
           delta_m_new = delta_l_new  
           u = np.linspace(-(N-1)/2*delta_u,(N-1)/2*delta_u,N)
           v = np.linspace(-(N-1)/2*delta_v,(N-1)/2*delta_v,N)
           l_cor = np.linspace(-1/(2*delta_u),1/(2*delta_u),N)
           m_cor = np.linspace(-1/(2*delta_v),1/(2*delta_v),N)
           uu,vv = np.meshgrid(u,v)
           u_dim1 = uu.shape[0]
           u_dim2 = uu.shape[1] 
           #######################################################
        
        #DO CALIBRATION
        ####################################################### 

        V_R_pq = np.zeros(uu.shape,dtype=complex)
        V_G_pq = np.zeros(uu.shape,dtype=complex)
        temp =np.ones(phi_new.shape ,dtype=complex)

        for i in xrange(u_dim1):
            for j in xrange(u_dim2):
                if u_dim2 <> 1:
                   u_t = uu[i,j]
                   v_t = vv[i,j]
                else:
                   u_t = uu[i]
                   v_t = vv[i]
                #BASELINE CORRECTION (Single operation)
                #####################################################
                #ADDITION
                v_t = v_t - delta_b
                #SCALING
                u_t = u_t/phi
                v_t = v_t/(sin_delta*phi)
                #ROTATION (Clockwise)
                u_t_r = u_t*np.cos(theta) + v_t*np.sin(theta)
                v_t_r = -1*u_t*np.sin(theta) + v_t*np.cos(theta)
                #u_t_r = u_t
                #v_t_r = v_t
                #NON BASELINE TRANSFORMATION (NxN) operations
                #####################################################
                #ROTATION (Anti-clockwise)
                u_t_m = u_t_r*np.cos(theta_new) - v_t_r*np.sin(theta_new)
                v_t_m = u_t_r*np.sin(theta_new) + v_t_r*np.cos(theta_new)
                #u_t_m = u_t_r
                #v_t_m = v_t_r
                #SCALING
                u_t_m = phi_new*u_t_m
                v_t_m = phi_new*sin_delta*v_t_m
                #ADDITION
                v_t_m = v_t_m + b_new
           
                #print "u_t_m = ",u_t_m
                #print "v_t_m = ",v_t_m                

                R = self.A_1 + self.A_2*np.exp(-2*1j*np.pi*(u_t_m*self.l_0+v_t_m*self.m_0))

                if not approx:
                   d,Q = np.linalg.eigh(R)
                   D = np.diag(d)
                   Q_H = Q.conj().transpose()
                   abs_d=np.absolute(d)
                   index=abs_d.argmax()
                   if (d[index] >= 0):
                      g=Q[:,index]*np.sqrt(d[index])
                   else:
                      g=Q[:,index]*np.sqrt(np.absolute(d[index]))*1j
                   G = np.dot(np.diag(g),temp)
                   G = np.dot(G,np.diag(g.conj()))
                   if self.antenna == "all":
                      if u_dim2 <> 1:
                         V_R_pq[i,j] = R[b_list[0],b_list[1]]
                         V_G_pq[i,j] = G[b_list[0],b_list[1]]
                      else:
                         V_R_pq[i] = R[b_list[0],b_list[1]]
                         V_G_pq[i] = G[b_list[0],b_list[1]]
                   else:
                       for k in xrange(p_new.shape[0]):
                           for l in xrange(p_new.shape[1]):
                               if (p_new[k,l] == b_list[0]) and (q_new[k,l] == b_list[1]):
                                  if u_dim2 <> 1:
                                     V_R_pq[i,j] = R[k,l]
                                     V_G_pq[i,j] = G[k,l]
                                  else:
                                     V_R_pq[i] = R[k,l]
                                     V_G_pq[i] = G[k,l]
                else:
                    R1 = (R - self.A_1)/self.A_2
                    P = R1.shape[0]
                    if self.antenna == "all":
                       G = self.A_1 + ((0.5*self.A_2)/P)*(np.sum(R1[b_list[0],:])+np.sum(R1[:,b_list[1]]))
                       G = (G + ((0.5*self.A_2)/P)**2*R1[b_list[0],b_list[1]]*np.sum(R1))
                       if u_dim2 <> 1:
                          V_R_pq[i,j] = R[b_list[0],b_list[1]]
                          V_G_pq[i,j] = G
                       else:
                          V_R_pq[i] = R[b_list[0],b_list[1]]
                          V_G_pq[i] = G
                    else:
                        for k in xrange(p_new.shape[0]):
                            for l in xrange(p_new.shape[1]):
                                if (p_new[k,l] == b_list[0]) and (q_new[k,l] == b_list[1]):
                                   G = self.A1 + ((0.5*self.A2)/P)*(np.sum(R1[k,:])+np.sum(R1[:,l]))
                                   G = (G + ((0.5*self.A2)/P)**2*R1[k,l]*np.sum(R1))
                                   if u_dim2 <> 1:
                                      V_R_pq[i,j] = R[k,l]
                                      V_G_pq[i,j] = G
                                   else:
                                      V_R_pq[i] = R[k,l]
                                      V_G_pq[i] = G
        #print "V_G_pq = ",V_G_pq                     
        #if u_dim2 <> 1:
#   p  lt.contourf(uu,vv,V_G_pq)
        #else:
        #   plt.plot(V_G_pq)
        
        #plt.show()

        return u,v,V_G_pq,V_R_pq,phi,delta_b,theta,l_cor,m_cor
    
    def vis_function(self,type_w,avg_v,V_G_pq,V_G_qp,V_R_pq):
        if type_w == "R":
           vis = V_R_pq
        elif type_w == "RT":
           vis = V_R_pq**(-1)
        elif type_w == "R-1":
           vis = V_R_pq - 1
        elif type_w == "RT-1":
           vis = V_R_pq**(-1)-1
        elif type_w == "G":
           if avg_v:
              vis = (V_G_pq+V_G_qp)/2
           else:
              vis = V_G_pq
        elif type_w == "G-1":
           if avg_v:
              vis = (V_G_pq+V_G_qp)/2-1
           else:
              vis = V_G_pq-1
        elif type_w == "GT":
           if avg_v:
              vis = (V_G_pq**(-1)+V_G_qp**(-1))/2
           else:
              vis = V_G_pq**(-1)
        elif type_w == "GT-1":
           if avg_v:
              vis = (V_G_pq**(-1)+V_G_qp**(-1))/2-1
           else:
              vis = V_G_pq**(-1)-1
        elif type_w == "GTR-R":
           if avg_v:
              vis = ((V_G_pq**(-1)+V_G_qp**(-1))/2)*V_R_pq-V_R_pq
           else:
              vis = V_G_pq**(-1)*V_R_pq - V_R_pq
        elif type_w == "GTR":
           if avg_v:
              vis = ((V_G_pq**(-1)+V_G_qp**(-1))/2)*V_R_pq
           else:
              vis = V_G_pq**(-1)*V_R_pq
        elif type_w == "GTR-1":
           if avg_v:
              vis = ((V_G_pq**(-1)+V_G_qp**(-1))/2)*V_R_pq-1
           else:
              vis = V_G_pq**(-1)*V_R_pq-1
        return vis
    
    # sigma --- degrees, resolution --- arcsecond, image_s --- degrees, window --- degrees
    def sky_2D(self,resolution,image_s,s,sigma = None,type_w="G-1",avg_v=False,plot=False,mask=False,wave=None,dec=None,approx=False,window=0.2,subtract=False):
        if wave  == None:
           wave = self.wave           
        if dec == None:
           dec = self.dec
        ant_len = len(self.a_list)
        counter = 0
        baseline = [0,1]
        #u,v,V_G_qp,V_R_qp,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline,resolution=resolution,image_s=image_s,s=s,wave=wave,dec=dec,approx=approx)
        #N = l_cor.shape[0]
        #delta_u = u[1]-u[0]
        #delta_v = v[1]-v[0]
        #print "u = ",u
        #print "v = ",v

        for k in xrange(ant_len):
            for j in xrange(k+1,ant_len):
                baseline[0] = self.a_list[k]
                baseline[1] = self.a_list[j]
                counter = counter + 1                 
                print "counter = ",counter
                print "baseline = ",baseline
                if avg_v:
                   baseline_new = [0,0]
                   baseline_new[0] = baseline[1]
                   baseline_new[1] = baseline[0]
                   u,v,V_G_qp,V_R_qp,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline_new,resolution=resolution,image_s=image_s,s=s,wave=wave,dec=dec,approx=approx)
                else:
                   V_G_qp = 0

                u,v,V_G_pq,V_R_pq,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline,resolution=resolution,image_s=image_s,s=s,wave=wave,dec=dec,approx=approx)

                if (k==0) and (j==1):
                   vis = self.vis_function(type_w,avg_v,V_G_pq,V_G_qp,V_R_pq)
                else:
                   vis = vis + self.vis_function(type_w,avg_v,V_G_pq,V_G_qp,V_R_pq)
        
        vis = vis/counter           

        vis_old = np.copy(vis)

        l_old = np.copy(l_cor)
        m_old = np.copy(m_cor)
        
        N = l_cor.shape[0]

        delta_u = u[1]-u[0]
        delta_v = v[1]-v[0]

        if sigma <> None:

           uu,vv = np.meshgrid(u,v)

           sigma = (np.pi/180) * sigma

           g_kernal = (2*np.pi*sigma**2)*np.exp(-2*np.pi**2*sigma**2*(uu**2+vv**2))
       
           vis = vis*g_kernal

           vis = np.roll(vis,-1*(N-1)/2,axis = 0)
           vis = np.roll(vis,-1*(N-1)/2,axis = 1)

           image = np.fft.fft2(vis)*(delta_u*delta_v)
        else:
 
           image = np.fft.fft2(vis)/N**2

 
        #ll,mm = np.meshgrid(l_cor,m_cor)

        image = np.roll(image,1*(N-1)/2,axis = 0)
        image = np.roll(image,1*(N-1)/2,axis = 1)

        image = image[:,::-1]
        #image = image[::-1,:]

        #image = (image/1)*100

        if plot:

           l_cor = l_cor*(180/np.pi)
           m_cor = m_cor*(180/np.pi)

           fig = plt.figure() 
           cs = plt.imshow(image.real,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
           fig.colorbar(cs)
           self.plt_circle_grid(image_s)

           #print "amax = ",np.amax(image.real)
           #print "amax = ",np.amax(np.absolute(image))

           plt.xlim([-image_s,image_s])
           plt.ylim([-image_s,image_s])

           if mask:
              self.create_mask_all(plot_v=True,dec=dec)
           
           #self.create_mask(baseline,plot_v = True)

           plt.xlabel("$l$ [degrees]")
           plt.ylabel("$m$ [degrees]")
           plt.show()
        
           fig = plt.figure() 
           cs = plt.imshow(image.imag,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
           fig.colorbar(cs)
           self.plt_circle_grid(image_s)

           plt.xlim([-image_s,image_s])
           plt.ylim([-image_s,image_s])

           if mask:
              self.create_mask_all(plot_v=True)
           #self.create_mask(baseline,plot_v = True)

           plt.xlabel("$l$ [degrees]")
           plt.ylabel("$m$ [degrees]")
           plt.show()
        
        if subtract:
           point_sources = np.array([(0.02,0,0)])
           point_sources = np.append(point_sources,[(0.065,self.l_0,-1*self.m_0)],axis=0) 
           #print "point_sources = ",point_sources
           point_real,point_imag,point_abs = self.extract_flux(image,l_old,m_old,window,point_sources,False)
           #print "point_real = ",point_real.real
           vis = vis_old - self.create_vis(point_real.real,u,v)
           #point_real = np.copy(point_sources)
           #vis = self.create_vis(point_real.real,u,v)
           if sigma <> None:

              uu,vv = np.meshgrid(u,v)

              #sigma = (np.pi/180) * sigma

              g_kernal = (2*np.pi*sigma**2)*np.exp(-2*np.pi**2*sigma**2*(uu**2+vv**2))
       
              vis = vis*g_kernal

              vis = np.roll(vis,-1*(N-1)/2,axis = 0)
              vis = np.roll(vis,-1*(N-1)/2,axis = 1)

              image = np.fft.fft2(vis)*(delta_u*delta_v)
           else:
              image = np.fft.fft2(vis)/N**2
                      
           image = np.roll(image,1*(N-1)/2,axis = 0)
           image = np.roll(image,1*(N-1)/2,axis = 1)

           image = image[:,::-1]
           if plot:

              #l_cor = l_cor*(180/np.pi)
              #m_cor = m_cor*(180/np.pi)

              fig = plt.figure() 
              cs = plt.imshow(image.real,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
              fig.colorbar(cs)
              self.plt_circle_grid(image_s)

              #print "amax = ",np.amax(image.real)
              #print "amax = ",np.amax(np.absolute(image))

              plt.xlim([-image_s,image_s])
              plt.ylim([-image_s,image_s])

              if mask:
                 self.create_mask_all(plot_v=True,dec=dec)
           
              #self.create_mask(baseline,plot_v = True)

              plt.xlabel("$l$ [degrees]")
              plt.ylabel("$m$ [degrees]")
              plt.show()
        
              fig = plt.figure() 
              cs = plt.imshow(image.imag,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
              fig.colorbar(cs)
              self.plt_circle_grid(image_s)

              plt.xlim([-image_s,image_s])
              plt.ylim([-image_s,image_s])

              if mask:
                 self.create_mask_all(plot_v=True)
              #self.create_mask(baseline,plot_v = True)

              plt.xlabel("$l$ [degrees]")
              plt.ylabel("$m$ [degrees]")
              plt.show()

        return image,l_old,m_old

    # sigma --- degrees, resolution --- arcsecond, image_s --- degrees
    def sky_pq_2D(self,baseline,resolution,image_s,s,sigma = None,type_w="G-1",avg_v=False,plot=False,mask=False,wave=None,dec=None,label_v=False,save_fig=False,approx=False,difference=False,subtract=False,window=0.2):
        if wave == None:
           wave = self.wave
        if dec == None:
           dec = self.dec
        
        if avg_v:
           baseline_new = [0,0]
           baseline_new[0] = baseline[1]
           baseline_new[1] = baseline[0]
           u,v,V_G_qp,V_R_qp,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline_new,resolution=resolution,image_s=image_s,s=s,wave=wave,dec=dec,approx=approx)
        else:
           V_G_qp = 0

        u,v,V_G_pq,V_R_pq,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline,resolution=resolution,image_s=image_s,s=s,wave=wave,dec=dec,approx=approx)


        l_old = np.copy(l_cor)
        m_old = np.copy(m_cor)
        
        N = l_cor.shape[0]

        vis = self.vis_function(type_w,avg_v,V_G_pq,V_G_qp,V_R_pq)
        if difference and approx:

           if avg_v:
              baseline_new = [0,0]
              baseline_new[0] = baseline[1]
              baseline_new[1] = baseline[0]
              u,v,V_G_qp,V_R_qp,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline_new,resolution=resolution,image_s=image_s,s=s,wave=wave,dec=dec,approx=False)
           else:
              V_G_qp = 0

           u,v,V_G_pq,V_R_pq,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline,resolution=resolution,image_s=image_s,s=s,wave=wave,dec=dec,approx=False)
           vis2 = self.vis_function(type_w,avg_v,V_G_pq,V_G_qp,V_R_pq)
           vis = vis2 - vis

        #vis = V_G_pq-1

        delta_u = u[1]-u[0]
        delta_v = v[1]-v[0]

        vis_old = np.copy(vis)

        if sigma <> None:

           uu,vv = np.meshgrid(u,v)

           sigma = (np.pi/180) * sigma

           g_kernal = (2*np.pi*sigma**2)*np.exp(-2*np.pi**2*sigma**2*(uu**2+vv**2))
       
           vis = vis*g_kernal

           vis = np.roll(vis,-1*(N-1)/2,axis = 0)
           vis = np.roll(vis,-1*(N-1)/2,axis = 1)

           image = np.fft.fft2(vis)*(delta_u*delta_v)
        else:
 
           image = np.fft.fft2(vis)/N**2

 
        #ll,mm = np.meshgrid(l_cor,m_cor)

        image = np.roll(image,1*(N-1)/2,axis = 0)
        image = np.roll(image,1*(N-1)/2,axis = 1)

        image = image[:,::-1]
        #image = image[::-1,:]

        #image = (image/0.1)*100

        if plot:

           l_cor = l_cor*(180/np.pi)
           m_cor = m_cor*(180/np.pi)

           fig = plt.figure() 
           cs = plt.imshow(image.real,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
           #cs = plt.imshow(np.absolute(image),interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
           fig.colorbar(cs)
           self.plt_circle_grid(image_s)
           if label_v:
              self.plot_source_labels_pq(baseline,im=image_s,plot_x = False)

           #print "amax = ",np.amax(image.real)
           #print "amax = ",np.amax(np.absolute(image))

           plt.xlim([-image_s,image_s])
           plt.ylim([-image_s,image_s])

           if mask:
             p,p_labels = self.create_mask(baseline,plot_v = True,dec=dec)

           #for k in xrange(len(p)):
           #    plt.plot(p[k,1]*(180/np.pi),p[k,2]*(180/np.pi),"kv")

           plt.xlabel("$l$ [degrees]")
           plt.ylabel("$m$ [degrees]")
           plt.title("Baseline "+str(baseline[0])+str(baseline[1])+" --- Real")
           
           if save_fig:     
              plt.savefig("R_pq"+str(baseline[0])+str(baseline[1])+".pdf",bbox_inches="tight") 
              plt.clf()
           else:
              plt.show()
        
           fig = plt.figure() 
           cs = plt.imshow(image.imag,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
           fig.colorbar(cs)
           self.plt_circle_grid(image_s)
           if label_v:
              self.plot_source_labels_pq(baseline,im=image_s,plot_x = False)

           plt.xlim([-image_s,image_s])
           plt.ylim([-image_s,image_s])
           
           if mask:
              self.create_mask(baseline,plot_v = True,dec=dec)

           plt.xlabel("$l$ [degrees]")
           plt.title("Baseline "+str(baseline[0])+str(baseline[1])+" --- Imag")
           plt.ylabel("$m$ [degrees]")
           if save_fig:     
              plt.savefig("I_pq"+str(baseline[0])+str(baseline[1])+".pdf",bbox_inches="tight") 
              plt.clf()
           else:
              plt.show()

        image_old = np.copy(image)
         
        if subtract and mask:
           #point_sources = np.array([(0.02,0,0)])
           #point_sources = np.append(point_sources,[(0.065,self.l_0,-1*self.m_0)],axis=0) 
           point_sources = np.copy(p)
           point_real,point_imag,point_abs = self.extract_flux(image,l_old,m_old,window,point_sources,False)
           point_sources = np.copy(p)+0j
           
           point_real_old,l = self.extract_proto_mask(baseline,point_real,p_labels)
           point_imag_old,l = self.extract_proto_mask(baseline,point_imag,p_labels)
           point_abs_old,l = self.extract_proto_mask(baseline,point_abs,p_labels)

           print "point_real_old = ",point_real_old
           print "point_imag_old = ",point_imag_old
           #point_sources = complex(point_sources)
           point_sources[:,0] = point_real[:,0].real + point_imag[:,0].real*1j
           point_sources[:,2] = point_sources[:,2]*(-1)
           vis = vis_old - self.create_vis(point_sources,u,v)
           #point_real = np.copy(point_sources)
           #vis = self.create_vis(point_sources,u,v)
           if sigma <> None:

              uu,vv = np.meshgrid(u,v)

              #sigma = (np.pi/180) * sigma

              g_kernal = (2*np.pi*sigma**2)*np.exp(-2*np.pi**2*sigma**2*(uu**2+vv**2))
       
              vis = vis*g_kernal

              vis = np.roll(vis,-1*(N-1)/2,axis = 0)
              vis = np.roll(vis,-1*(N-1)/2,axis = 1)

              image = np.fft.fft2(vis)*(delta_u*delta_v)
           else:
              image = np.fft.fft2(vis)/N**2
                      
           image = np.roll(image,1*(N-1)/2,axis = 0)
           image = np.roll(image,1*(N-1)/2,axis = 1)

           image = image[:,::-1]
           if plot:

              #l_cor = l_cor*(180/np.pi)
              #m_cor = m_cor*(180/np.pi)

              fig = plt.figure() 
              cs = plt.imshow(image.real,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
              fig.colorbar(cs)
              self.plt_circle_grid(image_s)

              #print "amax = ",np.amax(image.real)
              #print "amax = ",np.amax(np.absolute(image))

              plt.xlim([-image_s,image_s])
              plt.ylim([-image_s,image_s])

              if mask:
                 self.create_mask(baseline,plot_v=True,dec=dec)
           
              #self.create_mask(baseline,plot_v = True)

              plt.xlabel("$l$ [degrees]")
              plt.ylabel("$m$ [degrees]")
              plt.show()
        
              fig = plt.figure() 
              cs = plt.imshow(image.imag,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
              fig.colorbar(cs)
              self.plt_circle_grid(image_s)

              plt.xlim([-image_s,image_s])
              plt.ylim([-image_s,image_s])

              if mask:
                 self.create_mask(baseline,plot_v=True)
              #self.create_mask(baseline,plot_v = True)

              plt.xlabel("$l$ [degrees]")
              plt.ylabel("$m$ [degrees]")
              plt.show()
              
              #point_real,point_imag,point_abs = self.extract_flux(image,l_old,m_old,window,point_sources,False)
              #point_real,l = self.extract_proto_mask(baseline,point_real,p_labels)
              #point_imag,l = self.extract_proto_mask(baseline,point_imag,p_labels)
              #point_abs,l = self.extract_proto_mask(baseline,point_abs,p_labels)
              #print "point_real_old = ", (point_real_old[:,0].real/(self.A_2))*100
              #print "point_real = ", (point_real[:,0].real/self.A_2)*100
              #print "point_abs_old = ", (point_abs_old[:,0].real/(self.A_2))*100
              #print "point_abs = ", (point_abs[:,0].real/(self.A_2))*100
              #plt.errorbar(range(len(point_real_old[:,0])),100*point_real_old[:,0].real/(21*self.A_2),yerr=200*point_real[:,0]/(21*self.A_2),fmt=":")
              #plt.show()
              #plt.errorbar(range(len(point_abs_old[:,0])),100*point_abs_old[:,0].real/(21*self.A_2),yerr=200*point_abs[:,0]/(21*self.A_2),fmt=":")
              #plt.show()

        return image_old,l_old,m_old,vis_old,u,v,image

    def plt_circle_grid(self,grid_m):
        plt.hold('on')
        rad = np.arange(1,1+grid_m,1)
        x = np.linspace(0,1,500)
        y = np.linspace(0,1,500)

        x_c = np.cos(2*np.pi*x)
        y_c = np.sin(2*np.pi*y)
        for k in range(len(rad)):
            plt.plot(rad[k]*x_c,rad[k]*y_c,"k",ls=":",lw=1)
    
    def create_mask_all(self,plot_v = False,dec=None):
        if dec == None:
           dec = self.dec
        sin_delta = np.absolute(np.sin(dec))
        
        point_sources = np.array([(1,0,0)])
        point_sources_labels = np.array([(0,0,0,0)])
        point_sources = np.append(point_sources,[(1,self.l_0,-1*self.m_0)],axis=0) 
        point_sources_labels = np.append(point_sources_labels,[(0,0,0,0)],axis=0)       
        point_sources = np.append(point_sources,[(1,2*self.l_0,-2*self.m_0)],axis=0) 
        point_sources_labels = np.append(point_sources_labels,[(0,0,0,0)],axis=0)       
        point_sources = np.append(point_sources,[(1,-1*self.l_0,1*self.m_0)],axis=0) 
        point_sources_labels = np.append(point_sources_labels,[(0,0,0,0)],axis=0)       
 
        #SELECTING ONLY SPECIFIC INTERFEROMETERS
        #####################################################
        d_list = self.calculate_delete_list()

        p = np.ones(self.phi_m.shape,dtype = int)
        p = np.cumsum(p,axis=0)-1
        q = p.transpose()

        if d_list == np.array([]):
            p_new = p
            q_new = q
            phi_new = self.phi_m
        else:
            p_new = np.delete(p,d_list,axis = 0)
            p_new = np.delete(p_new,d_list,axis = 1)
            q_new = np.delete(q,d_list,axis = 0)
            q_new = np.delete(q_new,d_list,axis = 1)

            phi_new = np.delete(self.phi_m,d_list,axis = 0)
            phi_new = np.delete(phi_new,d_list,axis = 1)

            b_new = np.delete(self.b_m,d_list,axis = 0)
            b_new = np.delete(b_new,d_list,axis = 1)

            theta_new = np.delete(self.theta_m,d_list,axis = 0)
            theta_new = np.delete(theta_new,d_list,axis = 1)
        #####################################################
        if plot_v == True:
           plt.plot(0,0,"rx")
           plt.plot(self.l_0*(180/np.pi),self.m_0*(180/np.pi),"rx")
           plt.plot(-1*self.l_0*(180/np.pi),-1*self.m_0*(180/np.pi),"rx")

        len_a = len(self.a_list)
        b_list = [0,0]

        first = True

        for h in xrange(len_a):
            for i in xrange(h+1,len_a):
                b_list[0] = self.a_list[h]
                b_list[1] = self.a_list[i]
                phi = self.phi_m[b_list[0],b_list[1]]
                delta_b = self.b_m[b_list[0],b_list[1]]
                theta = self.theta_m[b_list[0],b_list[1]]
                for j in xrange(theta_new.shape[0]):
                    for k in xrange(j+1,theta_new.shape[0]):
                        if not np.allclose(phi_new[j,k],phi):
                           l_cordinate = phi_new[j,k]/phi*(np.cos(theta_new[j,k]-theta)*self.l_0+sin_delta*np.sin(theta_new[j,k]-theta)*self.m_0)                
                           m_cordinate = phi_new[j,k]/phi*(np.cos(theta_new[j,k]-theta)*self.m_0-sin_delta**(-1)*np.sin(theta_new[j,k]-theta)*self.l_0)                
                           if plot_v == True:
                              plt.plot(l_cordinate*(180/np.pi),m_cordinate*(180/np.pi),"rx")  
                              plt.plot(-1*l_cordinate*(180/np.pi),-1*m_cordinate*(180/np.pi),"rx")  
                           point_sources = np.append(point_sources,[(1,l_cordinate,-1*m_cordinate)],axis=0) 
                           point_sources_labels = np.append(point_sources_labels,[(j,k,b_list[0],b_list[1])],axis=0)
                           point_sources = np.append(point_sources,[(1,-1*l_cordinate,1*m_cordinate)],axis=0) 
                           point_sources_labels = np.append(point_sources_labels,[(k,j,b_list[0],b_list[1])],axis=0)
        return point_sources,point_sources_labels

    def create_mask(self,baseline,plot_v = False,dec = None,plot_markers = False):
        if dec == None:
           dec = self.dec
        #sin_delta = np.absolute(np.sin(dec))
        sin_delta = np.sin(dec)
        point_sources = np.array([(1,0,0)])
        point_sources_labels = np.array([(0,0,0,0)])
        point_sources = np.append(point_sources,[(1,self.l_0,-1*self.m_0)],axis=0) 
        point_sources_labels = np.append(point_sources_labels,[(baseline[0],baseline[1],baseline[0],baseline[1])],axis=0)
        point_sources = np.append(point_sources,[(1,-1*self.l_0,1*self.m_0)],axis=0) 
        point_sources_labels = np.append(point_sources_labels,[(baseline[1],baseline[0],baseline[0],baseline[1])],axis=0)
        
        #SELECTING ONLY SPECIFIC INTERFEROMETERS
        #####################################################
        b_list = self.get_antenna(baseline,self.ant_names)
        #print "b_list = ",b_list
        d_list = self.calculate_delete_list()
        #print "d_list = ",d_list

        phi = self.phi_m[b_list[0],b_list[1]]
        delta_b = self.b_m[b_list[0],b_list[1]]
        theta = self.theta_m[b_list[0],b_list[1]]


        p = np.ones(self.phi_m.shape,dtype = int)
        p = np.cumsum(p,axis=0)-1
        q = p.transpose()

        if d_list == np.array([]):
            p_new = p
            q_new = q
            phi_new = self.phi_m
        else:
            p_new = np.delete(p,d_list,axis = 0)
            p_new = np.delete(p_new,d_list,axis = 1)
            q_new = np.delete(q,d_list,axis = 0)
            q_new = np.delete(q_new,d_list,axis = 1)

            phi_new = np.delete(self.phi_m,d_list,axis = 0)
            phi_new = np.delete(phi_new,d_list,axis = 1)

            b_new = np.delete(self.b_m,d_list,axis = 0)
            b_new = np.delete(b_new,d_list,axis = 1)

            theta_new = np.delete(self.theta_m,d_list,axis = 0)
            theta_new = np.delete(theta_new,d_list,axis = 1)
        #####################################################
        if plot_v == True:
           if plot_markers:
              mk_string = self.return_color_marker([0,0])
              plt.plot(0,0,self.return_color_marker([0,0]),label="(0,0)",mfc='none',ms=7)
              plt.hold('on')
              mk_string = self.return_color_marker(baseline)
              plt.plot(self.l_0*(180/np.pi),self.m_0*(180/np.pi),self.return_color_marker(baseline),label="("+str(baseline[0])+","+str(baseline[1])+")",mfc='none',mec=mk_string[0],ms=7)
              mk_string = self.return_color_marker([baseline[1],baseline[0]])
              plt.plot(-1*self.l_0*(180/np.pi),-1*self.m_0*(180/np.pi),self.return_color_marker([baseline[1],baseline[0]]),label="("+str(baseline[1])+","+str(baseline[0])+")",mfc='none',mec=mk_string[0],ms=7)
           else:             
              plt.plot(0,0,"rx")
              plt.plot(self.l_0*(180/np.pi),self.m_0*(180/np.pi),"rx")
              plt.plot(-1*self.l_0*(180/np.pi),-1*self.m_0*(180/np.pi),"gx")
        for j in xrange(theta_new.shape[0]):
            for k in xrange(j+1,theta_new.shape[0]):
                #print "Hallo:",j," ",k
                if not np.allclose(phi_new[j,k],phi):
                   #print "phi = ",phi_new[j,k]/phi
                   l_cordinate = (phi_new[j,k]*1.0)/(1.0*phi)*(np.cos(theta_new[j,k]-theta)*self.l_0+sin_delta*np.sin(theta_new[j,k]-theta)*self.m_0) 
                   #print "l_cordinate = ",l_cordinate*(180/np.pi)               
                   m_cordinate = (phi_new[j,k]*1.0)/(phi*1.0)*(np.cos(theta_new[j,k]-theta)*self.m_0-sin_delta**(-1)*np.sin(theta_new[j,k]-theta)*self.l_0)                
                   #print "m_cordinate = ",m_cordinate*(180/np.pi)               
                   if plot_v == True:
                      if plot_markers:
                         mk_string = self.return_color_marker([j,k])
                         plt.plot(l_cordinate*(180/np.pi),m_cordinate*(180/np.pi),self.return_color_marker([j,k]),label="("+str(j)+","+str(k)+")",mfc='none',mec=mk_string[0],ms=7)  
                         mk_string = self.return_color_marker([k,j])
                         plt.plot(-1*l_cordinate*(180/np.pi),-1*m_cordinate*(180/np.pi),self.return_color_marker([k,j]),label="("+str(k)+","+str(j)+")",mfc='none',mec=mk_string[0],ms=7) 
                         plt.legend(loc=8,ncol=9,numpoints=1,prop={"size":7},columnspacing=0.1) 
                      else:
                         plt.plot(l_cordinate*(180/np.pi),m_cordinate*(180/np.pi),"rx")  
                         plt.plot(-1*l_cordinate*(180/np.pi),-1*m_cordinate*(180/np.pi),"gx")  
                   point_sources = np.append(point_sources,[(1,l_cordinate,-1*m_cordinate)],axis=0) 
                   point_sources_labels = np.append(point_sources_labels,[(j,k,baseline[0],baseline[1])],axis=0)
                   point_sources = np.append(point_sources,[(1,-1*l_cordinate,1*m_cordinate)],axis=0) 
                   point_sources_labels = np.append(point_sources_labels,[(k,j,baseline[0],baseline[1])],axis=0)
        
        return point_sources,point_sources_labels

    #window is in degrees, l,m,point_sources in radians, point_sources[k,:] ---> kth point source 
    def extract_flux(self,image,l,m,window,point_sources,plot):
        window = window*(np.pi/180)
        point_sources_real = np.copy(point_sources)
        point_sources_imag = np.copy(point_sources)
        point_sources_abs = np.copy(point_sources)
        for k in range(len(point_sources)):
            l_0 = point_sources[k,1]
            m_0 = point_sources[k,2]*(-1)
            
            l_max = l_0 + window/2.0
            l_min = l_0 - window/2.0
            m_max = m_0 + window/2.0
            m_min = m_0 - window/2.0

            m_rev = m[::-1]

            #ll,mm = np.meshgrid(l,m)
            
            image_sub = image[:,(l<l_max)&(l>l_min)]
            #ll_sub = ll[:,(l<l_max)&(l>l_min)]
            #mm_sub = mm[:,(l<l_max)&(l>l_min)]
            
            if image_sub.size <> 0:  
               image_sub = image_sub[(m_rev<m_max)&(m_rev>m_min),:]
               #ll_sub = ll_sub[(m_rev<m_max)&(m_rev>m_min),:]
               #mm_sub = mm_sub[(m_rev<m_max)&(m_rev>m_min),:]
            
            #PLOTTING SUBSET IMAGE
            if plot:
               l_new = l[(l<l_max)&(l>l_min)]
               if l_new.size <> 0:
                  m_new = m[(m<m_max)&(m>m_min)]
                  if m_new.size <> 0:
                     l_cor = l_new*(180/np.pi)
                     m_cor = m_new*(180/np.pi)

                     # plt.contourf(ll_sub*(180/np.pi),mm_sub*(180/np.pi),image_sub.real)
                     # plt.show()

                     #fig = plt.figure() 
                     #cs = plt.imshow(mm*(180/np.pi),interpolation = "bicubic", cmap = "jet")
                     #fig.colorbar(cs)
                     #plt.show()
                     #fig = plt.figure() 
                     #cs = plt.imshow(ll*(180/np.pi),interpolation = "bicubic", cmap = "jet")
                     #fig.colorbar(cs)
                     #plt.show()
                   
                     fig = plt.figure() 
                     cs = plt.imshow(image_sub.real,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],l_cor[-1],m_cor[0],m_cor[-1]])
                     #plt.plot(l_0*(180/np.pi),m_0*(180/np.pi),"rx")  
                     fig.colorbar(cs)
                     plt.title("REAL")
                     plt.show()
                     fig = plt.figure() 
                     cs = plt.imshow(image_sub.imag,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],l_cor[-1],m_cor[0],m_cor[-1]])
                     fig.colorbar(cs)
                     plt.title("IMAG")
                     plt.show()
            #print "image_sub = ",image_sub
            if image_sub.size <> 0:
               max_v_r = np.amax(image_sub.real)
               max_v_i = np.amax(image_sub.imag)
               min_v_r = np.amin(image_sub.real)
               min_v_i = np.amin(image_sub.imag)
               max_v_a = np.amax(np.absolute(image_sub))
               if np.absolute(max_v_r) > np.absolute(min_v_r):
                  point_sources_real[k,0] = max_v_r
               else:
                  point_sources_real[k,0] = min_v_r
               if np.absolute(max_v_i) > np.absolute(min_v_i):
                  point_sources_imag[k,0] = max_v_i
               else:
                  point_sources_imag[k,0] = min_v_i
               point_sources_abs[k,0] = max_v_a
           
            else:
              point_sources_real[k,0] = 0
              point_sources_imag[k,0] = 0
              point_sources_abs[k,0] = 0
       
        return point_sources_real,point_sources_imag,point_sources_abs 

    def determine_flux_block_pq(self,baseline,f=1.445,dec=-74.66*(np.pi/180),l=1*(np.pi/180),m=0*(np.pi/180),A2=0.2,resolution=150,image_s=3,s=2,sigma_t=0.05,type_w_t="G-1",window=0.2):
             
        wave_v = 3e8/(1.0*f*1e9)
        #remember I am resetting the specifications of the sources
        self.l_0 = l
        self.m_0 = m
        self.A_2 = A2
        
        mask,point_source_labels = self.create_mask(baseline,dec=dec)
        image,l_v,m_v = self.sky_pq_2D(baseline,resolution,image_s,s,sigma = sigma_t, type_w=type_w_t, avg_v=False, plot=False, wave=wave_v,dec=dec)
        point_real,point_imag = self.extract_flux(image,l_v,m_v,window,mask,False)

        na = self.phi_m.shape[0]

        d = np.ones((na,))+0j
        d = d*point_real[0,0]
        mat = np.diag(d) 

        for k in xrange(1,len(point_real)):
            mat[point_source_labels[k,0],point_source_labels[k,1]] = point_real[k,0]+point_imag[k,0]*1j

        print "point_source_label = ",point_source_labels
        print "point_real = ",point_real
        print "mat = ",np.absolute(mat)
       
        plt.imshow(mat.real,interpolation="nearest")
        plt.show()

        plt.imshow(np.absolute(mat),interpolation="nearest")
        plt.show()
         

    def determine_flux_l_pq(self,baseline,l_min = 1,l_max = 3,number = 10,resolution=250,image_s=3,s=2,sigma_t=0.05,type_w_t="G-1",window=0.2):
        l_min = l_min*(np.pi/180.0)
        l_max = l_max*(np.pi/180.0)
        l_v_k = np.linspace(l_min,l_max,number)
        #wave_v = 3e8/(1.0*f_v)
        l_old = self.l_0
        mask,point_source_labels = self.create_mask(baseline)
        mask,point_source_labels = self.extract_proto_mask(baseline,mask,point_source_labels)

        result =  np.zeros((len(mask),len(l_v_k)))
        result2 =  np.zeros((len(mask),len(l_v_k)))

        for k in xrange(len(l_v_k)):
            print "k = ",k
            print "l_v = ",l_v_k
            self.l_0 = l_v_k[k]
            mask,point_source_labels = self.create_mask(baseline)
            mask,point_source_labels = self.extract_proto_mask(baseline,mask,point_source_labels)
            image,l_v,m_v,vis,u_c,v_c,i = self.sky_pq_2D(baseline,resolution,image_s,s,sigma = sigma_t, type_w=type_w_t, avg_v=False, plot=False)
            point_real,point_imag,point_abs = self.extract_flux(image,l_v,m_v,window,mask,False)
            result[:,k] = point_abs[:,0]
            result2[:,k] = point_real[:,0]
        print "v.PICKLENAME = ",II(v.PICKLENAME)
        v.PICKLENAME = "l_results_"+type_w_t+"_{0}_{1}".format(*baseline) #freq list, results, mask, point source labels
        file_name = II(v.PICKLEFILE)
        f = open(file_name, 'wb')
        pickle.dump(l_v_k*(180/np.pi),f)
        pickle.dump(result,f)
        pickle.dump(result2,f)
        pickle.dump(mask,f)
        pickle.dump(point_source_labels,f)
        f.close()
        self.l_0 = l_old
        return l_v_k,result,result2,mask,point_source_labels
    
    def determine_flux_dec_pq(self,baseline,dec_min = -80,dec_max = -30,number = 10,resolution=250,image_s=3,s=2,sigma_t=0.05,type_w_t="G-1",window=0.2):
        dec_min = dec_min*(np.pi/180.0)
        dec_max = dec_max*(np.pi/180.0)
        dec_v_k = np.linspace(dec_min,dec_max,number)
        #wave_v = 3e8/(1.0*f_v)
        mask,point_source_labels = self.create_mask(baseline)
        mask,point_source_labels = self.extract_proto_mask(baseline,mask,point_source_labels)

        result =  np.zeros((len(mask),len(dec_v_k)))
        result2 =  np.zeros((len(mask),len(dec_v_k)))

        for k in xrange(len(dec_v_k)):
            print "k = ",k
            print "dec_v = ",dec_v_k
            mask,point_source_labels = self.create_mask(baseline)
            mask,point_source_labels = self.extract_proto_mask(baseline,mask,point_source_labels)
            image,l_v,m_v,vis,u_c,v_c,i = self.sky_pq_2D(baseline,resolution,image_s,s,sigma = sigma_t, type_w=type_w_t, avg_v=False, plot=False,dec=dec_v_k[k])
            point_real,point_imag,point_abs = self.extract_flux(image,l_v,m_v,window,mask,False)
            result[:,k] = point_abs[:,0]
            result2[:,k] = point_real[:,0]
        print "v.PICKLENAME = ",II(v.PICKLENAME)
        v.PICKLENAME = "declination_"+type_w_t+"_{0}_{1}".format(*baseline) #freq list, results, mask, point source labels
        file_name = II(v.PICKLEFILE)
        f = open(file_name, 'wb')
        pickle.dump(dec_v_k*(180/np.pi),f)
        pickle.dump(result,f)
        pickle.dump(result2,f)
        pickle.dump(mask,f)
        pickle.dump(point_source_labels,f)
        f.close()
        return dec_v_k,result,result2,mask,point_source_labels
   
    def determine_flux_m_pq(self,baseline,m_min = 1,m_max = 3,number = 10,resolution=250,image_s=3,s=2,sigma_t=0.05,type_w_t="G-1",window=0.2):
        m_min = m_min*(np.pi/180.0)
        m_max = m_max*(np.pi/180.0)
        m_v_k = np.linspace(m_min,m_max,number)
        #wave_v = 3e8/(1.0*f_v)
        m_old = self.m_0
        mask,point_source_labels = self.create_mask(baseline)
        mask,point_source_labels = self.extract_proto_mask(baseline,mask,point_source_labels)

        result =  np.zeros((len(mask),len(m_v_k)))
        result2 =  np.zeros((len(mask),len(m_v_k)))

        for k in xrange(len(m_v_k)):
            print "k = ",k
            print "m_v = ",m_v_k
            self.m_0 = m_v_k[k]
            mask,point_source_labels = self.create_mask(baseline)
            mask,point_source_labels = self.extract_proto_mask(baseline,mask,point_source_labels)
            image,l_v,m_v,vis,u_c,v_c,i = self.sky_pq_2D(baseline,resolution,image_s,s,sigma = sigma_t, type_w=type_w_t, avg_v=False, plot=False)
            point_real,point_imag,point_abs = self.extract_flux(image,l_v,m_v,window,mask,False)
            result[:,k] = point_abs[:,0]
            result2[:,k] = point_real[:,0]
        print "v.PICKLENAME = ",II(v.PICKLENAME)
        v.PICKLENAME = "m_results_"+type_w_t+"_{0}_{1}".format(*baseline) #freq list, results, mask, point source labels
        file_name = II(v.PICKLEFILE)
        f = open(file_name, 'wb')
        pickle.dump(m_v_k*(180/np.pi),f)
        pickle.dump(result,f)
        pickle.dump(result2,f)
        pickle.dump(mask,f)
        pickle.dump(point_source_labels,f)
        f.close()
        self.m_0 = m_old
        return m_v_k,result,result2,mask,point_source_labels
    
    def determine_flux_wave_pq(self,baseline,f_min = 1.2,f_max = 1.95,number = 10,resolution=250,image_s=3,s=2,sigma_t=0.05,type_w_t="G-1",window=0.2,error_v=False,exper=100,p=13):
        f_min = f_min*1e9
        f_max = f_max*1e9
        f_v = np.linspace(f_min,f_max,number)
        wave_v = 3e8/(1.0*f_v)

        mask,point_source_labels = self.create_mask(baseline)
        result_error = []
        result_error2 = []
 
        if error_v:
           mask_new = np.copy(mask)
           mask_new[:,2] = mask_new[:,2]*-1
           result_error = np.zeros((exper,number))
           result_error2 = np.zeros((exper,number))
        mask,point_source_labels = self.extract_proto_mask(baseline,mask,point_source_labels)
         
        result =  np.zeros((len(mask),len(wave_v)))
        result2 =  np.zeros((len(mask),len(wave_v)))

        for k in xrange(len(wave_v)):
            print "k = ",k
            #print "f_v = ",f_v
            image,l_v,m_v,vis,u_c,v_c,i = self.sky_pq_2D(baseline,resolution,image_s,s,sigma = sigma_t, type_w=type_w_t, avg_v=False, plot=False,wave=wave_v[k])
            point_real,point_imag,point_abs = self.extract_flux(image,l_v,m_v,window,mask,False)
            result[:,k] = point_abs[:,0]
            result2[:,k] = point_real[:,0]
            if error_v:
               for j in xrange(exper):
                   #print "j = ",j
                   error_sources = self.create_error_sources(baseline_sources = mask_new,A2=self.A_2,p=p,number=1,fov=image_s)
                   error_sources_c = np.copy(error_sources)
                   error_sources_c[:,2] = error_sources[:,2]*-1
                   image_error,l_e,m_e = self.error_sky_pq_2D(baseline,l_v,m_v,u_c,v_c,vis,error_sources,window=window,sigma=sigma_t,plot=False)
                   point_real_e,point_imag_e,point_abs_e = self.extract_flux(image_error,l_e,m_e,window,error_sources_c,False)
                   #print "error_sources_c = ",error_sources_c[:,0]
                   #print "point_real_e = ",point_real_e.real[:,0]
                   #print "point_abs_e = ",point_abs_e.real[:,0]
                   result_error2[j,k] = np.absolute(point_real_e[0,0].real - error_sources_c[0,0].real)
                   result_error[j,k] = np.absolute(point_abs_e[0,0].real - np.absolute(error_sources_c[0,0]))
        #print "v.PICKLENAME = ",II(v.PICKLENAME)
        if error_v:
           v.PICKLENAME = "freq_results_"+type_w_t+"_error_"+str(int(p))+"_{0}_{1}".format(*baseline) #freq list, results, mask, point source labels
        else:
           v.PICKLENAME = "freq_results_"+type_w_t+"_{0}_{1}".format(*baseline) #freq list, results, mask, point source labels
        file_name = II(v.PICKLEFILE)
        f = open(file_name, 'wb')
        pickle.dump(f_v/1e9,f)
        pickle.dump(result,f)
        pickle.dump(result2,f)
        pickle.dump(result_error,f)
        pickle.dump(result_error2,f)
        pickle.dump(mask,f)
        pickle.dump(point_source_labels,f)
        f.close()
        return f_v,result,result2,mask,point_source_labels
    
    def determine_flux_A_pq(self,baseline,A_min = 0.001,A_max = 0.5,number = 10,resolution=250,image_s=3,s=2,sigma_t=0.05,type_w_t="G-1",window=0.2,error_v = False,exper = 100,p=13):
        A_v_k = np.linspace(A_min,A_max,number)
        #wave_v = 3e8/(1.0*f_v)
        A_old = self.A_2
        mask,point_source_labels = self.create_mask(baseline)
        result_error = []
        result_error2 = []
 
        if error_v:
           mask_new = np.copy(mask)
           mask_new[:,2] = mask_new[:,2]*-1
           result_error = np.zeros((exper,number))
           result_error2 = np.zeros((exper,number))
        
        mask,point_source_labels = self.extract_proto_mask(baseline,mask,point_source_labels)
       
          
        result =  np.zeros((len(mask),len(A_v_k)))
        result2 =  np.zeros((len(mask),len(A_v_k)))

        for k in xrange(len(A_v_k)):
            print "k = ",k
            #print "A_v = ",A_v_k
            self.A_2 = A_v_k[k]
            #mask,point_source_labels = self.create_mask(baseline)
            #mask,point_source_labels = self.extract_proto_mask(baseline,mask,point_source_labels)
            
            image,l_v,m_v,vis,u_c,v_c,i = self.sky_pq_2D(baseline,resolution,image_s,s,sigma = sigma_t, type_w=type_w_t, avg_v=False, plot=False)
            point_real,point_imag,point_abs = self.extract_flux(image,l_v,m_v,window,mask,False)
            result[:,k] = point_abs[:,0]
            result2[:,k] = point_real[:,0]
            if error_v:
               for j in xrange(exper):
                   #print "j = ",j
                   error_sources = self.create_error_sources(baseline_sources = mask_new,A2=self.A_2,p=p,number=1,fov=image_s)
                   error_sources_c = np.copy(error_sources)
                   error_sources_c[:,2] = error_sources[:,2]*-1
                   image_error,l_e,m_e = self.error_sky_pq_2D(baseline,l_v,m_v,u_c,v_c,vis,error_sources,window=window,sigma=sigma_t,plot=False)
                   point_real_e,point_imag_e,point_abs_e = self.extract_flux(image_error,l_e,m_e,window,error_sources_c,False)
                   #print "error_sources_c = ",error_sources_c[:,0]
                   #print "point_real_e = ",point_real_e.real[:,0]
                   #print "point_abs_e = ",point_abs_e.real[:,0]
                   result_error2[j,k] = np.absolute(point_real_e[0,0].real - error_sources_c[0,0].real)
                   result_error[j,k] = np.absolute(point_abs_e[0,0].real - np.absolute(error_sources_c[0,0]))
 
        #print "v.PICKLENAME = ",II(v.PICKLENAME)
        #print "result_error = ",result_error
        #print "result_error2 = ",result_error2

        if error_v:
           v.PICKLENAME = "flux_A2_"+type_w_t+"_error_"+str(int(p))+"_{0}_{1}".format(*baseline) #freq list, results, mask, point source labels
        else:
           v.PICKLENAME = "flux_A2_"+type_w_t+"_{0}_{1}".format(*baseline) #freq list, results, mask, point source labels
        file_name = II(v.PICKLEFILE)
        f = open(file_name, 'wb')
        pickle.dump(A_v_k,f)
        pickle.dump(result,f)
        pickle.dump(result2,f)
        pickle.dump(result_error,f)
        pickle.dump(result_error2,f)
        pickle.dump(mask,f)
        pickle.dump(point_source_labels,f)
        f.close()
        self.A_2 = A_old
        return A_v_k,result,result2,mask,point_source_labels
    
    def plot_pq(self,baseline,type_w="G-1",parameter="freq",flux_type=1,error_v=False,p=13):

        if parameter == "freq":
           if error_v:
              v.PICKLENAME = "freq_results_"+type_w+"_error_"+str(int(p))+"_{0}_{1}".format(*baseline) #freq list, results, mask, point source labels
           else:
              v.PICKLENAME = "freq_results_"+type_w+"_{0}_{1}".format(*baseline) #freq list, results, mask, point source labels
           x_label = r"Frequency [GHz]"
        elif parameter == "l":
           v.PICKLENAME = "l_results_"+type_w+"_{0}_{1}".format(*baseline) #freq list, results, mask, point source labels
           x_label = r"$l$ [$^{\circ}$]"
        elif parameter == "m":    
           v.PICKLENAME = "m_results_"+type_w+"_{0}_{1}".format(*baseline) #freq list, results, mask, point source labels
           x_label = r"$m$ [$^{\circ}$]"
        elif parameter == "dec":
           v.PICKLENAME = "declination_"+type_w+"_{0}_{1}".format(*baseline) #freq list, results, mask, point source labels
           x_label = r"$\delta_0$ [$^{\circ}$]"
        elif parameter == "A2":
           if error_v:
              v.PICKLENAME = "flux_A2_"+type_w+"_error_"+str(int(p))+"_{0}_{1}".format(*baseline) #freq list, results, mask, point source labels
           else:
              v.PICKLENAME = "flux_A2_"+type_w+"_{0}_{1}".format(*baseline) #freq list, results, mask, point source labels
           x_label = r"$A_2$ [Jy]"

        file_name = II(v.PICKLEFILE)
        f = open(file_name, 'rb')
        x = pickle.load(f)
        y1 = pickle.load(f)
        y2 = pickle.load(f)
        e1 = pickle.load(f)
        e2 = pickle.load(f)
        mask = pickle.load(f)
        p_labels = pickle.load(f)
        f.close()

        e1 = np.median(e1,axis=0)
        e2 = np.median(e2,axis=0)

        if flux_type == 1:
           y_label = r"Flux [% of $A_2$]"
           if parameter == "A2":
              y1 = (y1/x)*100
              y2 = (y2/x)*100
              if error_v:
                 e1 = (e1/x)*100*2
                 e2 = (e2/x)*100*2
           else:
              y1 = (y1/self.A_2)*100
              y2 = (y2/self.A_2)*100
              if error_v:
                 e1 = (e1/self.A_2)*100*2
                 e2 = (e2/self.A_2)*100*2
        elif flux_type == 2:
           y_label = r"Flux [% of $A_2$]"
           if parameter == "A2":
              y1 = (y1/(self.norm_f*x))*100
              y2 = (y2/(self.norm_f*x))*100
              if error_v:
                 e1 = (e1/(self.norm_f*x))*100*2
                 e2 = (e2/(self.norm_f*x))*100*2
           else:
              y1 = (y1/(self.norm_f*self.A_2))*100
              y2 = (y2/(self.norm_f*self.A_2))*100
              if error_v:
                 e1 = (e1/(self.norm_f*x))*100*2
                 e2 = (e2/(self.norm_f*x))*100*2
        else:
           y_label = r"Flux [Jy]"

        labels_1 = ['({0},{1})'.format(*s_label) for s_label in p_labels[:,0:2]]
        
        m_str = "-"
        for i in xrange(len(labels_1)):
            if i == 7:
               m_str = "--"
            if error_v:
               plt.errorbar(x,y1[i,:],e1,ls=m_str,label=labels_1[i])
            else:
               plt.plot(x,y1[i,:],m_str,label = labels_1[i])#lw = 2.0
            plt.hold('on')
        plt.legend(prop={'size':10})
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim([x[0],x[-1]])
        plt.show()
        labels_1 = ['({0},{1})'.format(*s_label) for s_label in p_labels[:,0:2]]
        
        m_str = "-"
        for i in xrange(len(labels_1)):
            if i == 7:
               m_str = "--"
            if error_v:
               plt.errorbar(x,y2[i,:],e2,ls=m_str,label=labels_1[i])
            else:
               plt.plot(x,y2[i,:],m_str,label = labels_1[i])#lw = 2.0
            plt.hold('on')
        plt.legend(prop={'size':10})
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim([x[0],x[-1]])
        plt.show()


    def plot_l_pq(self,baseline,s_index,k=5):
        v.PICKLENAME = "l_results"+"_{0}_{1}".format(*baseline) #freq list, results, mask, point source labels
        file_name = II(v.PICKLEFILE)
        f = open(file_name, 'rb')
        l_v_k = pickle.load(f)
        result = pickle.load(f)
        mask = pickle.load(f)
        p_labels = pickle.load(f)
        f.close()
        
        labels_1 = ['({0},{1})'.format(*s_label) for s_label in p_labels[:,0:2]]
        m_str = "-"
        for i in xrange(len(labels_1)):
            if i == 7:
               m_str = "--"
            plt.plot(l_v_k*(180.0/np.pi),result[i,:],m_str,label = labels_1[i])#lw = 2.0
            plt.hold('on')
        plt.legend(prop={'size':10})
        plt.xlabel("l [degrees]")
        plt.ylabel("Flux [Jy]")
        plt.xlim([l_v_k[0]*(180.0/np.pi),l_v_k[-1]*(180.0/np.pi)])
        plt.show()
        labels_1 = ['({0},{1},{2},{3})'.format(*s_label) for s_label in p_labels_new[0:k,:]]
        m_str = "-"
        for i in xrange(len(labels_1)):
            if i == 7:
               m_str = "--"
            #plt.plot(f_v/1e9,np.log(abs(result_new[i,:])),m_str,label = labels_1[i])#lw = 2.0
            plt.plot(l_v_k*(180.0/np.pi),result_new[i,:],m_str,label = labels_1[i])#lw = 2.0
            plt.hold('on')
        plt.legend(prop={'size':10})
        plt.xlabel("l [degrees]")
        plt.ylabel("Flux [Jy]")
        plt.xlim([l_v_k[0]*(180.0/np.pi),l_v_k[-1]*(180.0/np.pi)])
        plt.show()


    def plot_freq_pq(self,baseline,s_index,k=5):
        v.PICKLENAME = "freq_results"+"_{0}_{1}".format(*baseline) #freq list, results, mask, point source labels
        file_name = II(v.PICKLEFILE)
        f = open(file_name, 'rb')
        f_v = pickle.load(f)
        result = pickle.load(f)
        mask = pickle.load(f)
        p_labels = pickle.load(f)
        f.close()
        
        #Split ghosts into ssecondary suppression, primary suppression ghost and anti-ghost + remaining
        result_new = result[3:,:]
        result_new = result_new[s_index,:]
        
        p_labels_new = p_labels[3:,:]     
        p_labels_new = p_labels_new[s_index,:]  

        labels_1 = ['({0},{1},{2},{3})'.format(*s_label) for s_label in p_labels[0:3,:]]
        m_str = "-"
        for i in xrange(len(labels_1)):
            if i == 7:
               m_str = "--"
            plt.plot(f_v/1e9,result[i,:],m_str,label = labels_1[i])#lw = 2.0
            plt.hold('on')
        plt.legend(prop={'size':10})
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Flux [Jy]")
        plt.xlim([f_v[0]/1e9,f_v[-1]/1e9])
        plt.show()
        labels_1 = ['({0},{1},{2},{3})'.format(*s_label) for s_label in p_labels_new[0:k,:]]
        m_str = "-"
        for i in xrange(len(labels_1)):
            if i == 7:
               m_str = "--"
            #plt.plot(f_v/1e9,np.log(abs(result_new[i,:])),m_str,label = labels_1[i])#lw = 2.0
            plt.plot(f_v/1e9,result_new[i,:],m_str,label = labels_1[i])#lw = 2.0
            plt.hold('on')
        plt.legend(prop={'size':10})
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Flux [Jy]")
        plt.xlim([f_v[0]/1e9,f_v[-1]/1e9])
        plt.show()

    def determine_flux_pos_pq(self,baseline,i_max = 1,number = 10,resolution=250,image_s=3,s=2,sigma_t=0.05,type_w_t="G-1",window=0.2,window_c=0.2):
        l = np.linspace(np.absolute(i_max)*(-1),np.absolute(i_max),number)
        m = np.copy(l)
        ll,mm = np.meshgrid(l,m)
        mask = self.create_mask(baseline)

        l_c_min = window_c*(-1)
        l_c_max = window_c
        m_c_min = window_c*(-1)
        m_c_max = window_c

        result = np.zeros((len(mask),ll.shape[0],ll.shape[1]))

        for i in xrange(ll.shape[0]):
            for j in xrange(ll.shape[1]):
                print "i = ",i
                print "j = ",j

                critical_region = False

                #print "ll[i,j] =", ll[i,j]
                #print "mm[i,j] =", mm[i,j]

                #print "l_c_min = ",l_c_min
                #print "l_c_max = ",l_c_max
                #print "m_c_min = ",m_c_min
                #print "m_c_max = ",m_c_max
           

                if (ll[i,j] < l_c_max) and (ll[i,j] > l_c_min) and (mm[i,j] < m_c_max) and (mm[i,j] > m_c_min):
                   print "Critical reg"
                   critical_region = True                

                if not critical_region: 
                   self.l_0 = ll[i,j]*(np.pi/180)
                   self.m_0 = mm[i,j]*(np.pi/180)
                   image,l_v,m_v = self.sky_pq_2D(baseline,resolution,image_s,s,sigma = sigma_t, type_w=type_w_t, avg_v=False, plot=False)
                   mask = self.create_mask(baseline)
                   point_real,point_imag = self.extract_flux(image,l_v,m_v,window,mask,False)
                   result[:,i,j] = point_real[:,0]
                else:
                   result[:,i,j] = 0  

        return result,l,m
    
    def extract_deteuro_mask_all(self,baseline,mask,p_labels):
        p = baseline[0]
        q = baseline[1]
        mask1 = np.logical_not(np.logical_or(p_labels[:,0]==p,p_labels[:,1]==q))
        mask2 = np.logical_not(np.logical_and(p_labels[:,0]==0,p_labels[:,1]==0))
        deteuro_mask = np.logical_and(mask1,mask2)
        return mask[deteuro_mask],p_labels[deteuro_mask]

    def extract_proto_mask_all(self,baseline,mask,p_labels):
        p = baseline[0]
        q = baseline[1]
        mask1 = np.logical_or(p_labels[:,0]==p,p_labels[:,1]==q)
        mask2 = np.logical_and(p_labels[:,0]==0,p_labels[:,1]==0)
        proto_mask = np.logical_or(mask1,mask2)
        return mask[proto_mask],p_labels[proto_mask]
    
    def extract_proto_mask(self,baseline,mask,p_labels):
        p = baseline[0]
        q = baseline[1]
        mask1 = np.logical_or(p_labels[:,0]==p,p_labels[:,1]==q)
        mask2 = np.logical_not(np.logical_and(p_labels[:,0]==p,p_labels[:,1]==q))
        proto_mask = np.logical_and(mask1,mask2)
        return mask[proto_mask],p_labels[proto_mask]

    def create_error_sources(self,baseline_sources,A2=0.2,p=0.7,number=10,fov=3,m_dist=0.25):
        error_sources = np.zeros((number,3),dtype=np.complex64)
     
        flux_level = A2 * (p/100.0)
        #flux_level = 0.5
        #print "baseline_sources = ",len(baseline_sources)
        #print "baseline_sources = ",baseline_sources
        alpha =  np.random.uniform(low=0, high=1, size = number)
        error_sources[:,0] = np.sqrt(alpha)*flux_level + np.sqrt(1-alpha)*flux_level*1j
        for k in xrange(number):
            #print "k = ",k
            big = False
            while not big:
                  #print "k2 = ",k
                  l = np.random.uniform(low=-1*np.absolute(fov), high=np.absolute(fov), size = 1)*(np.pi/180)
                  m = np.random.uniform(low=-1*np.absolute(fov), high=np.absolute(fov), size = 1)*(np.pi/180)

                  d = np.sqrt((l-baseline_sources[:,1])**2 + (m-baseline_sources[:,2])**2)
                  #print "d = ",d*(180/np.pi)
                  d_min = np.amin(d)
                  #print "d_min = ",d_min*(180/np.pi)
                  #print "m_dist = ",m_dist
                  if d_min > (m_dist*(np.pi/180.0)):
                     big = True
                     error_sources[k,1] = l
                     error_sources[k,2] = m
                     point_temp = np.zeros((1,3))
                     #print "point_temp = ",point_temp
                     point_temp[0,0] = 0.0
                     point_temp[0,1] = l
                     point_temp[0,2] = m 
                     #print "baseline_sources = ",baseline_sources.shape
                     baseline_sources = np.append(baseline_sources,point_temp,axis=0)

        return error_sources
    
    def plt_circle_range(self,l_c,m_c,radius):
        plt.hold('on')
        x = np.linspace(0,1,500)
        y = np.linspace(0,1,500)

        x_c = radius*np.cos(2*np.pi*x)+l_c
        y_c = radius*np.sin(2*np.pi*y)+m_c
        plt.plot(x_c,y_c,"k",lw=1)

    def create_vis(self,point_sources,u,v):
        uu,vv = np.meshgrid(u,v)
        vis = np.zeros(uu.shape)
        for k in xrange(len(point_sources)):
            l0 = point_sources[k,1].real
            m0 = point_sources[k,2].real

            vis = vis + point_sources[k,0]*np.exp(-2*np.pi*1j*(uu*l0 + vv*m0))
        return vis
    
    def error_sky_pq_2D(self,baseline,l,m,u,v,vis,error_model,dec=None,window=0.2,sigma=0.05,plot=True,image_s=3.0):
        vis = vis + self.create_vis(error_model,u,v)

        if dec == None:
           dec = self.dec
        
        delta_u = u[1]-u[0]
        delta_v = v[1]-v[0]
        N = l.shape[0]

        uu,vv = np.meshgrid(u,v)
        sigma = (np.pi/180) * sigma

        g_kernal = (2*np.pi*sigma**2)*np.exp(-2*np.pi**2*sigma**2*(uu**2+vv**2))
       
        vis = vis*g_kernal

        vis = np.roll(vis,-1*(N-1)/2,axis = 0)
        vis = np.roll(vis,-1*(N-1)/2,axis = 1)
        
        image = np.fft.fft2(vis)*(delta_u*delta_v)
        
        image = np.roll(image,1*(N-1)/2,axis = 0)
        image = np.roll(image,1*(N-1)/2,axis = 1)

        image = image[:,::-1]

        #if norm:
        #   image = image/((self.n**2-self.n)/2.0)
        
        if plot:

           l_cor = l*(180/np.pi)
           m_cor = m*(180/np.pi)

           fig = plt.figure() 
           cs = plt.imshow(image.real,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
           #cs = plt.imshow(np.absolute(image),interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
           fig.colorbar(cs)
           self.plt_circle_grid(image_s)
           p = self.create_mask(baseline,plot_v = True,dec=dec)

           for k in xrange(len(error_model)):
               plt.plot(error_model[k,1]*(180/np.pi),error_model[k,2]*(180/np.pi),"kx",mfc=None)
               self.plt_circle_range(error_model[k,1]*(180/np.pi),error_model[k,2]*(180/np.pi),window)

           plt.xlim([-image_s,image_s])
           plt.ylim([-image_s,image_s])

           plt.xlabel("$l$ [degrees]")
           plt.ylabel("$m$ [degrees]")
           
           plt.show()
        
           fig = plt.figure() 
           cs = plt.imshow(image.imag,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
           fig.colorbar(cs)
           self.plt_circle_grid(image_s)
           p = self.create_mask(baseline,plot_v = True,dec=dec)

           for k in xrange(len(error_model)):
               plt.plot(error_model[k,1]*(180/np.pi),error_model[k,2]*(180/np.pi),"kx",mfc=None)
               self.plt_circle_range(error_model[k,1]*(180/np.pi),error_model[k,2]*(180/np.pi),window)
           plt.xlim([-image_s,image_s])
           plt.ylim([-image_s,image_s])

           plt.xlabel("$l$ [degrees]")
           plt.ylabel("$m$ [degrees]")
           plt.show()

        return image,l,m

    def determine_noise_statistics(self,image):
        flat_image = (np.absolute(image.flatten())/self.A_2)*100
        flat_image2 = (image.real.flatten()/self.A_2)*100
        #print "std1 = ",np.std(flat_image)
        #print "std2 = ",np.std(flat_image2)
        #gkde=stats.gaussian_kde(flat_image)
        #gkde2=stats.gaussian_kde(flat_image2)
        #gkde2.set_bandwidth(bw_method='silverman')
        plt.hist(flat_image2, bins=200, range = (-0.01,0.01), normed=1,histtype="bar")
        #plt.show()
        #ind = np.linspace(-0.005,0.005,3000)
        #kdepdf = gkde.evaluate(ind)
        #gkde2.covariance_factor = lambda : .0005
        #gkde2._compute_covariance()
        #kdepdf2 = gkde2.evaluate(ind)
        #plt.hold('on')
        #plt.plot(ind,kdepdf)
        #plt.hold('off')
        #plt.plot(ind,kdepdf2,'r')
        plt.show()


    def determine_flux_A2_proto_pq(self,baseline,A_2_min = 0.001, A_2_max = 0.5,number = 20,resolution=250,image_s=3,s=2,sigma_t=0.05,type_w_t="GT-1",window=0.2,l_0 = 1.0*(np.pi/180),m_0=0.0*(np.pi/180),dec=-74.66*(np.pi/180),f=1.445,norm=False):
        wave_v = 3e8/(1.0*f*1e9)
        #remember I am resetting the specifications of the sources
        self.l_0 = l_0
        self.m_0 = m_0
        
        A_2_v = np.linspace(A_2_min,A_2_max,number)
        mask,point_source_labels = self.create_mask(baseline,dec=dec)
        mask_old = np.copy(mask)
        mask_old[:,2] = mask_old[:,2]*(-1)
        #print "point_source_labels1 = ",point_source_labels
        mask,point_source_labels = self.extract_proto_mask(baseline,mask,point_source_labels)
        #print "point_source_labels2 = ",point_source_labels
        
        #four = np.array([(1,0,0),(1,self.l_0,-1*self.m_0),(1,-1*self.l_0,self.m_0),(1,2*self.l_0,-2*self.m_0)])
        error_real = np.zeros((len(A_2_v),))
        error_abs = np.zeros((len(A_2_v),))
        result_real = np.zeros((len(mask),len(A_2_v)))
        result_imag = np.zeros((len(mask),len(A_2_v)))
        result_abs = np.zeros((len(mask),len(A_2_v)))
        #image,l_v,m_v,vis_old,u,v = self.sky_pq_2D(baseline,resolution,image_s,s,sigma = sigma_t, type_w=type_w_t, avg_v=False, plot=True, mask=True, wave=wave_v,dec=dec)
       
        for i in xrange(len(A_2_v)):
            print "*****************************************"
            print "i = ",i
            print "A_2 = ",A_2_v[i] 
            self.A_2 = A_2_v[i]
            error_sources =  self.create_error_sources(baseline_sources = mask_old,A2 = self.A_2,p=13,number=45,fov=image_s)
            #print "error_sources = ",(np.absolute(error_sources[:,0])/self.A_2)*100
            image,l_v,m_v,vis_old,u,v = self.sky_pq_2D(baseline,resolution,image_s,s,sigma = sigma_t, type_w=type_w_t, avg_v=False, plot=False, wave=wave_v,dec=dec)
            image_error,l_e,m_e = self.error_sky_pq_2D(baseline,l_v,m_v,u,v,vis_old,error_sources,dec,window,sigma=0.05,plot=False)
            #for k in xrange(len(error_sources)):
            #    plt.plot(error_sources[k,1]*(180/np.pi),error_sources[k,2]*(180/np.pi),"kx")
            #    plt.hold("on")

            #for k in xrange(len(mask_old)):
            #    plt.plot(mask_old[k,1]*(180/np.pi),mask_old[k,2]*(180/np.pi),"rx")
            #    plt.hold("on")
            #plt.show()

            #for k in xrange(len(error_sources)):
            #    print "k = ",k
            #    print "d = ",np.amin(np.sqrt((error_sources[k,1]-mask_old[:,1])**2 + (error_sources[k,2]-mask_old[:,2])**2))*(180/np.pi)
            #print "vis_old = ",vis_old
            #print "u = ",u
            #print "v = ",v:827
            error_sources_t = np.copy(error_sources)
            error_sources_t[:,2] = error_sources_t[:,2]*(-1)
            if i == 0:
               point_real,point_imag,point_abs = self.extract_flux(image,l_v,m_v,window,mask,False)
               point_real_err,point_imag_err,point_abs_err = self.extract_flux(image_error,l_v,m_v,window,error_sources_t,False)
            else:
               point_real,point_imag,point_abs = self.extract_flux(image,l_v,m_v,window,mask,False)
               point_real_err,point_imag_err,point_abs_err = self.extract_flux(image_error,l_v,m_v,window,error_sources_t,False)
            #print "point_abs_err = ",(point_abs_err[:,0]/self.A_2)*100
            print "error_sources = ",error_sources[:,0].real
            print "point_real_err = ",point_real_err[:,0].real
            #print "point_imag_err = ",point_imag_err
            #print "*****************************************"
            error_real_t = np.absolute(point_real_err[:,0].real - error_sources[:,0].real)
            error_abs_t = np.absolute(point_abs_err[:,0].real - np.absolute(error_sources[:,0]))
            print "error_real = ",error_real
            print "mean = ",np.mean(error_real_t)
            print "std = ",np.std(error_real_t)
            print "median = ",np.median(error_real_t)
            print "median_error_abs = ",np.median(error_abs_t)
            print "*****************************************"
            error_real[i] = np.median(error_real_t)
            error_abs[i] = np.median(error_abs_t)
            result_real[:,i] = point_real[:,0]
            result_imag[:,i] = point_imag[:,0]
            #result_abs[:,i] = np.sqrt(point_real[:,0]**2+point_imag[:,0]**2)
            result_abs[:,i] = point_abs[:,0]
        #plt.plot(A_2_v,(result_real[0,:]/(21*A_2_v)*100))
        #plt.hold('on')
        #plt.plot(A_2_v,np.absolute(result_imag[0,:]/(21*A_2_v)*100))
        #plt.plot(A_2_v,(result_abs[0,:]/(21*A_2_v)*100))
        #plt.ylim([0,0.8])
        #plt.show() 
        if norm:
           error_real = error_real/self.norm_f
           error_abs = error_abs/self.norm_f
           result_real = result_real/self.norm_f
           result_imag = result_imag/self.norm_f
           result_abs = result_abs/self.norm_f
        error_real = np.ones(error_real.shape)*np.mean(error_real/A_2_v)
        error_abs = np.ones(error_abs.shape)*np.mean(error_abs/A_2_v)
        #print "s_label = ",s_label
        #print "s_label[:,2:] = ",s_label[:,2:]
        labels_1 = ['({0},{1})'.format(*s_label) for s_label in point_source_labels[:,0:2]]
        #labels_1 = ['({1},{2})'.format(*s_label[2:]) for s_label in point_source_labels]
        m_str = "-"
        for i in xrange(len(labels_1)):
            if i == 7:
               m_str = "--"
            print "len(result_real[i,:]) = ",len(result_real[i,:])
            print "len(error_real) = ",len(error_real)
            plt.errorbar(A_2_v,(result_real[i,:])/(A_2_v)*100,yerr = 200*(error_real),ls=m_str,lw=1.5,label=labels_1[i], capsize=8, mew=2)#lw = 2.0
            plt.hold('on')
        plt.legend(prop={'size':10})
        plt.xlabel("Flux [Jy]")
        plt.ylabel("% Flux [Jy]")
        #plt.xlim([f_v[0]/1e9,f_v[-1]/1e9])
        plt.show()
        
        #labels_1 = ['({0},{1},{2},{3})'.format(*s_label) for s_label in point_source_labels]
        #m_str = "-"
        #for i in xrange(len(labels_1)):
        #    if i == 7:
        #       m_str = "--"
        #    plt.plot(A_2_v,(result_imag[i,:])/(21*A_2_v)*100,m_str,label = labels_1[i])#lw = 2.0
        #    plt.hold('on')
        #plt.legend(prop={'size':10})
        #plt.xlabel("Flux [Jy]")
        #plt.ylabel("% Flux [Jy]")
        #plt.xlim([f_v[0]/1e9,f_v[-1]/1e9])
        #plt.show()
        
        m_str = "-"
        for i in xrange(len(labels_1)):
            if i == 7:
               m_str = "--"
            #plt.plot(A_2_v,(result_abs[i,:])/(21*A_2_v)*100,m_str,label = labels_1[i])#lw = 2.0
            plt.errorbar(A_2_v,(result_abs[i,:])/(A_2_v)*100,yerr = 200*(error_abs),ls=m_str,lw=1.5,label=labels_1[i], capsize=8, mew=2)#lw = 2.0
            plt.hold('on')
        plt.legend(prop={'size':10})
        plt.xlabel("Flux [Jy]")
        plt.ylabel("% Flux [Jy]")
        #plt.xlim([f_v[0]/1e9,f_v[-1]/1e9])
        plt.show()        
        return result_real,result_abs,A_2_v
    
    def determine_flux_A2(self,A_2_min = 0.001, A_2_max = 0.5,number = 5,resolution=250,image_s=3,s=2,sigma_t=0.05,type_w_t="GT-1",window=0.2):
        A_2_v = np.linspace(A_2_min,A_2_max,number)
        four = np.array([(1,0,0),(1,self.l_0,-1*self.m_0),(1,-1*self.l_0,self.m_0),(1,2*self.l_0,-2*self.m_0)])

        result = np.zeros((len(four),len(A_2_v)))

        for i in xrange(len(A_2_v)):
            print "i = ",i
            print "A_2 = ",A_2_v[i] 
            self.A_2 = A_2_v[i]
            image,l_v,m_v = self.sky_2D(resolution,image_s,s,sigma = sigma_t, type_w=type_w_t, avg_v=False, plot=False)
            point_real,point_imag,point_abs = self.extract_flux(image,l_v,m_v,window,four,False)
            result[:,i] = point_real[:,0]
        v.PICKLENAME = "flux_A2" #freq list, results, mask, point source labels
        file_name = II(v.PICKLEFILE)
        f = open(file_name, 'wb')
        pickle.dump(A_2_v,f)
        pickle.dump(result,f)
        pickle.dump(mask,f)
        pickle.dump(point_source_labels,f)
        f.close()
        return result,A_2_v
    
    def dec_graph(self,baseline,dec_min=-90*(np.pi/180),dec_max=-10*(np.pi/180),npoints=100):
        p_temp,p_l = self.create_mask(baseline,plot_v = False,dec = dec_min)
        p_n = p_temp.shape[0]
        dec_v = np.linspace(dec_min,dec_max,npoints)

        mask_dec = np.zeros((len(dec_v),p_n,3))

        for k in xrange(len(dec_v)):
            mask_dec[k,:,:],p_l = self.create_mask(baseline,plot_v=False,dec=dec_v[k])
        
        for p in xrange(p_n):
            plt.plot(mask_dec[:,p,1]*(180/np.pi),mask_dec[:,p,2]*(-180/np.pi),"b")
            plt.hold('on')
        plt.plot(mask_dec[0,:,1]*(180/np.pi),mask_dec[0,:,2]*(-180/np.pi),ls="",marker="s",c="r")
        plt.plot(mask_dec[-1,:,1]*(180/np.pi),mask_dec[-1,:,2]*(-180/np.pi),ls="",marker="o",c="r")
        plt.xlim([-3,3])
        plt.ylim([-3,3])
        self.plt_circle_grid(3)
        plt.axis("image")
        plt.xlabel("$l$ [degrees]")
        plt.ylabel("$m$ [degrees]")
        plt.show()
    
    def plot_source_labels_pq(self,baseline,im=3,plot_x = False,r=60.0,f_s=6,r_rand = 1):
        point_sources,point_source_labels = self.create_mask(baseline)
        plt.hold('on')
        labels = ['({0},{1})'.format(*s_label) for s_label in point_source_labels[:,0:2]]
        #labels = ['({0},{1},{2},{3})'.format(*s_label) for s_label in point_source_labels]
        if plot_x:
           plt.plot(point_sources[:,1]*(180/np.pi),point_sources[:,2]*(-180/np.pi),ls="",marker="x",c="r")
        counter = 0
        for label, x, y in zip(labels, point_sources[:, 1]*(180/np.pi), point_sources[:, 2]*(-180/np.pi)):
                        
            delta_r = r + np.random.uniform(0,r_rand)
            delta_x = x
            delta_y = y
            if delta_x == 0:
               delta_x_new = 0
               delta_y_new = delta_r
            else:
               m = delta_y/delta_x
               delta_x_new = delta_r/np.sqrt(1 + m**2)
               if (delta_x < 0):
                  delta_x_new = delta_x_new*(-1)
               delta_y_new = delta_x_new*m
               #if (delta_x_new < 0) and ()
               print "************************"
               print "label = ",label
               print "delta_x =",delta_x_new
               print "delta_y =",delta_y_new
               print "m = ",m
               print "size = ",np.sqrt(delta_x_new**2+delta_y_new**2)
               print "************************"

            #plt.annotate(label, xy = (x, y), xytext = (delta_x_new, delta_y_new), textcoords = 'offset points', fontsize = 8, ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.5', fc = 'white', alpha = 0.05), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
            plt.annotate(label, xy = (x, y), xytext = (delta_x_new, delta_y_new), textcoords = 'offset points', fontsize = f_s, ha='center', va = 'center',bbox = dict(boxstyle = 'round,pad=0.5', fc = 'white', alpha = 0.05), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
       
    def get_sorted_index_pq(self,baseline,resolution=250,image_s=3,s=2,sigma_t=0.05,type_w_t="G-1",window=0.2):
        image,l_v,m_v,vis,u,v,i = self.sky_pq_2D(baseline,resolution,image_s,s,sigma = sigma_t, type_w=type_w_t, avg_v=False, plot=False)
        mask,point_source_labels = self.create_mask(baseline)
        point_real,point_imag,point_abs = self.extract_flux(image,l_v,m_v,window,mask,False)
        point_real = point_real[3:,:]
        s_index = np.argsort(np.absolute(point_real[:,0]))
        s_index = s_index[::-1]
        return s_index
   
    def determine_ghosts_theta(self,resolution=150,image_s=3,s=2,sigma_t=0.05,type_w_t="GT-1",window=0.2):
        baseline_ind = np.array([])
        len_ant = len(self.a_list)
        baseline = [0,0]
        x = np.zeros((self.norm_f,))
        y = np.zeros((self.norm_f,))

        counter = 0
        for i in xrange(len_ant):
            for j in xrange(i+1,len_ant):
                
                baseline[0] = i
                baseline[1] = j
                print "baseline = ",baseline
                print "A2 = ",self.A_2
                image,l_v,m_v,vis_v,u,v,i_old = self.sky_pq_2D(baseline,resolution,image_s,s,sigma = sigma_t, type_w=type_w_t, avg_v=False, plot=False)
                mask,p_labels = self.create_mask(baseline)
                mask,p_labels = self.extract_proto_mask(baseline,mask,p_labels)

                mask_temp = np.copy(mask)
                mask_temp[:,1] = mask_temp[:,1]*(180/np.pi)
                mask_temp[:,2] = mask_temp[:,2]*(180/np.pi)
                print "mask = ",mask_temp
                point_real,point_imag,point_abs = self.extract_flux(image,l_v,m_v,window,mask,False)
                point_abs[:,1] = point_abs[:,1]*(180/np.pi)
                point_abs[:,2] = point_abs[:,2]*(180/np.pi)
                point_abs[:,0] = (point_abs[:,0]/(21*self.A_2))*100
                print "point_abs = ",point_abs
                y[counter] = np.amax(point_abs[:,0]) 
                x[counter] = self.phi_m[i,j]
                counter = counter + 1
                print "counter = ", counter
        plt.plot(x,y,"rs")
        plt.show() 
        return x,y 
 
    def determine_top_k_ghosts(self,k=10,resolution=150,image_s=3,s=2,sigma_t=0.05,type_w_t="GT-1",window=0.2):
        baseline_ind = np.array([])
        len_ant = len(self.a_list)
        baseline = [0,0]

        for i in xrange(len_ant):
            for j in xrange(i+1,len_ant):
                
                baseline[0] = i
                baseline[1] = j
                print "baseline = ",baseline
                image,l_v,m_v = self.sky_pq_2D(baseline,resolution,image_s,s,sigma = sigma_t, type_w=type_w_t, avg_v=False, plot=False)
                mask = self.create_mask(baseline)
                mask = mask[3:,:]
                mask_temp = np.copy(mask)
                mask_temp[:,1] = mask_temp[:,1]*(180/np.pi)
                mask_temp[:,2] = mask_temp[:,2]*(180/np.pi)
                print "mask = ",mask_temp
                point_real,point_imag = self.extract_flux(image,l_v,m_v,window,mask,False)
                mask_temp = np.copy(point_real)
                mask_temp[:,1] = mask_temp[:,1]*(180/np.pi)
                mask_temp[:,2] = mask_temp[:,2]*(180/np.pi)
                print "point_real = ",mask_temp 
                baseline_t = np.zeros((len(mask),2))
                baseline_t[:,0] = i
                baseline_t[:,1] = j
                
                if (i == 0) and (j == 1):
                   baseline_result = np.copy(baseline_t)
                   point_result = np.copy(point_real)
                else:
                   baseline_result = np.concatenate((baseline_result,baseline_t))
                   point_result = np.concatenate((point_result,point_real))
                print "baseline_result = ",baseline_result
                #print "point_result = ",point_result 
        s_index = np.argsort(np.absolute(point_result[:,0]))
        #s_index = s_index[::-1]
        print "s_index = ",s_index
        print "s_index.shape = ",s_index.shape
        point_result = point_result[s_index,:]
        point_result = point_result[::-1,:]
        baseline_result = baseline_result[s_index,:]
        baseline_result = baseline_result[::-1,:]
        print "point_result_fin = ",point_result
        return point_result[0:k,:],baseline_result[0:k,:] 
    
    def plot_ghost_images(self,im_s):
        num_ant = len(self.a_list)
        counter = 0
        for j in xrange(num_ant):
            for k in xrange(j+1,num_ant):
                counter = counter + 1
                print "counter = ",counter
                baseline = [j,k]
                baseline_r = [k,j]
                image,l_v,m_v = self.sky_pq_2D(baseline,150,im_s,2,sigma = 0.05,type_w="G-1",avg_v=False,plot=True,mask=True,label_v = False,save_fig=True)
                image,l_v,m_v = self.sky_pq_2D(baseline_r,150,im_s,2,sigma = 0.05,type_w="G-1",avg_v=False,plot=True,mask=True,label_v = False,save_fig=True)

    #def create_mask_all(self,plot_v = False,dec=None):
    def extract_alpha_mask(self,mask,p_labels):
        alpha_mask = np.logical_and(p_labels[:,0]==0,p_labels[:,1]==0)
        return mask[alpha_mask],p_labels[alpha_mask]
    
    def extract_beta_mask(self,mask,p_labels):
        mask1 = np.logical_or(p_labels[:,0]==p_labels[:,2],p_labels[:,1]==p_labels[:,3])
        mask2 = np.logical_not(np.logical_and(p_labels[:,0]==0,p_labels[:,1]==0))
        beta_mask = np.logical_and(mask1,mask2)
        return mask[beta_mask],p_labels[beta_mask]
    
    def extract_gamma_mask(self,mask,p_labels):
        mask1 = np.logical_not(np.logical_or(p_labels[:,0]==p_labels[:,2],p_labels[:,1]==p_labels[:,3]))
        mask2 = np.logical_not(np.logical_and(p_labels[:,0]==0,p_labels[:,1]==0))
        gamma_mask = np.logical_and(mask1,mask2)
        return mask[gamma_mask],p_labels[gamma_mask]
    
    def plot_ghost_mask_all_paper(self,dec=-74.66*(np.pi/180),im_s=3):
        mask,p_labels = self.create_mask_all(dec=dec)
        alpha,a_labels = self.extract_alpha_mask(mask,p_labels)
        beta,b_labels = self.extract_beta_mask(mask,p_labels)
        gamma,g_labels = self.extract_gamma_mask(mask,p_labels)
        plt.plot(alpha[:,1]*(180/np.pi),-1*alpha[:,2]*(180/np.pi),"s",label=r"$\alpha$-category",mfc='none',mec="k",ms=7) 
        plt.plot(beta[:,1]*(180/np.pi),-1*beta[:,2]*(180/np.pi),"o",label=r"$\beta$-category",mfc='none',mec="b",ms=7) 
        plt.plot(gamma[:,1]*(180/np.pi),-1*gamma[:,2]*(180/np.pi),"x",label=r"$\gamma$-category",mfc='none',mec="r",ms=5) 
        self.plt_circle_grid(abs(im_s))
        plt.legend(loc=8,ncol=3,numpoints=1,prop={"size":12},columnspacing=0.1) 

        #plt.title("Baseline: "+str(baseline[0])+str(baseline[1]))
        plt.xlabel("$l$ [degrees]")
        plt.ylabel("$m$ [degrees]")
        plt.axis("image")
        plt.xlim([-1*abs(im_s),abs(im_s)])
        plt.ylim([-1*abs(im_s),abs(im_s)])
        plt.show()


    def plot_ghost_mask_paper(self,baseline,dec=-74.66*(np.pi/180),im_s=3):
        mask,p_labels = self.create_mask(baseline,plot_v=True,dec=dec,plot_markers=True)
        self.plt_circle_grid(abs(im_s))
        plt.title("Baseline: "+str(baseline[0])+str(baseline[1]))
        plt.xlabel("$l$ [degrees]")
        plt.ylabel("$m$ [degrees]")
        plt.axis("image")
        plt.xlim([-1*abs(im_s),abs(im_s)])
        plt.ylim([-1*abs(im_s),abs(im_s)])
        plt.show()        
        
        proto_mask, proto_labels = self.extract_proto_mask_all(baseline,mask,p_labels)
        deteuro_mask, deteuro_labels = self.extract_deteuro_mask_all(baseline,mask,p_labels)
        plt.plot(proto_mask[:,1]*(180/np.pi),-1*proto_mask[:,2]*(180/np.pi),"o",label="Proto-ghosts",mfc='none',mec="b",ms=7) 
        plt.plot(deteuro_mask[:,1]*(180/np.pi),-1*deteuro_mask[:,2]*(180/np.pi),"s",label="Deteuro-ghosts",mfc='none',mec="g",ms=7) 
        self.plt_circle_grid(abs(im_s))
        plt.legend(loc=8,ncol=2,numpoints=1,prop={"size":12}) 

        plt.title("Baseline: "+str(baseline[0])+str(baseline[1]))
        plt.xlabel("$l$ [degrees]")
        plt.ylabel("$m$ [degrees]")
        plt.axis("image")
        plt.xlim([-1*abs(im_s),abs(im_s)])
        plt.ylim([-1*abs(im_s),abs(im_s)])
        plt.show()

    def plot_ghost_masks(self,im_s):
        num_ant = len(self.a_list)
        counter = 0
        for j in xrange(num_ant):
            for k in xrange(j+1,num_ant):
                counter = counter + 1
                print "counter = ",counter
                baseline = [j,k]
                plt.title("Baseline: "+str(j)+str(k))
                plt.xlabel("$l$ [degrees]")
                plt.ylabel("$m$ [degrees]")
                self.create_mask(baseline,plot_v = True, dec = None, plot_markers = True)
                self.plt_circle_grid(abs(im_s))
                plt.axis("image")
                plt.xlim([-1*abs(im_s),abs(im_s)])
                plt.ylim([-1*abs(im_s),abs(im_s)])
                #plt.show()
                plt.savefig("T_gp_"+str(j)+str(k)+".pdf",bbox_inches="tight") 
                plt.clf()
    
    def return_color_marker(self,baseline):
        if (baseline[0] == 0) and (baseline[1] == 0):
           return "ko"
        if (baseline[0] > baseline[1]):
           b1 = baseline[1] #color
           b2 = baseline[0] + 6 #marker
        else:
           b1 = baseline[0] #color
           b2 = baseline[1] #marker
        c_s = ["b","g","r","c","m","y"]
        m_s = ["_","v","^","<",">","o","s","p","*","+","x","D"]

        return c_s[b1]+m_s[b2-1]        
        
def runall ():
    
        p_wrapper = Pyxis_helper()

        ms_object = Mset("KAT7_1445_1x16_12h.ms")

        ms_object.extract()

        #CREATE LINEAR TRANSFORMATION MATRICES
        #*************************************
        #e = Ellipse(ms_object)
        #e.calculate_baseline_trans()
        #e.create_phi_c()
        #e.test_angle(0,1)
        #e.test_angle(2,3)
        #e.test_angle(4,5)
        #*************************************
        
        point_sources = np.array([(1,0,0),(0.2,(1*np.pi)/180,(0*np.pi)/180)])
        #point_sources = np.array([(1j,1*(np.pi/180),1*(np.pi/180)),(0,0,0)])
        #s = Sky_model(ms_object,point_sources)
        #point_sources = np.array([(0.2,0,0),(-0.01,(-1*np.pi/180),(0*np.pi/180)),(0.012,(1*np.pi/180),(-1*np.pi/180)),(-0.1,(-1*np.pi/180),(-1*np.pi/180))])
        #s.meqskymodel(point_sources,antenna=[4,5])

        #s.visibility("CORRECTED_DATA")
        
        #p_wrapper.image_settings(filter=None)
        #options = p_wrapper.image_advanced_settings(antenna=[4,5])
        #imager.make_image(column="CORRECTED_DATA",dirty=options)
        #p_wrapper.flip_fits()
        
        #c_eig = Calibration(s.ms,"all","eig_cal",s.total_flux) #0A sum of weight must be positive exception..., AB ---> value error
        #c_eig.read_R("CORRECTED_DATA")
        #c_eig.cal_G_eig()
        #c_eig.write_to_MS("CORRECTED_DATA","GTR")

        #p_wrapper.pybdsm_search_pq([4,5])
          

        #u_temp = c_eig.u_m[0,1,:]
        #v_temp = c_eig.v_m[0,1,:]

        #V_G_pq_s = c_eig.G[0,1,:]
        #V_R_pq_s = c_eig.R[0,1,:]
        
        #plt.plot(u_temp,v_temp,'k')
        #plt.show()
        #longest 1-5
        #shortest 2-3
        t = T_ghost(point_sources,"all","KAT7_1445_1x16_12h.ms")
        #def sky_2D(self,resolution,image_s,s,sigma = None,type_w="G-1",avg_v=False,plot=False,mask=False,wave=None,dec=None,approx=False,window=0.2,subtract=False):
        #t.determine_ghosts_theta(resolution=150,image_s=3,s=2,sigma_t=0.025,type_w_t="GT-1",window=0.1)
        #image,l_v,m_v = t.sky_2D(150,3,2,sigma=0.05,type_w="G-1",avg_v=True,plot=True,dec=-74.66*(np.pi/180),mask=False,subtract=False,window=0.2)
        #t.determine_noise_statistics(image)
        #image,l_v,m_v,vis_old,u,v,i_old = t.sky_pq_2D([1,4],130,3,2,sigma = 0.05,type_w="GT-1",avg_v=False,plot=True,dec=-74.66*(np.pi/180),mask=True,label_v = False)
        #t.determine_flux_block_pq(self,baseline,f=1.95,dec=-60*(np.pi/180),l=1*(np.pi/180),m=0*(np.pi/180),A2=0.2,resolution=250,image_s=3,s=2,sigma_t=0.05,type_w_t="G-1",window=0.2):
        #t.determine_flux_block_pq([1,2],f=1.445,dec=-70*(np.pi/180),l=1*(np.pi/180),m=0*(np.pi/180),A2=0.2,resolution=150,image_s=3,s=2,sigma_t=0.05,type_w_t="GTR-R",window=0.2)
        #t.determine_flux_block_pq([1,2],f=1.3,dec=-50*(np.pi/180),l=-1*(np.pi/180),m=1*(np.pi/180),A2=0.2,resolution=150,image_s=3,s=2,sigma_t=0.05,type_w_t="GTR-R",window=0.2)
        #t.determine_flux_A2_proto_pq([4,5],A_2_min = 0.005, A_2_max = 0.5,number = 10,resolution=150,image_s=3,s=2,sigma_t=0.05,type_w_t="G-1",window=0.1,l_0 = 1.0*(np.pi/180),m_0=0.0*(np.pi/180),dec=-74.66*(np.pi/180),f=1.445)
        #t.determine_flux_A2_proto_pq([4,5],A_2_min = 0.005, A_2_max = 0.5,number = 10,resolution=110,image_s=3,s=2,sigma_t=0.05,type_w_t="G-1",window=0.1,l_0 = 1.0*(np.pi/180),m_0=1.0*(np.pi/180),dec=-50*(np.pi/180),f=1.2)
        #t.plot_ghost_mask_paper([4,5],dec=-74.66*(np.pi/180),im_s=3.5)
        #t.plot_ghost_mask_all_paper(dec=-74.66*(np.pi/180),im_s=3.5)
        #t.sky_2D(150,3,2,0.05,type_w="GT-1",avg_v=True,plot=True,mask=False,wave=None,dec=None,approx=False,window=0.2,subtract=True)
        #t.determine_flux_A2_proto_pq([4,5],A_2_min = 0.001, A_2_max = 0.5,number = 10,resolution=150,image_s=3,s=2,sigma_t=0.1,type_w_t="G-1",window=0.3,l_0 = 1.0*(np.pi/180),m_0=1.0*(np.pi/180),dec=-50*(np.pi/180),f=1.5)
        #image,l_v,m_v = t.sky_pq_2D([0,1],150,3,2,sigma = 0.05,type_w="GT-1",avg_v=False,plot=True,mask=True,label_v = False)
        #image,l_v,m_v = t.sky_pq_2D([0,2],150,3,2,sigma = 0.05,type_w="G-1",avg_v=False,plot=True,mask=True,label_v = False)
        #image,l_v,m_v = t.sky_pq_2D([0,2],150,3,2,sigma = 0.05,type_w="GT-1",avg_v=False,plot=True,mask=True,label_v = False)
        #image,l_v,m_v = t.sky_pq_2D([1,2],150,3,2,sigma = 0.05,type_w="G-1",avg_v=False,plot=True,mask=True,label_v = False)
        #image,l_v,m_v = t.sky_pq_2D([1,2],150,3,2,sigma = 0.05,type_w="GT-1",avg_v=False,plot=True,mask=True,label_v = False)
        #image,l_v,m_v = t.sky_pq_2D([0,1],150,3,2,sigma = 0.05,type_w="G-1",avg_v=False,plot=True,mask=True,label_v = False,approx=True)
        #image,l_v,m_v = t.sky_pq_2D([0,1],150,3,2,sigma = 0.05,type_w="G-1",avg_v=False,plot=True,mask=True,label_v = False,approx=True,difference=True)
        #t.create_mask([2,3],plot_v = True, dec = None, plot_markers = True)
        #t.plot_ghost_masks(6)
        #t.plot_ghost_images(6)
        
        #plt.show()
        #image,l_v,m_v = t.sky_pq_2D([1,5],250,5,2,sigma = 0.05,type_w="G-1",avg_v=False,plot=True,mask=True,label_v = True)
        #s_index = t.get_sorted_index_pq([2,3],resolution=150,image_s=4,s=2,sigma_t=0.05,type_w_t="GTR-R",window=0.3)
        freq,flux,flux2,mask,p_labels = t.determine_flux_wave_pq([4,5],f_min = 1.445,f_max = 1.9,number = 20,resolution=150,image_s=3,s=2,sigma_t=0.05,type_w_t="G-1",window=0.2,error_v=True,exper=1000,p=13)
        #t.dec_graph([4,5],dec_min=-90*(np.pi/180),dec_max=-50*(np.pi/180),npoints=100)
        t.plot_pq([4,5],parameter="freq",error_v=True,flux_type=1,p=13)
        #def plot_pq(self,baseline,type_w="G-1",parameter="freq",flux_type=1):
        #freq,flux,mask,p_labels = t.determine_flux_wave_pq([2,3],f_min = 1.2,f_max = 1.95,number = 10,resolution=150,image_s=4,s=2,sigma_t=0.05,type_w_t="GTR-R",window=0.2)
        #l_x,flux,mask,p_labels = t.determine_flux_l_pq([2,3],l_min = 1,l_max = 2,number = 100,resolution=150,image_s=4,s=3,sigma_t=0.05,type_w_t="GTR-R",window=0.3)
        #t.plot_l_pq([2,3],s_index,k=10)
        #image,l_v,m_v = t.sky_pq_2D([1,5],150,3,2,sigma = 0.05,type_w="G-1",avg_v=False,plot=True,mask=True,label_v = True)
        #freq,flux,mask,p_labels = t.determine_flux_wave_pq([2,3],f_min = 1.2,f_max = 1.95,number = 5,resolution=150,image_s=4,s=2,sigma_t=0.05,type_w_t="G-1",window=0.2)
        #for k in xrange(len(flux)):
        #    plt.plot(freq,flux[k,:])
        #    plt.hold('on')
        #plt.show()
        #p = t.create_mask([2,3],plot_v = False,dec = -70)
        #t.dec_graph([2,3],dec_min=-90*(np.pi/180),dec_max=-20*(np.pi/180),npoints=100)
        #t.dec_graph([1,5],dec_min=-90*(np.pi/180),dec_max=-20*(np.pi/180),npoints=100)
        #print "p.shape = ",p.shape
        #image,l_v,m_v = t.sky_pq_2D([2,3],150,3,2,sigma = 0.05,type_w="G-1",avg_v=False,plot=True,mask=True,dec=-20*(np.pi/180))
        #image,l_v,m_v = t.sky_pq_2D([1,5],150,3,2,sigma = 0.05,type_w="G-1",avg_v=False,plot=True,mask=True,dec=-20*(np.pi/180))
        #image,l_v,m_v = t.sky_pq_2D([1,3],150,3,2,sigma = 0.05,type_w="GT-1",avg_v=False,plot=True,mask=True)
        #image,l_v,m_v = t.sky_pq_2D([0,1],150,3,2,sigma = 0.05,type_w="G-1",avg_v=False,plot=True,mask=True)
        #image,l_v,m_v = t.sky_pq_2D([0,1],150,3,2,sigma = 0.05,type_w="RT-1",avg_v=False,plot=True,mask=True)
        #image,l_v,m_v = t.sky_pq_2D([0,1],150,3,2,sigma = 0.05,type_w="GT-1",avg_v=False,plot=True,mask=True)
        #image,l_v,m_v = t.sky_2D(150,3,2,sigma = 0.05,type_w="GT-1",avg_v=False,plot=True)
        #ghosts,baselines = t.determine_top_k_ghosts(window=0.1)
        #ghosts[:,0] = (ghosts[:,0]/0.2)*100
        #ghosts[:,1] = ghosts[:,1]*(180/np.pi)
        #ghosts[:,2] = ghosts[:,2]*(180/np.pi)
        #print "ghosts = ",ghosts
        #print "baselines = ",baselines
        #plt.plot(ghosts[0,1],ghosts[0,2]*(-1),"ms")
        #plt.plot(ghosts[1,1],ghosts[1,2]*(-1),"ms")
        #plt.plot(ghosts[2,1],ghosts[2,2]*(-1),"ms")
        #plt.plot(ghosts[5,1],ghosts[5,2]*(-1),"ms")
        #plt.plot(ghosts[7,1],ghosts[7,2]*(-1),"ms")
        #plt.plot(ghosts[9,1],ghosts[9,2]*(-1),"ms")
        #plt.show()

        #result,A_2_v = t.determine_flux_A2(type_w_t="GTR-R")
        #plt.plot(A_2_v,np.absolute(result[0,:])/A_2_v,"b")
        #plt.hold("on") 
        #plt.plot(A_2_v,np.absolute(result[1,:])/A_2_v,"r")
        #plt.plot(A_2_v,np.absolute(result[2,:])/A_2_v,"g")
        #plt.plot(A_2_v,np.absolute(result[3,:])/A_2_v,"k")
        #plt.show()
        #result,l_cor,m_cor = t.determine_flux_pos_pq([3,5],number=50,resolution=150,window=0.1,window_c=0.25) 
        
        #fig = plt.figure() 
        #cs = plt.imshow(result[0,:],cmap = "jet", extent = [l_cor[0],l_cor[-1],m_cor[0],m_cor[-1]])
        #plt.plot(l_0*(180/np.pi),m_0*(180/np.pi),"rx")  
        #fig.colorbar(cs)
        #plt.title("0")
        #plt.show()

        
        #fig=plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.set_xlabel("$l$ [degrees]")
        #ax.set_ylabel("$m$ [degrees]")
        #ax.set_zlabel("Jy")
        #ll,mm = np.meshgrid(l_cor,m_cor)
        #ax.plot_wireframe(ll,mm, result[0,:])
        #ax.plot_wireframe(ll,mm, result[0,:],rstride = 30,cstride = 5)
        #plt.show()


        #fig = plt.figure() 
        #cs = plt.imshow(result[1,:],cmap = "jet", extent = [l_cor[0],l_cor[-1],m_cor[0],m_cor[-1]])
        #plt.plot(l_0*(180/np.pi),m_0*(180/np.pi),"rx")  
        #fig.colorbar(cs)
        #plt.title("1")
        #plt.show()
      
        #fig=plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.set_xlabel("$l$ [degrees]")
        #ax.set_ylabel("$m$ [degrees]")
        #ax.set_zlabel("Jy")
        #ll,mm = np.meshgrid(l_cor,m_cor)
        #ax.plot_wireframe(ll,mm, result[6,:])
        #ax.plot_wireframe(ll,mm, result[0,:],rstride = 30,cstride = 5)
        #plt.show()

 
        #fig = plt.figure() 
        #cs = plt.imshow(result[6,:],interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],l_cor[-1],m_cor[0],m_cor[-1]])
        #plt.plot(l_0*(180/np.pi),m_0*(180/np.pi),"rx")  
        #fig.colorbar(cs)
        #plt.title("6")
        #plt.show()
        
        #fig=plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.set_xlabel("$l$ [degrees]")
        #ax.set_ylabel("$m$ [degrees]")
        #ax.set_zlabel("Jy")
        #ll,mm = np.meshgrid(l_cor,m_cor)
        #ax.plot_wireframe(ll,mm, result[6,:])
        #ax.plot_wireframe(ll,mm, result[0,:],rstride = 30,cstride = 5)
        #plt.show()
 
        #fig = plt.figure() 
        #cs = plt.imshow(result[33,:],interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],l_cor[-1],m_cor[0],m_cor[-1]])
        #plt.plot(l_0*(180/np.pi),m_0*(180/np.pi),"rx")  
        #fig.colorbar(cs)
        #plt.title("33")
        #plt.show()
        #t.plot_visibilities_pq([1,5],u=u_temp/s.ms.wave,v=v_temp/s.ms.wave,resolution=150,image_s=3,s=1)

        #t.plot_visibilities_pq([0,1],u=u_temp/s.ms.wave,v=v_temp/s.ms.wave,resolution=150,image_s=3,s=1,approx=True)
        
        # CREATING ONE IMAGE OF A BASELINE AND EXTRACTING THE FLUXES OF THE SOURCES
        #**************************************************************************
        #image,l_v,m_v = t.sky_pq_2D([3,5],250,3,2,sigma = 0.05,type_w="G-1",avg_v=False,plot=False)
        #point_sources = t.create_mask([3,5])
        #point_real,point_imag = t.extract_flux(image,l_v,m_v,0.2,point_sources,False)
        #for k in range(len(point_sources)):
        #    print "k,l, m, amp_r, amp_c =",k,point_real[k,1]*(180/np.pi),-1*point_real[k,2]*(180/np.pi),point_real[k,0],point_imag[k,0],np.sqrt(point_real[k,0]**2+point_imag[k,0]**2) 
        #**************************************************************************

        #t.plot_visibilities_pq([4,5],u=-1*u_temp/s.ms.wave,v=-1*v_temp/s.ms.wave,resolution=90,image_s=3,s=1)
        #p_temp = t.create_mask([4,5])
        #print "p_temp = ",p_temp
        #s.meqskymodel(p_temp,antenna=[4,5])

        #u_temp,v_temp,V_G_pq,V_R_pq,phi,delta_b,theta,l_cor,m_cor = t.visibilities_pq_2D(u=u_temp/s.ms.wave,v=v_temp/s.ms.wave,baseline=[0,1])

        #plt.plot(V_R_pq_s.real,"r")
        #plt.hold("on")
        #plt.plot(V_R_pq.real,"b")
        #plt.show()
        
        #plt.plot(V_R_pq_s.imag,"r")
        #plt.hold("on")
        #plt.plot(V_R_pq.imag,"b")
        #plt.show()
        
        #plt.plot(np.absolute(V_R_pq_s-V_R_pq)**2)
        #plt.show()

        #plt.plot(V_G_pq_s.real,"r")
        #plt.hold("on")
        #plt.plot(V_G_pq.real,"b")
        #plt.plot([0,len(V_G_pq)],[np.mean(V_G_pq.real),np.mean(V_G_pq.real)],'k')
        #plt.show()

        #plt.plot(V_G_pq_s.imag,"r")
        #plt.hold("on")
        #plt.plot(V_G_pq.imag,"b")
        #plt.show()

        #plt.plot(np.absolute(V_G_pq_s-V_G_pq)**2)
        #plt.show()
        
        #p_wrapper.image_settings(niter=500000,filter=None,threshold="0.00001Jy")
        #options = p_wrapper.image_advanced_settings(antenna=[4,5])
        #p_wrapper.make_image_with_mask(options=options,column="CORRECTED_DATA")
        #p_wrapper.flip_fits()
        #imager.make_image(column="CORRECTED_DATA",dirty=options,restore=options)
        #imager.make_threshold_mask(threshold=0.001) # make mask 
        #imager.make_image(column="CORRECTED_DATA",dirty=options,mask=True,restore=options2)

        #plt.plot(c_eig.R[0,1,:].real)
        #plt.hold("on")
        #plt.plot(c_eig.G[0,1,:].real)
        #plt.show()
        #plt.plot(c_eig.R[0,1,:].imag)
        #plt.hold("on")
        #plt.plot(c_eig.G[0,1,:].imag)
        #plt.show()
