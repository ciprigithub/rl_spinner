import numpy as np
import time
### Make sure to have CoppeliaSim's Python API available
class ParticleFilter:
    def __init__(self,init_pose):
        print("PF Starting...")
        self.laser_script=sim.getScript(1,"/Laser_Script")
        self.agent_script=sim.getScript(1,"/Agent_Script")
        self.robot_script=sim.getScript(1,"/Robot_Script")
        

        self.n_particles = 1 ## number of particles
        self.noise_v=0#0.01    ## noise for linear speed motion command
        self.noise_w=0#0.01    ## noise for angular speed motion command
        
        self.init_disp=0.05   ## particles starts in a square +-init_disp
        self.init_ang=np.pi/10 ## the orientation is randomly set +-init_ang

        # Initializing particles
        # A particle is pose + v + w + time
        self.particles = np.zeros((self.n_particles, 6))  
        
        
        self.gt=np.zeros(3)
        
        self.weights = np.ones(self.n_particles) / self.n_particles

        self.particles[:, 0] = init_pose[0]#+np.random.uniform(-self.init_disp, self.init_disp, self.n_particles)
        self.particles[:, 1] = init_pose[1]#+np.random.uniform(-self.init_disp, self.init_disp, self.n_particles)
        self.particles[:, 2] = init_pose[2]#np.random.uniform(self.init_ang,self.init_ang, self.n_particles)
        
        self.motion_history=[]
        
        ## Get v,w,time from the time map
        for i in range(self.n_particles):
            pose=[self.particles[i,0],self.particles[i,1],self.particles[i,2]]
            action = sim.callScriptFunction('query_action_from_pose',self.agent_script,pose)
            self.particles[i, 3] = 0.1  ## fixed linear 
            self.particles[i, 4] = action["angular"]  ##w
            self.particles[i, 5] = action["timestep"] ##time

        self.pc_handle = sim.getObject("/PointCloud")
        
        
        
        self.trace = sim.addDrawingObject(sim.drawing_linestrip+sim.drawing_cyclic,3,0,-1,100,[0,1,0],None,None,[0,1,0])  
        self.gt= sim.addDrawingObject(sim.drawing_linestrip+sim.drawing_cyclic,3,0,-1,100,[1,0,0],None,None,[1,0,0])  
        
        
        target_pose=[1,1,0]
        sim.callScriptFunction("place_target",self.robot_script,target_pose)
        
        self.show_cloud=True
        self.update_visualization()
        print("PF Initizalized")
        
    
    """def motion_record(self, v,w):
        now_ns = time.time_ns()      # nanosegundos desde epoch
        self.motion_history.append((now_ns, v, w))
    
    """
    def prediction(self,v,w,delta,last_w,elapsed):
        # dt=delta / 1e9    
        # delta time is now in seconds
        dt=delta
        
        #print("Particle 1: ",self.particles[0,0],self.particles[0,1],self.particles[0,2])
        
        #z = sim.callScriptFunction('laser_get_observations',self.laser_script)
        for i in range(self.n_particles):
            v_noise = v +np.random.normal(0,self.noise_v)
            w_noise = w +np.random.normal(0,self.noise_w)
            theta = self.particles[i, 2]
            """
            if (elapsed!=0):
                if (last_w!=0):
                    R=v_noise/last_w
                    self.particles[i,0]+= R * ( -np.sin(theta)  + np.sin(theta+last_w*elapsed)) 
                    self.particles[i,1]+= R * (  np.cos(theta) - np.cos(theta+last_w*elapsed))
                    self.particles[i,2]+=  last_w * elapsed
                else:
                    self.particles[i,0]+= v_noise*np.cos(theta)*elapsed
                    self.particles[i,1]+= v_noise*np.sin(theta)*elapsed
                    self.particles[i,2]+= last_w*elapsed
            
            self.particles[i,2]=(self.particles[i, 2] + np.pi) % (2*np.pi) - np.pi        
            theta = self.particles[i, 2]
            """
            if (w_noise!=0):
                R=v_noise/w_noise
                
                self.particles[i,0]= self.particles[i,0] + R * ( -np.sin(theta) + np.sin(theta+ (w_noise*dt)))
                self.particles[i,1]= self.particles[i,1] + R * (  np.cos(theta) - np.cos(theta+ (w_noise*dt))) 
                self.particles[i,2]= self.particles[i,2] + w_noise * dt
            else:
                self.particles[i,0]+= v_noise*np.cos(theta)*dt
                self.particles[i,1]+= v_noise*np.sin(theta)*dt
                self.particles[i,2]+= w_noise*dt
            
            self.particles[i,2]=(self.particles[i, 2] + np.pi) % (2*np.pi) - np.pi       
           
            ##self.update(i,np.array(z)) ## update the weigths fo the i-th particle given z
            
            ##update time map values
            pose=[self.particles[i,0],self.particles[i,1],self.particles[i,2]]
            action = sim.callScriptFunction('query_action_from_pose',self.agent_script,pose)
            #self.particles[i, 3] = 0.1  ## fixed linear 
            self.particles[i, 4] = action["angular"]  ##w
            self.particles[i, 5] = action["timestep"] ##time

        self.update_visualization()            
    
    
        
        
    """def prediction(self):
        buffer_size= len(self.motion_history)
        print("Predicting with",buffer_size,"velocity commands")
        #print(self.motion_history)
        
        for i in range(1, buffer_size):
            t_prev, v_prev, w_prev = self.motion_history[i-1]
            t_curr, v_curr, w_curr = self.motion_history[i]
            dt = t_curr - t_prev
            
            self.motion_model(v_prev,w_prev,dt)
            self.update_gt(v_prev,w_prev,dt)

        # borrar los comandos ya usados
        self.motion_history = self.motion_history[buffer_size-1:]
        self.update_visualization()

    """        
    def update(self,i,z):
        sigma=1 # sigma sensor
        pose=[self.particles[i,0],self.particles[i,1],self.particles[i,2]]
        dist = sim.callScriptFunction('laser_get_observations_from_pose',self.laser_script,pose)
            
        dif=z-np.array(dist)
        mean_squared=np.mean(dif**2)
        self.weights[i]=np.exp(-mean_squared/sigma) + 1.e-300
         
        

    def resample(self):
        pass
        
    def update_visualization(self):
        
        if (self.show_cloud):
            sim.removePointsFromPointCloud(self.pc_handle,0,None,0)
        
            pts = []
            for i in range(self.n_particles):
                x = self.particles[i,0]
                y = self.particles[i,1]
                
                pts += [x, y, 0.05]
            
            sim.insertPointsIntoPointCloud(self.pc_handle ,0, pts,[0,0,255])
            #paint the first one for instance (by now)
        sim.addDrawingObjectItem(self.trace,[self.particles[0,0],self.particles[0,1],0.1])
        
        #print("Visualization: ",self.particles[0,0],self.particles[0,1],self.particles[0,2])

    def avg_pose(self):
        #avg angular
        print(self.particles[0,0],self.particles[0,1],self.particles[0,2])
        avg_angular=np.mean(self.particles[:,4])
        avg_time=np.mean(self.particles[:,5])
        return avg_angular,avg_time
    
    def execute_robot_action(self,v_lin,v_ang,dt):
        
        
        sim.callScriptFunction('cmd_vel',self.robot_script,v_lin,v_ang)
        
            
        startTime = time.time_ns()      # nanosegundos desde epoch
        waitedTime = time.time_ns() - startTime
        while ( waitedTime < (dt * 1e9)):
            waitedTime = time.time_ns() - startTime
        #sim.callScriptFunction('cmd_vel',self.robot_script,0,0)
        
        
        return(waitedTime/1e9)


def sysCall_init():
    sim = require('sim')
    self.robot= sim.addDrawingObject(sim.drawing_linestrip+sim.drawing_cyclic,3,0,-1,100,[0,0,1],None,None,[0,0,1])  

def trace_from_robot(pose):
    sim.addDrawingObjectItem(self.robot,[pose[0],pose[1],0.1])
    
def sysCall_thread():
     
     robot_handle=sim.getObject("/base_link_visual")
     position=sim.getObjectPosition(robot_handle,-1)
     orientation=sim.getObjectOrientation(robot_handle,-1)
     
     pf=ParticleFilter([position[0],position[1],orientation[2]])        ## spreads particles around the initial pose of the robot
     
     #experiment_type="Exploitation"  ## executes the learnt policy
     #experiment_type="Moving_only"
     experiment_type="Average_FP"    ## executes the averaged particles pose
     #experiment_type="Full_FP"       ## weigth each particle and get the best or average one
     
     
     prediction_time=0
     last_angular=0
     elapsed_time=0
     while not sim.getSimulationStopping():
     
        
        position=sim.getObjectPosition(robot_handle,-1)
        orientation=sim.getObjectOrientation(robot_handle,-1)
        
        #sim.addDrawingObjectItem(pf.gt,[position[0],position[1],0.1])
     
        if experiment_type=="Exploitation":
            action = sim.callScriptFunction('query_action_from_pose',pf.agent_script,[position[0],position[1],orientation[2]])
            if action:
                pf.execute_robot_action(0.1,action["angular"],action["timestep"])    
        
        if experiment_type=="Average_FP":
            ## Move the robot and move the particles
        
            now=time.time_ns()    
            avg_ang,avg_time=pf.avg_pose()
            print(avg_ang,avg_time)
            pf.execute_robot_action(0.1,avg_ang,avg_time)
            total_time=(time.time_ns()-now)/1e9
            pf.prediction(0.1,avg_ang,total_time,0,0)
        
        if experiment_type=="Moving_only":
            
            now=time.time_ns()    
            action = sim.callScriptFunction('query_action_from_pose',pf.agent_script,[position[0],position[1],orientation[2]])
            
            if action:
                total_time=pf.execute_robot_action(0.1,action["angular"],action["timestep"])    
                elapsed_time=(time.time_ns()-now)/1e9
                
                prediction_now=time.time_ns()
                pf.prediction(0.1,action["angular"],elapsed_time,0,0)
                prediction_time=(time.time_ns()-prediction_now)/1e9
                
                #last_angular=action["angular"]
                
                #print("Prediction time: ",prediction_time)
                
                
        
            
        
                