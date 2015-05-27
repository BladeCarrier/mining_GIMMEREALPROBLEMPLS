# -*- coding: utf-8 -*-
__doc__="""
a wrapper layer that's capable of creating and setting up a model
"""
from selenium import webdriver as wd
from base_wrapper import base_wrapper
from experiment_wrapper import experiment_wrapper
from auxilary import workspace_element, element_rectangle
import time

class model_wrapper(base_wrapper,experiment_wrapper):
    def __init__(self,driver,wait_load = 3, wait_short = 1,verbose= 1):
        self.driver = driver
        self.wait_load = wait_load
        self.wait_short = wait_short
        self.verbose = verbose
    def setup_model(self,data_name,target_colname,model_name,model_type = "Classification",dataset_folder = "My Datasets",elem_offset_y=15):
    
        self.report("deploying machinelearning model experiment")
        self.zoom_out()
        
        self._n_elems = 0
        def _inplace_elem(deployer, *args,**kwargs):
            deployer(*args,**kwargs)
            time.sleep(self.wait_short)

            self._n_elems+=1
            deployed_elem = workspace_element(self.driver,self._n_elems-1)
            draggable = deployed_elem if self.driver.name == 'phantomjs' else element_rectangle(deployed_elem)
            
            chains = wd.ActionChains(self.driver)
            chains.drag_and_drop_by_offset(draggable,0,elem_offset_y*(self._n_elems-1)).perform()
            time.sleep(self.wait_short)
            return workspace_element(self.driver,self._n_elems-1)
    
        data = _inplace_elem(self.deploy_dataset,data_name,dataset_folder)#0
    
        cmodel = _inplace_elem(self.deploy_model,model_name,model_type)#1
        
        trainer = _inplace_elem(self.deploy_model_trainer)#2
        self.select_column(trainer,target_colname)
        
        scorer = _inplace_elem(self.deploy_model_scorer)#3
    
        sinput = _inplace_elem(self.deploy_service_element,"Input")#4
    
        soutput = _inplace_elem(self.deploy_service_element,"Output")#5

        
            
        self.report("wiring elements")
        self.wire_elements(trainer,cmodel,0,0)#6
        self.wire_elements(trainer,data,2,0)#7
        self.wire_elements(scorer,data,2,0)#8
        self.wire_elements(scorer,trainer,0,3)#9
        self.wire_elements(trainer,sinput,2,0)#10
        self.wire_elements(soutput,scorer,0,3)#11 
        self._n_elems+=6
    
        self.report("experiment deployment finished")