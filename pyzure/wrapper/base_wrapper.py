from __future__ import print_function


__doc__="""
a base wrapper to handle basic interface interactions
"""

import selenium.webdriver as wd
from selenium.webdriver.support import select
import urllib2
import pandas as pd
import os
import json
from selenium.webdriver.common.keys import Keys
from auxilary import _apply_until_passes

import time




class base_wrapper:
    
    def __init__(self,driver,wait_load = 3, wait_short = 1,verbose= 1):
        self.driver = driver
        self.wait_load = wait_load
        self.wait_short = wait_short
        self.verbose = verbose

    def report(self,*args,**kwargs):
        verbosity = 0 if ('verbosity' not in kwargs) else kwargs['verbosity']
        if verbosity <= self.verbose:
            print(*args)
        

    def sign_in(self,login,password,keep_signed = False):
        self.report("signing in...")
        self.driver.get("https://studio.azureml.net/")
        elem= _apply_until_passes(self.wait_load,self.driver.find_element_by_class_name,"signInLink")    
        elem.click()
        inp_login = _apply_until_passes(self.wait_load,self.driver.find_element_by_id,"i0116")    
        inp_login.send_keys(login)
        inp_password = self.driver.find_element_by_id("i0118")
        inp_password.send_keys(password)
        if keep_signed:
            inp_keep = self.driver.find_element_by_id("idLbl_PWD_KMSI_Cb")
            inp_keep.click()
        inp_login_btn = self.driver.find_element_by_id("idSIButton9")
        inp_login_btn.click()
        time.sleep(self.wait_short)
        self.report("signed in as",login)

        
    
    def add_dataset_by_path(self,fileaddr,extension = ".csv",overwrite= False, change_name = False, new_name ="dataset.csv",wait_for_upload=True):
        if self.driver.name == 'phantomjs':
            raise "This feature is yet not available in headless mode. Sorry."
        extensions = ["",".csv",".nh.csv",".tsv",".nh.tsv",".txt",".svmlight",".arff",".zip",".RData"]
        assert extension in extensions

        fileaddr =os.path.abspath(fileaddr)
        self.report("adding dataset from",fileaddr)
        self.driver.get("https://studio.azureml.net/Home/")
        btn_new =_apply_until_passes(self.wait_load,self.driver.find_element_by_class_name,"fxs-drawertaskbar-newbutton-img")
        btn_new.click()
        dataset_tab = _apply_until_passes(self.wait_load,self.driver.find_element_by_css_selector,".Dataset > img:nth-child(1)")    
        dataset_tab.click()
        time.sleep(self.wait_short)
        fromloc_tab = self.driver.find_element_by_css_selector(".UploadDatasetMenuItem > img:nth-child(1)")    
        fromloc_tab.click()
        time.sleep(self.wait_short)
        file_inp = self.driver.find_element_by_css_selector("div.datalab-validation-wrapper:nth-child(2) > input:nth-child(3)")
        file_inp.send_keys(fileaddr)
        time.sleep(self.wait_short)
        
        name_inp = self.driver.find_element_by_css_selector("div.animation-wrapper:nth-child(2) > input:nth-child(2)")

        checkbox_overwrite = self.driver.find_element_by_css_selector(".deprecate-header > input:nth-child(1)")
        if checkbox_overwrite.is_selected(): #to unlock the input
            checkbox_overwrite.click()	

        if change_name:
            time.sleep(self.wait_short)
            name_inp.send_keys(Keys.CONTROL+'a'+Keys.NULL)            
            name_inp.send_keys(new_name)
        name = name_inp.get_attribute("value")
        
        if overwrite:
            checkbox_overwrite.click()
            switch_type =  self.driver.find_element_by_css_selector("div.datalab-validation-wrapper:nth-child(4) > select:nth-child(3)")
            select.Select(switch_type).select_by_index(extensions.index(extension))
            time.sleep(self.wait_short)

        tick_btn = self.driver.find_element_by_class_name("fx-dialog-ok")
        tick_btn.click()
        
        
        time.sleep(self.wait_short)
        failure = False
        try:
            failure = tick_btn.is_displayed()
        except:
            pass
        if failure:
            raise ValueError,"Couldnot create dataset. The cause must be that the name isn't unique or the parameters are invalid. Haven't you already uploaded it?"
                
        if wait_for_upload:
            self.report("waiting for file to upload")
            _apply_until_passes(self.wait_load,self.driver.find_element_by_css_selector,".fxs-progressbox-actionbutton > button:nth-child(1) > img:nth-child(1)")

        self.report(name,'dataset added')

    def press_delete(self):
        del_btn =self.driver.find_element_by_css_selector(".fxs-command-icon > img:nth-child(1)")
        del_btn.click
    

        

    def create_new_experiment(self,experiment_name = None):
        self.report("creating new experiment")
        self.driver.get("https://studio.azureml.net/Home/")
        btn_new =_apply_until_passes(self.wait_load,self.driver.find_element_by_class_name,"fxs-drawertaskbar-newbutton-img")
        btn_new.click()
        
        btn_blanc = _apply_until_passes(self.wait_load, self.driver.find_element_by_css_selector,".blank-icon")
        btn_blanc.click()
        exp_name = self.driver.find_element_by_css_selector("#experiment-description")
        if experiment_name != None:
            time.sleep(self.wait_short)
            exp_name.send_keys(Keys.CONTROL+'a'+Keys.NULL)
            exp_name.send_keys(experiment_name)
        else:            
            experiment_name = exp_name.get_attribute("value")
        self.report("created experiment with name:",experiment_name)
    
  
    
    def apply_service(self,data,url,api_key,withColumns=False):
        """apply published service given pandas data and authorization information. Warning: if withCollumns ==  True, an exact match of names is required."""

        inp_dict = {}
        if withColumns:
            inp_dict["ColumnNames"]= list(data.columns)

        inp_dict["Values"]= [list([str(cell) for cell in data.irow(i)]) for i in xrange(len(data))]
        
                
        data_json =  { "Inputs": {  "input1": inp_dict  }, "GlobalParameters": {}  }

        self.report("encoding data")
        body = str.encode(json.dumps(data_json))

        headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
        self.report("sending request")
        req = urllib2.Request(url, body, headers) 

        try:
            self.report("loading response")
            response = urllib2.urlopen(req)
            resp_string = response.read()
            self.report("parsing response")
            resp_dict =json.loads(resp_string)
            resp_dict =resp_dict[u'Results'][u'output1'][u'value']
            result = pd.DataFrame(resp_dict[u'Values'],columns=resp_dict[u'ColumnNames'])
            return result 
        except urllib2.HTTPError, error:
            self.report("The request failed with status code: " + str(error.code))
        
            # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
            self.report(error.info())
        
            self.report(json.loads(error.read()))
            return "request fail"

