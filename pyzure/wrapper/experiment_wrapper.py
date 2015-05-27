# -*- coding: utf-8 -*-
__doc__="""
abstract class to handle just the experiment interface interaction
"""

import selenium.webdriver as wd
from auxilary import _apply_until_passes, element_rectangle,element_port
import time 
from selenium.webdriver.common.keys import Keys
from base_wrapper import base_wrapper

class experiment_wrapper(base_wrapper):
    """abstract class to handle just the experiment interface interaction"""
    def __init__(self,driver,wait_load = 3, wait_short = 1,verbose= 1):
        self.driver = driver
        self.wait_load = wait_load
        self.wait_short = wait_short
        self.verbose = verbose
    def save_experiment(self,):
        save_btn = self.driver.find_element_by_css_selector(".fxs-drawercommands-commands-global > li:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > img:nth-child(1)")
        save_btn.click()
    def run_experiment(self,):
        self.report("Running...")
        while True:
            time.sleep(self.wait_short)
            try:
                run_btn = self.driver.find_element_by_css_selector(".fxs-command-action-play")
                assert run_btn.get_attribute("aria-disabled")!="true" 
            except: continue
            run_btn.click()
            break
    def create_scoring_experiment(self,):
        self.report("Creating scoring experiment...")
        while True:
            time.sleep(self.wait_load)
            try:
                cse_btn = self.driver.find_element_by_css_selector(".fxs-drawercommands-commands-global > li:nth-child(10) > div:nth-child(1)")
                assert cse_btn.get_attribute("aria-disabled")!="true"
            except: continue
            cse_btn.click()
            break
        while True:
            try:           
                tray = self.driver.find_element_by_css_selector(".fxs-drawertray-button")
                tray.find_element_by_xpath("//img[@alt = 'Completed Progress']")
                break
            except: pass
            try:
                self.driver.find_element_by_css_selector("#bubble-close").click()
                break
            except:pass
        try:
            time.sleep(self.wait_load)
            self.driver.find_element_by_css_selector("#bubble-close").click()
        except:pass
            
    def publish_experiment(self,wait_finished = True):
        while True:
            time.sleep(self.wait_load)
            try:
                pub_btn = self.driver.find_element_by_css_selector(".fxs-command-action-publish")
                assert pub_btn.get_attribute("aria-disabled")!="true"
            except: continue
            pub_btn.click()
            break
        self.report("Publishing...")
        time.sleep(self.wait_load)
        try:
            tick =  self.driver.find_element_by_css_selector(".fxs-confirmation-buttons > li:nth-child(1) > button:nth-child(1)")
            tick.click()
        except: pass
        if wait_finished:
            api_box =_apply_until_passes(self.wait_load,self.driver.find_element_by_css_selector,".fxs-copybutton-value > input:nth-child(1)")
            api_key= api_box.get_attribute("value")

            guide_url = self.driver.find_element_by_css_selector(".even > td:nth-child(1) > a:nth-child(1)").get_attribute("href")
            self.driver.get(guide_url)
            url_box= _apply_until_passes(self.wait_load,self.driver.find_element_by_css_selector,"#requestSummary > table:nth-child(2) > tbody:nth-child(1) > tr:nth-child(2) > td:nth-child(2) > p:nth-child(1) > code:nth-child(1)")
            url = url_box.text            
            return url,api_key

    def zoom_out(self,n_times=20):
        zoomout=self.driver.find_element_by_css_selector(".xe-zoomOut")
        for i in xrange(n_times): zoomout.click()
    def getPalette(self,name,location =None):
        if location ==None:
            location = self.driver.find_element_by_css_selector(".xe-utilityPanelRoot")
        return location.find_element_by_xpath("//div[text() = '"+name+"']")
    def deploy_dataset(self,dataset_name,folder = "My Datasets"):
        datasets = self.getPalette("Saved Datasets")
        datasets.click()
        time.sleep(self.wait_short)
        folder = self.getPalette(folder)
        folder.click()
        time.sleep(self.wait_short)
        panel = self.driver.find_element_by_css_selector(".xe-utilityPanelRoot")
        dataset =panel.find_element_by_xpath("//flexfill[text() = '"+dataset_name+"']")
        chains = wd.ActionChains(self.driver)
        chains.double_click(dataset).perform()
        folder.click()
        datasets.click()
        self.report("dataset",dataset_name,"deployed")
    def deploy_model(self,model_name,model_type = "Classification"):
        ml_expander = self.getPalette("Machine Learning")
        ml_expander.click()
        time.sleep(self.wait_short)
        init_expander = self.getPalette("Initialize Model")
        init_expander.click()
        time.sleep(self.wait_short)
        type_expander =self.getPalette(model_type)
        type_expander.click()
        panel =self.driver.find_element_by_css_selector(".xe-utilityPanelRoot")
        model =panel.find_element_by_xpath("//flexfill[text() = '"+model_name+"']")
        chains = wd.ActionChains(self.driver)
        chains.double_click(model).perform()
        type_expander.click()
        init_expander.click()
        ml_expander.click()
        self.report(model_name,"deployed")	
    def deploy_model_trainer(self):
        panel = self.driver.find_element_by_css_selector(".xe-utilityPanelRoot")
        ml_expander = self.getPalette("Machine Learning")
        ml_expander.click()
        time.sleep(self.wait_short)
        ml_trainer = self.getPalette("Train")
        ml_trainer.click()
        time.sleep(self.wait_short)
        trainer =panel.find_element_by_xpath("//flexfill[text() = 'Train Model']")
        chains = wd.ActionChains(self.driver)
        chains.double_click(trainer).perform()
        ml_trainer.click()
        ml_expander.click()
        self.report("trainer deployed")
    def select_column(self,element,colname,remove_prev = 0):
        element_rectangle(element).click()
        btn_select_cols = self.driver.find_element_by_css_selector("input.datalab-gridCellPropertyItem")
        btn_select_cols.click()    
        
        col_textbox = _apply_until_passes(self.wait_load,self.driver.find_element_by_css_selector,".itemTextBoxInput")
        for i in xrange(remove_prev):
            col_textbox.send_keys(Keys.BACKSPACE)
        col_textbox.send_keys(colname+"\n")
        time.sleep(self.wait_short)
        tick_btn = self.driver.find_element_by_css_selector("flexstatic.icon")
        tick_btn.click()
        self.report(colname,"selected as target column")
    def deploy_model_scorer(self):
        panel = self.driver.find_element_by_css_selector(".xe-utilityPanelRoot")
        ml_expander = self.getPalette("Machine Learning")
        ml_expander.click()
        time.sleep(self.wait_short)
        ml_scorer = self.getPalette("Score")
        ml_scorer.click()
        time.sleep(self.wait_short)
        scorer =panel.find_element_by_xpath("//flexfill[text() = 'Score Model']")
        chains = wd.ActionChains(self.driver)
        chains.double_click(scorer).perform()
        ml_scorer.click()
        ml_expander.click()
        self.report("scorer deployed")
    def deploy_service_element(self,elem_name ="Input"):
        panel = self.driver.find_element_by_css_selector(".xe-utilityPanelRoot")
        service_expander = self.getPalette("Web Service")
        service_expander.click()
        time.sleep(self.wait_short)
        scorer =panel.find_element_by_xpath("//flexfill[text() = '"+elem_name+"']")
        chains = wd.ActionChains(self.driver)
        chains.double_click(scorer).perform()
        service_expander.click()    
        self.report("deployed service",elem_name)
    def wire_elements(self,el_from, el_to,nport_from=0,nport_to=0):
        el_port_from = element_port(el_from,nport_from)
        el_port_to = element_port(el_to,nport_to)
        chains = wd.ActionChains(self.driver)
        chains.drag_and_drop(el_port_from,el_port_to).perform() 