# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:55:39 2015

@author: ayanami

Auxilary thing that does auxilary job
"""
import time
def workspace_element_selector(n_added):
    """creation stack position"""
    return ".svgRoot > g:nth-child(1) > g:nth-child("+str(n_added+1)+")"
def workspace_element(node,n_added):
    """creation stack position"""
    return node.find_element_by_css_selector(workspace_element_selector(n_added))

def element_rectangle(elem):
    return elem.find_element_by_tag_name("rect")
def element_port(elem,port_n=0):
    return elem.find_element_by_css_selector("g:nth-child("+str(port_n+1)+")")

def _apply_until_passes(wait_load,function,*args,**kwargs):
    """wait for the time interval and apply function. If an error occurs, repeat the whole thing"""
    while True:
        try:
            time.sleep(wait_load)    
            ans = function(*args,**kwargs)
            return ans
        except:pass