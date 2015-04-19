# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:41:44 2015

@author: ayanami
"""
import pandas as pd

class csvReader:
    def __init__(self,filepath,cyclic = False,transformer = None):
        self.filepath = filepath
        self.current = 1
        self.isFinished = False
        self.isCyclic = cyclic
        self.columns = pd.read_csv(self.filepath,nrows = 1).columns
        self.transformer = transformer
    def _readChunk(self,chunkSize):
        try:
            dframe = pd.read_csv(self.filepath,
                                 skiprows = self.current,
                                 nrows = chunkSize,
                                 header = None,
                                 names= self.columns)
            self.current +=chunkSize
            if len(dframe)!= chunkSize:
                if self.isCyclic:
                    self.current =1
                    return pd.concat([dframe,self.readChunk(chunkSize - len(dframe))])
                else:
                    #not cyclic
                    self.finished = True
            else:#got a complete chunk
                return dframe
        except:
            if self.isCyclic:
                self.current =1
                return self.readChunk(chunkSize)
            else:#not cyclic
                self.isFinished = True
                return False
    def readSplitted(self,chunkSize):
        if self.isFinished: return False
        dframe = self._readChunk(chunkSize)
        if dframe is False: return False
        if self.transformer != None:
            return self.transformer(dframe)
        else:
            return dframe
        
        