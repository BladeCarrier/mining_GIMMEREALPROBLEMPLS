# -*- coding: utf-8 -*-
__doc__="""
i configure some of the browsers to make them compatible with the wrapper
"""
import selenium.webdriver as wd

def phantomjs(path_to_binary):
    from selenium.webdriver.common.desired_capabilities import DesiredCapabilities    
    dcap = dict(DesiredCapabilities.PHANTOMJS)
    dcap["phantomjs.page.settings.userAgent"] = ("Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:29.0) Gecko/20100101 Firefox/29.0")

    template = {u'acceptSslCerts': True,
         u'applicationCacheEnabled': True,
         u'browserName': u'firefox',
         u'cssSelectorsEnabled': True,
         u'databaseEnabled': True,
         u'handlesAlerts': True,
         u'javascriptEnabled': True,
         u'locationContextEnabled': True,
         u'nativeEvents': False,
         u'platform': u'WINDOWS',
         u'rotatable': False,
         u'takesScreenshot': True,
         u'version': u'37.0.1',
         u'webStorageEnabled': True}
    for key in template:
        dcap[key] = template[key]
    
    driver = wd.PhantomJS(path_to_binary,desired_capabilities=dcap)
    driver.set_window_size(1124, 850)
    return  driver

def firefox_portable(path_to_binary):
    from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
    firebox_binary = FirefoxBinary(path_to_binary)
    driver = wd.Firefox(firefox_binary=firebox_binary)
    driver.set_window_size(1124, 850)
    return  driver
def firefox_default():
    driver = wd.Firefox()
    driver.set_window_size(1124, 850)
    return driver



