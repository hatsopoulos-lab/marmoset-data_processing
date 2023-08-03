# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 14:36:14 2021

@author: Dalton
"""
import multiprocessing as mp
import signal

class MyProcess(mp.Process):
    def __init__(self, toExecute):
        # Call Mother Class Constructor
        mp.Process.__init__(self)
        # Create shared Event for all process (__init__ is called in the main process)
        self.exit = mp.Event()
        # Set method to execute
        self.toExecute = toExecute
    def run(self):
        # Children process (in same process group as main process) will
        # receive Ctrl-C signal, this is not our logical exit procedure
        # (we use shared events tested in processing loops, and proper
        # exit).
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while not self.exit.is_set():
            self.toExecute()
        print("Process exited")
        
def manage_ctrlC(*args):
    # If you have multiple event processing processes, set each Event.
    myProcess1.exit.set()
    myProcess2.exit.set()
def toExecute1():
    while True:
        print("exec1");
def toExecute2():
    while True:
        print("exec2");
if __name__ == '__main__':
    global myProcess1, myProcess2
    # Init process (you can start multiple event processing processes)
    myProcess1 = MyProcess(toExecute1)
    myProcess1.start()
    
    myProcess2 = MyProcess(toExecute2)
    myProcess2.start()
    myProcess1.join()
    myProcess2.join()
    # Manage Ctrl_C keyboard event 
    signal.signal(signal.SIGINT, manage_ctrlC)
 
