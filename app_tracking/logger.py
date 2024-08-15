from datetime import datetime
import os

class App_Logger:
    def __init__(self,path):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def log(self,level,log_message):
        self.now = datetime.now()
        self.date = self.now.date()
        self.level = level
        self.current_time = self.now.strftime("%H:%M:%S")
        with open(self.path,"a+") as file:
            file.write(str(self.date) + str(self.level)+"/" + str(self.current_time) + "\t\t" + log_message +"\n")
