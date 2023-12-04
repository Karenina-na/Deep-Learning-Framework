import sys
import datetime


# 修改日志
class Logger:
    def __init__(self):
        self.terminal = sys.stdout
        # yyyy-mm-dd-hh:mm:ss.log
        filename = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S') + ".log"
        path = "./Result/logs/"
        self.log = open(path + filename, "w")

    def __del__(self):
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger()
