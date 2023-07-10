import logging
import logging.handlers
import os

# 如果日志文件夹不存在，则创建
log_dir = "log-day"  # 日志存放文件夹名称
log_path = os.getcwd() + os.sep + log_dir
if not os.path.isdir(log_path):
    os.makedirs(log_path)

# logging初始化工作
logging.basicConfig()

# myapp的初始化工作
myapp = logging.getLogger('myapp')
myapp.setLevel(logging.DEBUG)

# 添加TimedRotatingFileHandler
# 定义一个1天换一次log文件的handler
# 保留3个旧log文件
timefilehandler = logging.handlers.TimedRotatingFileHandler(
    log_dir + os.sep + "sec.log",
    when='M',
    interval=1,
    backupCount=3
)
# 设置后缀名称，跟strftime的格式一样
timefilehandler.suffix = "%Y-%m-%d_%H-%M-%S.log"
# timefilehandler.suffix = "%Y-%m-%d.log"

formatter = logging.Formatter('%(asctime)s|%(name)-12s: %(levelname)-8s %(message)s')
timefilehandler.setFormatter(formatter)
myapp.addHandler(timefilehandler)

