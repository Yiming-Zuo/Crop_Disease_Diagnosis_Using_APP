from visualdl import LogWriter
import time
writer = LogWriter(logdir="./log_test/scalar_test/train")

writer.add_scalar(tag="acc", step=1, value=0.5678)
writer.add_scalar(tag="acc", step=2, value=0.6878)
writer.add_scalar(tag="acc", step=3, value=0.9878)

time.sleep(10000)

writer.close()