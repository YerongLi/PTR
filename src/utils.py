# class TqdmLoggingHandler(logging.Handler):
#     def __init__(self, level=logging.NOTSET):
#         super().__init__(level)

#     def emit(self, record):
#         try:
#             msg = self.format(record)
#             tqdm.tqdm.write(msg)
#             self.flush()
#         except Exception:
#             self.handleError(record)
class progress_bar_log():
    def __init__(self, log, div=6):
        self.length = div
        self.bars = { k : None for k in range(self.length)}
        self.log = log
    
    def check(self, i, total):
        progress = int(i/total*self.length) % self.length
        if (progress in self.bars): 
                del self.bars[progress]
                self.log.info(f'{progress}/{self.length}')

