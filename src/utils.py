import logging
log = logging.getLogger(__name__)
# log.addHandler(TqdmLoggingHandler())

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)
class bar():
    def __init__(self):
        self.length = 6
        self.bars = { k : None for k in range(self.length)}
    def check(i, total):
        progress = int(i/total*len(self.bars)) % len(self.bars)
        if (progress in self.bars): 
                del self.bars[progress]
                log.info(f'{progress}/{self.length}')

